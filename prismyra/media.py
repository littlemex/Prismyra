"""Images and video as part of the context.

This is where the fork pays best. A frame costs the vision tower once and then sits in the context's keys and values
like any other token, so twenty questions about one image pay for encoding it once, and a video -- thousands of tokens
-- amortises further still. Nothing about the fork itself changes: vision tokens are read by every branch and written
by none, which is the asymmetry the whole design rests on.

Two things do change, and both are handled here.

**The placeholders.** The model reads an image where its prompt holds a run of image-pad tokens, and how long that run
has to be depends on the image's resolution. The processor expands one placeholder into the right number, so the text
is assembled with a placeholder and handed to the processor rather than tokenised directly.

**The positions.** With media present the model uses a three-axis rotary scheme whose text axis no longer counts
tokens: an image occupies one position per grid cell, not one per token. The model works out the difference while
reading the context and records it, and a branch continues from there -- so the offset is carried on the `Prefill` and
added in the branch pass. Getting it wrong does not raise. Every branch simply reads the context from the wrong place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

#: What a caller writes to put an image or a clip in the context. The processor replaces the pad token with as many
#: copies as the resolution needs, so one of these stands for one image however large it is.
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


@dataclass
class Encoded:
    """One context, ready for the model: its tokens and whatever media travels with them."""

    input_ids: torch.Tensor
    tokens: int
    #: `pixel_values`, `image_grid_thw` and the rest, passed to the backbone unchanged. Empty for text.
    media: dict[str, Any] = field(default_factory=dict)

    @property
    def has_media(self) -> bool:
        return bool(self.media)


def encode(
    context: str,
    images: list | None,
    videos: list | None,
    processor,
    tokenizer,
    device: str,
) -> Encoded:
    """Turn a context and its media into what the backbone takes.

    Text with no media goes through the tokenizer, exactly as before -- the processor is not merely slower here, it
    would
    change the token sequence the measured figures were taken on.
    """
    if not images and not videos:
        ids = tokenizer(context, add_special_tokens=True)["input_ids"]
        return Encoded(input_ids=torch.tensor([ids], device=device), tokens=len(ids))

    if processor is None:
        raise RuntimeError(
            "images or video were passed but this model has no processor; only a multimodal checkpoint can read them"
        )

    # Placeholders first, in the order the caller gave them, then the text. Leading rather than trailing because a
    # question about a picture reads better after the picture, and because the branch's tokens must come last.
    prefix = IMAGE_PLACEHOLDER * len(images or []) + VIDEO_PLACEHOLDER * len(videos or [])
    batch = processor(
        text=[prefix + context],
        images=list(images) if images else None,
        videos=list(videos) if videos else None,
        return_tensors="pt",
    )
    ids = batch["input_ids"]
    # `attention_mask` is dropped on purpose: this is one unpadded sequence, so the mask is all ones and passing it
    # makes
    # the framework materialise a mask the fast attention kernel cannot take.
    media = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
        if key not in ("input_ids", "attention_mask")
    }
    return Encoded(input_ids=ids.to(device), tokens=int(ids.shape[-1]), media=media)


def position_offset(backbone, tokens: int) -> int:
    """How far along the model thinks the context reached, which is not how many tokens it held.

    With media present the model's three-axis positions advance by an image's grid rather than by its token count, and
    it
    records the difference as `rope_deltas` when it reads the context. A branch continues from `tokens + delta`. Without
    media the delta is absent and the answer is just the token count.
    """
    deltas = getattr(backbone, "rope_deltas", None)
    if deltas is None:
        return tokens
    if torch.is_tensor(deltas):
        return tokens + int(deltas.flatten()[0].item())
    return tokens + int(deltas)


# --------------------------------------------------------------------------- bytes to frames
#: Frames sampled from a clip, evenly across its length. More is not better past a point: each frame costs tokens in the
#: context, and the context is the expensive half.
DEFAULT_VIDEO_FRAMES = 16


def decode_image(data: bytes):
    """Image bytes to something the processor takes. Converted to RGB, because a PNG may arrive with an alpha channel
    and the vision tower expects three."""
    import io

    from PIL import Image

    return Image.open(io.BytesIO(data)).convert("RGB")


def decode_video(data: bytes, frames: int = DEFAULT_VIDEO_FRAMES):
    """Video bytes to evenly spaced frames.

    Decoding is done here rather than handed to the framework because which decoder the framework reaches for depends on
    what happens to be installed -- its default is one that often is not -- and a missing decoder should be a sentence
    saying so rather than an import error from three layers down.
    """
    import tempfile

    import numpy as np

    with tempfile.NamedTemporaryFile(suffix=".video") as handle:
        handle.write(data)
        handle.flush()
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError(
                "reading a video file needs opencv: pip install opencv-python-headless. Frames passed directly as "
                "arrays or images need no decoder."
            ) from e

        capture = cv2.VideoCapture(handle.name)
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if total <= 0:
                raise RuntimeError("that video has no readable frames; the container or codec may be unsupported")
            wanted = {round(float(i)) for i in np.linspace(0, total - 1, min(frames, total))}
            picked, index = [], 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index in wanted:
                    picked.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                index += 1
        finally:
            capture.release()

    if not picked:
        raise RuntimeError("that video decoded to no frames")
    return np.stack(picked)
