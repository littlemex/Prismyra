"""Whether the card has room for one more copy of the weights, for the GPU tests that build their own engines."""

from __future__ import annotations

from pathlib import Path

import torch


def needed_bytes(model: str) -> int:
    """The checkpoint's weight files plus a tenth, for a local directory; for a hub id, four fifths of the card."""
    path = Path(model)
    if path.is_dir():
        return int(1.1 * sum(f.stat().st_size for f in path.glob("*.safetensors")))
    return int(0.8 * torch.cuda.get_device_properties(0).total_memory)


def no_room_reason(model: str, test_file: str) -> str | None:
    """None when another engine fits; otherwise the reason to give for skipping."""
    free, total = torch.cuda.mem_get_info()
    need = needed_bytes(model)
    if free >= need:
        return None
    return (
        f"needs {need / 2**30:.1f} GiB free for another copy of the weights and the card has {free / 2**30:.1f} of "
        f"{total / 2**30:.1f} GiB free -- an engine elsewhere in this pytest process still holds them. On a card this "
        f"size, run the file on its own: pytest -m gpu tests/{Path(test_file).name}"
    )
