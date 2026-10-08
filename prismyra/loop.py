"""Looping the later layers at the read-out position: an experiment, not a feature.

After the normal pass over a question, the read-out position's final hidden state (before the final norm) is fed back
into layer `k` and layers `k..L` are run again at that position, one or more times. Only the read-out position is
looped. Everything it attends to -- the context, and the question's earlier tokens -- must look exactly as it did in the
first pass, and the looped position must not be appended to any layer's state a second time.

The fork already gives that. A branch pass starts from the snapshot taken when the cache held exactly the context
(`fork.restore_and_fork`): every gated-delta layer's recurrent and convolution state goes back to the end of the
context, which is the state *before* the question's tokens, and every attention layer's length goes back to the end of
the context, so the branch's tokens are written into the same slots again rather than after the last pass's. So a loop
is one more branch pass from the snapshot in which a single value is replaced: the input of layer `k` at the read-out
position. Positions before it cannot see the replacement (the model is causal), so they recompute exactly the first
pass's keys, values and states, and the read-out position at layers `k..L` sees the same context it saw the first time.

That is the same computation as running only the read-out position through `k..L`, at a higher price: the question's
other tokens and layers `1..k-1` are recomputed each loop. The time measured here is therefore an upper bound on what a
loop needs; the lower bound is about `(L - k + 1) / L` of one single-token pass.

`alpha` blends rather than replaces: the injected state is `h_in + alpha * (h_final - h_in)`, where `h_in` is what
layer `k` received at that position in the first pass. `alpha = 1` feeds the final state back as it is; `alpha = 0`
changes nothing, which is the check that the machinery itself is exact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .fork import branch_ids
from .readout import plan, score
from .schema import Question


def decoder_parts(backbone, layers: int):
    """The text decoder's layer list and its final norm, found by shape rather than by attribute path.

    The path differs between the text-only and the multimodal classes of the same family, so it is searched for: the
    one `ModuleList` holding exactly `layers` modules, and the `norm` beside it.
    """
    for name, module in backbone.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) == layers and name.endswith("layers"):
            parent = backbone.get_submodule(name.rsplit(".", 1)[0]) if "." in name else backbone
            norm = getattr(parent, "norm", None)
            if norm is None:
                raise RuntimeError(f"found the layers at {name!r} but no final norm beside them")
            return module, norm
    raise RuntimeError(f"no ModuleList of {layers} decoder layers in the backbone")


@dataclass
class Looped:
    """Probabilities per (k, alpha, loop). Loop 0 is the plain branch read, identical for every k and alpha."""

    options: list[str]
    base: list[float]
    loops: dict = field(default_factory=dict)
    base_ms: float = 0.0
    loop_ms: dict = field(default_factory=dict)


class _Hooks:
    """Capture layer inputs and the final norm's input at one position; optionally replace one layer's input there."""

    def __init__(self, layers, norm, capture_at: list[int]):
        self.layers, self.norm = layers, norm
        self.capture_at = capture_at
        self.row = 0
        self.position = 0
        self.inputs: dict[int, torch.Tensor] = {}
        self.final: torch.Tensor | None = None
        self.replace: tuple[int, torch.Tensor] | None = None
        #: Off while the context is read: the hooks are about the branch pass, and the context pass may be shorter
        #: than the read-out position.
        self.active = False
        self.handles = []
        for k in capture_at:
            self.handles.append(layers[k - 1].register_forward_pre_hook(self._pre(k), with_kwargs=True))
        self.handles.append(norm.register_forward_pre_hook(self._norm, with_kwargs=True))

    def _pre(self, k: int):
        def hook(module, args, kwargs):
            if not self.active:
                return None
            hs = args[0] if args else kwargs["hidden_states"]
            if self.replace is not None and self.replace[0] == k:
                hs = hs.clone()
                hs[self.row, self.position] = self.replace[1].to(hs.dtype)
                if args:
                    args = (hs, *args[1:])
                else:
                    kwargs = {**kwargs, "hidden_states": hs}
            self.inputs[k] = hs[self.row, self.position].detach().float().clone()
            return args, kwargs

        return hook

    def _norm(self, module, args, kwargs):
        if not self.active:
            return None
        hs = args[0] if args else kwargs["hidden_states"]
        self.final = hs[self.row, self.position].detach().float().clone()
        return None

    def close(self):
        for h in self.handles:
            h.remove()


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def loop_readout(
    engine, context: str, question: Question, ks: list[int], alphas: list[float], loops: int, time_it: bool = True
) -> Looped:
    """Read `question` about `context` once through the fork, then loop layers `k..L` at the read-out position.

    `ks` are 1-based layer numbers of this checkpoint. Returns the plain read and, for every (k, alpha), the
    probabilities after each of `loops` extra loops.
    """
    import time

    decoder = getattr(engine.config, "text_config", engine.config)
    layers, norm = decoder_parts(engine.backbone, decoder.num_hidden_layers)
    planned = plan(question, engine.tokenizer)
    width = engine._width_for([planned])
    position = len(branch_ids(planned.text, engine.tokenizer)) - 1
    device = engine.torch_device

    def read(hidden_normed) -> list[float]:
        (p,) = engine.heads.apply(
            hidden_normed, [question.options], score(hidden_normed, engine.unembedding, [planned.token_ids], None)
        )
        return p.float().tolist()

    hooks = _Hooks(layers, norm, sorted(set(ks)))
    hooks.position = position
    try:
        with engine._lock, torch.inference_mode(), engine.open_context(context) as opened:
            prefill = opened._prefill
            hooks.active = True
            _sync(device)
            t = time.perf_counter()
            hidden = engine._branch(prefill, [planned.text], 1, width)
            _sync(device)
            out = Looped(options=list(question.options), base=read(hidden), base_ms=(time.perf_counter() - t) * 1000)
            first_in = {k: v.clone() for k, v in hooks.inputs.items()}
            first_final = hooks.final.clone()
            for k in ks:
                for alpha in alphas:
                    h_final = first_final
                    for n in range(1, loops + 1):
                        inject = first_in[k] + alpha * (h_final - first_in[k])
                        hooks.replace = (k, inject)
                        _sync(device)
                        t = time.perf_counter()
                        hidden = engine._branch(prefill, [planned.text], 1, width)
                        _sync(device)
                        ms = (time.perf_counter() - t) * 1000
                        hooks.replace = None
                        out.loops[(k, alpha, n)] = read(hidden)
                        out.loop_ms.setdefault((k, alpha, n), ms)
                        h_final = hooks.final.clone()
    finally:
        hooks.close()
    return out


def loop_one_pass(
    engine, context: str, question: Question, ks: list[int], alphas: list[float], loops: int
) -> Looped:
    """The same loop on the one-pass read (context and question as one sequence), which is the path a single `ask`
    takes. There is no snapshot to return to here, so each loop reads the whole sequence again from an empty cache with
    the one replacement at the last position: the same computation as the forked loop, at the price of re-reading the
    context. Exists to separate the loop's effect from the difference between the two paths, which on this checkpoint
    moves some short questions' probabilities by more than the loop does.

    Eager throughout (no recorded one-pass buckets), so the base read here is the eager read rather than a replay.
    """
    import time

    from .media import encode

    decoder = getattr(engine.config, "text_config", engine.config)
    layers, norm = decoder_parts(engine.backbone, decoder.num_hidden_layers)
    planned = plan(question, engine.tokenizer)
    suffix = branch_ids(planned.text, engine.tokenizer)
    device = engine.torch_device

    def read(hidden_normed) -> list[float]:
        (p,) = engine.heads.apply(
            hidden_normed, [question.options], score(hidden_normed, engine.unembedding, [planned.token_ids], None)
        )
        return p.float().tolist()

    hooks = _Hooks(layers, norm, sorted(set(ks)))
    try:
        with engine._lock, torch.inference_mode():
            encoded = encode(context, None, None, engine.processor, engine.tokenizer, engine.device)
            ids = torch.cat([encoded.input_ids, torch.tensor([suffix], device=engine.device)], dim=1)
            hooks.position = ids.shape[1] - 1
            hooks.active = True
            _sync(device)
            t = time.perf_counter()
            hidden = engine._read_one_pass(ids)
            _sync(device)
            out = Looped(options=list(question.options), base=read(hidden), base_ms=(time.perf_counter() - t) * 1000)
            first_in = {k: v.clone() for k, v in hooks.inputs.items()}
            first_final = hooks.final.clone()
            for k in ks:
                for alpha in alphas:
                    h_final = first_final
                    for n in range(1, loops + 1):
                        hooks.replace = (k, first_in[k] + alpha * (h_final - first_in[k]))
                        _sync(device)
                        t = time.perf_counter()
                        hidden = engine._read_one_pass(ids)
                        _sync(device)
                        ms = (time.perf_counter() - t) * 1000
                        hooks.replace = None
                        out.loops[(k, alpha, n)] = read(hidden)
                        out.loop_ms.setdefault((k, alpha, n), ms)
                        h_final = hooks.final.clone()
    finally:
        hooks.close()
    return out
