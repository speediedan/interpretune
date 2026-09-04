"""The library-independent HuggingFace forward reference.

Every value-level conformance case for the ``hf_native`` family compares against tensors taken straight off
the HF module by plain PyTorch hooks. Nothing under test participates in producing them, which is what
makes agreement evidence about the forward rather than about the pair.

Points are addressed by TransformerLens bridge names and resolved to a module path and io slot through the
same resolver the nnsight backend uses. When the activation-point vocabulary lands this becomes its first
consumer and gains a per-point width check; until then it covers what the resolver covers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from interpretune.analysis.backends.hook_mapping import HookNameResolver


def _unwrap(output: Any) -> torch.Tensor:
    """A block returns a tuple on transformers < 5 and a tensor on 5.x; the hidden state is element 0 either
    way."""
    return output[0] if isinstance(output, tuple) else output


def _rewrap(output: Any, replaced: torch.Tensor) -> Any:
    return (replaced,) + tuple(output[1:]) if isinstance(output, tuple) else replaced


@dataclass(frozen=True)
class ResolvedPoint:
    """A point name with the HF module path and io slot it resolved to."""

    name: str
    module_path: str
    io: str  # "input" | "output"


class HFReference:
    """One HF model, loaded once, hooked per call.

    CPU and float32 unless told otherwise.
    """

    def __init__(self, model_id: str, *, dtype: torch.dtype = torch.float32, device: str = "cpu") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model: Any = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        self.model = model.to(torch.device(device)).eval()
        self.device = device
        self.architecture = type(self.model).__name__
        self._resolver = HookNameResolver(self.architecture)

    # -- addressing -------------------------------------------------------------------------------

    def resolve(self, point: str) -> ResolvedPoint:
        """Resolve a TL-bridge-named point to the HF module and io slot, or raise with the resolver's reason."""
        path, io = self._resolver.resolve(point)
        return ResolvedPoint(point, path, io)

    def module_for(self, point: str) -> torch.nn.Module:
        """The HF submodule a point resolves to."""
        return self.model.get_submodule(self.resolve(point).module_path)

    # -- forward -----------------------------------------------------------------------------------

    def logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """A plain forward's logits, ``[batch, pos, vocab]``."""
        with torch.no_grad():
            return self._forward(input_ids, attention_mask).logits.detach().clone()

    def capture(
        self,
        input_ids: torch.Tensor,
        points: Sequence[str],
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Activations at ``points`` (each ``[batch, pos, ...]``) plus ``"logits"``, off the module by hook."""
        captured: dict[str, torch.Tensor] = {}
        handles = []
        for point in points:
            resolved = self.resolve(point)
            module = self.model.get_submodule(resolved.module_path)
            if resolved.io == "input":

                def pre_hook(_m, args, _name=point):
                    captured[_name] = args[0].detach().clone()

                handles.append(module.register_forward_pre_hook(pre_hook))
            else:

                def post_hook(_m, _args, output, _name=point):
                    captured[_name] = _unwrap(output).detach().clone()

                handles.append(module.register_forward_hook(post_hook))
        try:
            with torch.no_grad():
                out = self._forward(input_ids, attention_mask)
        finally:
            for h in handles:
                h.remove()
        missing = [p for p in points if p not in captured]
        assert not missing, f"hooks never fired for {missing}; the reference would be vacuous"
        captured["logits"] = out.logits.detach().clone()
        return captured

    def steered(
        self,
        input_ids: torch.Tensor,
        point: str,
        edit: Callable[[torch.Tensor], torch.Tensor],
        *,
        scope: str = "last_token",
        observe: Sequence[str] = (),
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Apply ``edit`` to the tensor at ``point`` for the positions ``scope`` selects, and observe downstream.

        The reference implementation of BOTH scopes, so the discriminator is validated against it before
        it is trusted on any backend. ``edit`` receives the selected region (``[batch, d]`` for last-token,
        ``[batch, pos, d]`` for all positions) and returns the edited region.
        """
        resolved = self.resolve(point)
        module = self.model.get_submodule(resolved.module_path)

        def apply(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.clone()
            if scope == "last_token":
                tensor[:, -1, ...] = edit(tensor[:, -1, ...])
            elif scope == "all_positions":
                tensor = edit(tensor)
            else:
                raise ValueError(f"unknown scope {scope!r}")
            return tensor

        handles = []
        if resolved.io == "input":
            handles.append(module.register_forward_pre_hook(lambda _m, args: (apply(args[0]),) + tuple(args[1:])))
        else:
            handles.append(module.register_forward_hook(lambda _m, _a, out: _rewrap(out, apply(_unwrap(out)))))

        captured: dict[str, torch.Tensor] = {}

        def _observe_input(name: str):
            def hook(_m: Any, args: Any) -> None:
                captured[name] = args[0].detach().clone()

            return hook

        def _observe_output(name: str):
            def hook(_m: Any, _args: Any, out: Any) -> None:
                captured[name] = _unwrap(out).detach().clone()

            return hook

        for name in observe:
            r = self.resolve(name)
            m = self.model.get_submodule(r.module_path)
            if r.io == "input":
                handles.append(m.register_forward_pre_hook(_observe_input(name)))
            else:
                handles.append(m.register_forward_hook(_observe_output(name)))
        try:
            with torch.no_grad():
                out = self._forward(input_ids, attention_mask)
        finally:
            for h in handles:
                h.remove()
        captured["logits"] = out.logits.detach().clone()
        return captured

    # -- internals ---------------------------------------------------------------------------------

    def _forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
        kwargs: dict[str, Any] = {"input_ids": input_ids.to(self.device)}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask.to(self.device)
        return self.model(**kwargs)


def last_real_position(attention_mask: torch.Tensor | None, seq_len: int) -> torch.Tensor | int:
    """Under left padding every row's last real token is the final position; under right padding it is per row."""
    if attention_mask is None:
        return seq_len - 1
    return attention_mask.long().cumsum(-1).argmax(-1)
