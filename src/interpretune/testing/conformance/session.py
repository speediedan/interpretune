"""One composed session per target class, and the runner passes every case reads from."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from interpretune.analysis.backends import ModuleCapabilities, get_model_backend, get_module_capabilities

from .inputs import ConformanceInputs, ConformanceTarget

_OPS_DIR = Path(__file__).parent


def register_conformance_ops() -> None:
    """Make the suite's op collection visible to the dispatcher, once per process."""
    from interpretune.analysis.ops.dispatcher import DISPATCHER

    if _OPS_DIR not in DISPATCHER.yaml_paths:
        DISPATCHER.yaml_paths.append(_OPS_DIR)
        DISPATCHER.reload_definitions()


@dataclass
class ConformanceSession:
    """The live objects a case needs, built once per target class."""

    target: ConformanceTarget
    inputs: ConformanceInputs
    session: Any
    runner: Any
    capabilities: ModuleCapabilities
    batches: list[dict[str, torch.Tensor]] = field(default_factory=list)

    @property
    def module(self):
        """The composed module."""
        return self.session.module

    @property
    def backend(self):
        """The model backend attached to the composed module, or ``None``."""
        return get_model_backend(self.module)

    @property
    def backend_name(self) -> str:
        """The backend's class name, for refusal messages."""
        return type(self.backend).__name__

    @property
    def family(self) -> str:
        """The target's declared forward family."""
        return self.target.forward_family

    def run(self, analysis_cfg):
        """Run one op through the runner and return its store.

        Every case goes through here.
        """
        return self.runner.run_analysis(analysis_cfgs=analysis_cfg)

    def batch_inputs(self, index: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The ids and attention mask the runner fed for batch ``index``."""
        batch = self.batches[index]
        ids = batch.get("input", batch.get("input_ids"))
        if ids is None:
            raise KeyError(f"batch {index} carries neither 'input' nor 'input_ids'")
        return ids, batch.get("attention_mask")


def _require_suite_dependencies() -> None:
    """Name the suite's own runtime dependencies before a missing one presents as setup errors.

    The seed datamodule the suite runs over reaches ``evaluate`` and, through the metric script it downloads,
    ``sklearn``; both are in interpretune's ``conformance`` extra. Checked up front, by name, because a clean
    adopter install otherwise fails inside dataset preparation with an error that reads as a broken suite.
    """
    import importlib.util

    missing = [name for name in ("evaluate", "sklearn") if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(
            f"the conformance suite needs {missing} (its session runs over the rte seed datamodule); install "
            "interpretune with the `conformance` extra: pip install 'interpretune[conformance]'"
        )


def build_conformance_session(target: ConformanceTarget, inputs: ConformanceInputs) -> ConformanceSession:
    """Load the component if the target says so, compose the session, run init, snapshot the batches."""
    from interpretune import AnalysisRunner, ITSession

    _require_suite_dependencies()
    register_conformance_ops()
    if target.batch_size is not None and target.batch_size != inputs.batch_size:
        inputs = replace(inputs, batch_size=target.batch_size)
    if target.load is not None:
        target.load()
    session = ITSession(target.build_session_cfg(inputs))
    runner = AnalysisRunner(run_cfg=dict(it_session=session, **inputs.runner_kwargs()))
    caps = get_module_capabilities(session.module)
    # The dataloader is deterministic and the runner reads it in order, so the first N batches here are the
    # batches every store's rows came from; cases needing the raw inputs (the HF reference) use these.
    batches: list[dict[str, torch.Tensor]] = []
    datamodule: Any = session.datamodule
    for i, batch in enumerate(datamodule.test_dataloader()):
        if i >= inputs.limit_batches:
            break
        batches.append({k: v for k, v in batch.items() if isinstance(v, torch.Tensor)})
    return ConformanceSession(
        target=target, inputs=inputs, session=session, runner=runner, capabilities=caps, batches=batches
    )


def tokenized_prompts(hf_tokenizer, prompts: Sequence[str]) -> list[torch.Tensor]:
    """Each prompt as its own unpadded ``[1, seq]`` id tensor: the inputs for the tight value cases."""
    return [hf_tokenizer(p, return_tensors="pt")["input_ids"] for p in prompts]
