"""What every conformance run measures: one model, one prompt list, one dataset slice, one op set per gate.

The suite owns these so that two repositories running the suite measure the same thing and a divergence
between them is comparable. The target owns only how to turn them into an ``ITSessionConfig`` for its
composition (see :class:`ConformanceTarget`).
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The suite model. Small, ungated, an architecture every resolver covers, and the model core's own parity
#: fixtures use, so a number here is comparable to a number there.
MODEL_ID = "gpt2"

#: The seed component and config the datamodule side is taken from. The rte_demo configs already carry the
#: bridge-vs-nnsight data-pipeline differences (`model_input_names`, `signature_columns`), which is why the
#: target names its datamodule *flavour* rather than authoring a datamodule.
SEED_REPO = "speediedan/rte"
SEED_CONFIGS = {
    "bridge": "rte_demo.gpt2.sae_lens",
    "nnsight": "rte_demo.gpt2.nnsight+sae_lens",
}

#: Canonical capture points, spelled in the TransformerLens bridge grammar every backend accepts through
#: `names_filter`. Restricted to what the HF-side resolver covers on gpt2 today; the norm point joins when the
#: activation-point vocabulary lands.
CAPTURE_LAYER = 5
CAPTURE_POINTS = (
    f"blocks.{CAPTURE_LAYER}.hook_in",
    f"blocks.{CAPTURE_LAYER}.hook_out",
    f"blocks.{CAPTURE_LAYER}.mlp.hook_out",
    f"blocks.{CAPTURE_LAYER}.attn.hook_out",
    "unembed.hook_in",
)
#: The point interventions are applied at, and the downstream point the scope discriminator observes.
INTERVENTION_POINT = f"blocks.{CAPTURE_LAYER}.hook_in"
OBSERVE_POINT = "blocks.11.hook_out"

PROMPTS = (
    "The capital of France is",
    "When the rain stopped, the children ran to the",
)

LIMIT_BATCHES = 2
#: Rows per batch. Two, so the padded-batch cases see real padding; a single-prompt backend declares 1 on its
#: target and runs every case unpadded, because looping per row is not a transparent implementation of the
#: batched contract (left-padded rows keep their padded positions in a batched forward).
BATCH_SIZE = 2
MAX_EPOCHS = 1


@dataclass
class ConformanceInputs:
    """Everything the suite fixes, plus the one helper a target uses to build its session config."""

    model_id: str = MODEL_ID
    device_type: str = "cpu"
    precision: str = "float32"
    limit_batches: int = LIMIT_BATCHES
    batch_size: int = BATCH_SIZE
    max_epochs: int = MAX_EPOCHS
    capture_layer: int = CAPTURE_LAYER
    capture_points: Sequence[str] = CAPTURE_POINTS
    intervention_point: str = INTERVENTION_POINT
    observe_point: str = OBSERVE_POINT
    prompts: Sequence[str] = PROMPTS
    workdir: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="it_conformance_")))

    def seed_config(self, flavour: str = "bridge"):
        """The seed's ``(datamodule_cfg, module_cfg, datamodule_cls, module_cls)`` for a data-pipeline flavour.

        Cache-only after ``ensure_local_seeds``; never touches the network. The module cfg returned is the
        seed's, which a target may reuse (its adapter-agnostic fields: auto-composition, generation, HF
        loading) or replace.
        """
        from interpretune.hub.api import load as hub_load
        from it_examples.seeds import ensure_local_seeds

        ensure_local_seeds()
        try:
            key = SEED_CONFIGS[flavour]
        except KeyError:
            raise ValueError(
                f"unknown datamodule flavour {flavour!r}; expected one of {sorted(SEED_CONFIGS)}"
            ) from None
        return hub_load(SEED_REPO, key)

    def session_cfg(
        self,
        adapter_ctx: Sequence[Any],
        *,
        flavour: str = "bridge",
        module_cfg_extras: dict[str, Any] | None = None,
        prepare: Callable[[Any, Any], None] | None = None,
    ):
        """An ``ITSessionConfig`` for ``adapter_ctx`` over the suite's fixed inputs.

        ``module_cfg_extras`` are set as attributes on the seed's module cfg (an adapter's own config field,
        e.g. ``my_adapter_cfg``); ``prepare`` may edit ``(datamodule_cfg, module_cfg)`` in place for
        anything more. Optimizer fields are cleared: an analysis run configures none, and leaving the seed's
        makes ``configure_optimizers`` run for nothing.
        """
        from interpretune import ITSessionConfig

        dm_cfg, it_cfg, dm_cls, m_cls = self.seed_config(flavour)
        it_cfg.sae_cfgs = []
        it_cfg.optimizer_init = {}
        it_cfg.lr_scheduler_init = {}
        it_cfg.core_log_dir = str(self.workdir / "logs")
        dm_cfg.dataset_path = str(self.workdir / "dataset")
        dm_cfg.eval_batch_size = self.batch_size
        dm_cfg.train_batch_size = self.batch_size
        self._place(it_cfg)
        for name, value in (module_cfg_extras or {}).items():
            setattr(it_cfg, name, value)
        if prepare is not None:
            prepare(dm_cfg, it_cfg)
        return ITSessionConfig(
            adapter_ctx=tuple(adapter_ctx),
            datamodule_cfg=dm_cfg,
            module_cfg=it_cfg,
            datamodule_cls=dm_cls,
            module_cls=m_cls,
        )

    def _place(self, it_cfg: Any) -> None:
        """Put every bundled config the seed carries on the suite's device and precision.

        The seed configs are written for their own examples (some name a CUDA device); the suite fixes
        device and precision so a number from one repository is comparable to a number from another. A
        hub adapter's own config field is placed by the target's ``prepare`` hook, since only it knows the
        field's shape.
        """
        hf = getattr(it_cfg, "hf_from_pretrained_cfg", None)
        if hf is not None and isinstance(getattr(hf, "pretrained_kwargs", None), dict):
            hf.pretrained_kwargs.update(device_map=self.device_type, dtype=self.precision)
        ns = getattr(it_cfg, "nnsight_cfg", None)
        if ns is not None:
            ns.device_map = self.device_type
            ns.torch_dtype = self.precision
        tl = getattr(it_cfg, "tl_cfg", None)
        if tl is not None and hasattr(tl, "__dict__"):
            tl.__dict__.update(device=self.device_type, dtype=self.precision)

    def runner_kwargs(self) -> dict[str, Any]:
        """The runner settings every case shares.

        ``max_epochs`` is explicit because the runner's default of
        ``-1`` makes the analysis generator iterate zero epochs and yield an empty store.
        """
        return dict(
            limit_analysis_batches=self.limit_batches,
            max_epochs=self.max_epochs,
            ignore_manual=True,
            cache_dir=str(self.workdir / "cache"),
            op_output_dataset_path=str(self.workdir / "out"),
        )


@dataclass(frozen=True)
class ConformanceTarget:
    """What a repository declares about the composition it is putting under the contract.

    Everything else (inputs, oracles, case selection, the report) is the suite's.

    Args:
        composition: adapter names in any order; canonicalized by the session.
        session_cfg_factory: ``(inputs) -> ITSessionConfig``; the target's ONLY required code. The default
            builds the seed session with no extras, which is right for a bundled composition.
        forward_family: ``"hf_native"`` for a backend that executes the HuggingFace forward in place (value
            cases against the HF reference apply); anything else gets structural and causal cases only.
        load: called once before the session is built, for a hub component that must be pulled or staged
            and registered first. ``None`` for bundled adapters.
        datamodule_flavour: which seed data pipeline to start from (``"bridge"`` passes ``input`` +
            ``attention_mask``; ``"nnsight"`` passes ``input_ids``).
        batch_size: rows per batch for this target; ``1`` for a backend that takes one prompt at a time.
    """

    composition: tuple[str, ...]
    session_cfg_factory: Callable[[ConformanceInputs], Any] | None = None
    forward_family: str = "hf_native"
    load: Callable[[], Any] | None = None
    datamodule_flavour: str = "bridge"
    batch_size: int | None = None
    """Override the suite's rows-per-batch.

    A backend that takes one prompt at a time declares ``1``: every case
    then runs unpadded, and the refusal of a larger batch becomes a case of its own.
    """

    @property
    def single_prompt(self) -> bool:
        """Whether this target declared it takes one prompt at a time."""
        return self.batch_size == 1

    def build_session_cfg(self, inputs: ConformanceInputs):
        """The target's session config, from its factory or the seed default."""
        if self.session_cfg_factory is not None:
            return self.session_cfg_factory(inputs)
        return inputs.session_cfg(self.composition, flavour=self.datamodule_flavour)
