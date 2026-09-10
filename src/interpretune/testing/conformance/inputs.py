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
    # gemma-3-1b-it with circuit-tracer over the nnsight backend: the analysis-backend cases' composition
    "circuit_tracer": "rte_demo.gemma3.circuit_tracer.neuronpedia",
}
#: The adapter-free flavour: the seed's STANDALONE datamodule entry plus a module config built here from
#: core classes only. What a hub adapter starts from, because the bridge and nnsight seed configs carry their
#: adapters' config classes and need those adapters installed to hydrate.
SEED_DATAMODULE = "rte_boolq"

#: Canonical capture points, spelled in the TransformerLens bridge grammar every backend accepts through
#: `names_filter`. The norm's module output is a point; its derived tensors (`hook_normalized`, `hook_scale`)
#: are not, because the reference captures what a module emits and a derived tensor is computed.
CAPTURE_LAYER = 5
CAPTURE_POINTS = (
    f"blocks.{CAPTURE_LAYER}.hook_in",
    f"blocks.{CAPTURE_LAYER}.hook_out",
    f"blocks.{CAPTURE_LAYER}.ln2.hook_out",
    f"blocks.{CAPTURE_LAYER}.mlp.hook_out",
    f"blocks.{CAPTURE_LAYER}.attn.hook_out",
    "unembed.hook_in",
)


@dataclass(frozen=True)
class LatentModelSpec:
    """One pretrained latent model the suite attaches, named the way sae_lens names a release and an id, and the
    model it fits: a spec is attached only to a session over that model."""

    release: str
    sae_id: str
    model_id: str = MODEL_ID


#: The latent model the LATENT_MODELS and GRADIENTS cases run over: one gpt2 residual SAE at the first block,
#: from the release core's own adapter tests load. A target that declares LATENT_MODELS attaches
#: ``inputs.latent_models`` in its session config (the seed path does so whenever the module config carries
#: ``sae_cfgs``); a declaration with no handle attached fails those cases by name rather than skipping them.
LATENT_MODELS = (LatentModelSpec(release="gpt2-small-res-jb", sae_id="blocks.0.hook_resid_pre"),)
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
    latent_models: Sequence[LatentModelSpec] = LATENT_MODELS
    attribution_prompt: str | None = None
    """The prompt the analysis-backend cases attribute; ``None`` means the suite carries none for this model and
    those cases skip with that reason (a suite input gap, not the backend's)."""
    attribution_top_n: int = 4
    attribution_scale_factor: float = 2.0
    workdir: Path | None = None
    """Where a session's dataset, logs, op cache and store outputs go.

    Created on first use, not at construction,
    since an inputs object is often built at import time for a class whose cases may never run; removed at process
    exit and by :meth:`cleanup`, because a run's cache and outputs reach tens of gigabytes and a directory per run that
    nothing removes fills a shared host (measured: 277 of them, 451 GB, in five days).
    """
    supplied_extras: dict[str, Any] = field(default_factory=dict)
    """Every ``module_cfg_extras`` entry the last ``session_cfg`` call set, so a coherence case can check that each
    reached the composed config as a declared field with the same object, rather than as a stray attribute."""

    def latent_models_for_model(self) -> list[LatentModelSpec]:
        """The latent specs that fit ``model_id``."""
        return [spec for spec in self.latent_models if spec.model_id == self.model_id]

    def _ensure_workdir(self) -> Path:
        """The working directory, created on first use and registered for removal at process exit."""
        if self.workdir is None:
            import atexit
            import shutil

            self.workdir = Path(tempfile.mkdtemp(prefix="it_conformance_"))
            atexit.register(shutil.rmtree, str(self.workdir), ignore_errors=True)
        return self.workdir

    def cleanup(self) -> None:
        """Remove the working directory now, if one was created; the class-scoped session fixture calls this."""
        import shutil

        if self.workdir is not None:
            shutil.rmtree(self.workdir, ignore_errors=True)
            self.workdir = None

    def seed_config(self, flavour: str = "hf"):
        """The seed's ``(datamodule_cfg, module_cfg, datamodule_cls, module_cls)`` for a data-pipeline flavour.

        Cache-only after ``ensure_local_seeds``; never touches the network. The module cfg returned is the
        seed's, which a target may reuse (its adapter-agnostic fields: auto-composition, generation, HF
        loading) or replace.
        """
        from interpretune.hub.api import load as hub_load
        from it_examples.seeds import ensure_local_seeds

        ensure_local_seeds()
        if flavour == "hf":
            return self._adapter_free_seed()
        try:
            key = SEED_CONFIGS[flavour]
        except KeyError:
            raise ValueError(
                f"unknown datamodule flavour {flavour!r}; expected 'hf' or one of {sorted(SEED_CONFIGS)}"
            ) from None
        return hub_load(SEED_REPO, key)

    def _adapter_free_seed(self):
        """The standalone datamodule entry, pointed at the suite model, plus a core-only module config.

        Mirrors what the bridge seed config carries on the data side (the attention mask as a model input and a
        signature column, left padding, a BOS token, the pad id) without any adapter's config class, so it hydrates on a
        bare core install.
        """
        from interpretune import HFGenerationConfig, ITConfig
        from interpretune.config.mixins import HFFromPretrainedConfig
        from interpretune.config.shared import AutoCompConfig
        from interpretune.hub.api import load_datamodule
        from it_examples.experiments.rte_boolq import (
            RTEBoolqDataModule,
            RTEBoolqEntailmentMapping,
            RTEBoolqGenerativeClassificationConfig,
            RTEBoolqModule,
        )

        dm_cfg, _dm_cls = load_datamodule(SEED_REPO, SEED_DATAMODULE)
        dm_cfg.model_name_or_path = self.model_id
        dm_cfg.os_env_model_auth_key = None
        dm_cfg.tokenizer_kwargs = {
            "model_input_names": ["input", "attention_mask"],
            "padding_side": "left",
            "add_bos_token": True,
        }
        dm_cfg.tokenizer_id_overrides = {"pad_token_id": 50256}
        dm_cfg.signature_columns = ["input", "attention_mask", "labels"]
        dm_cfg.enable_datasets_cache = True
        dm_cfg.prepare_data_map_cfg = {"batched": True}
        it_cfg = ITConfig(
            model_name_or_path=self.model_id,
            task_name="rte",
            auto_comp_cfg=AutoCompConfig(module_cfg_name="RTEBoolqConfig", module_cfg_mixin=RTEBoolqEntailmentMapping),
            hf_from_pretrained_cfg=HFFromPretrainedConfig(
                pretrained_kwargs={"device_map": self.device_type, "dtype": self.precision},
                model_head="transformers.GPT2LMHeadModel",
            ),
            generative_step_cfg=RTEBoolqGenerativeClassificationConfig(
                enabled=True, lm_generation_cfg=HFGenerationConfig(model_config={"max_new_tokens": 1})
            ),
        )
        return dm_cfg, it_cfg, RTEBoolqDataModule, RTEBoolqModule

    def session_cfg(
        self,
        adapter_ctx: Sequence[Any],
        *,
        flavour: str = "hf",
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
        self._attach_latent_models(it_cfg)
        it_cfg.optimizer_init = {}
        it_cfg.lr_scheduler_init = {}
        it_cfg.core_log_dir = str(self._ensure_workdir() / "logs")
        dm_cfg.dataset_path = str(self._ensure_workdir() / "dataset")
        dm_cfg.eval_batch_size = self.batch_size
        dm_cfg.train_batch_size = self.batch_size
        self._place(it_cfg)
        self.supplied_extras = dict(module_cfg_extras or {})
        for name, value in self.supplied_extras.items():
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

    def _attach_latent_models(self, it_cfg: Any) -> None:
        """Replace the seed's latent models with the suite's, on the suite's device and precision.

        Only a module config carrying ``sae_cfgs`` (the sae_lens adapter's field) can attach one. The adapter-free
        config has no such field and its backend declares no LATENT_MODELS, so nothing is attached and the gated
        cases skip as undeclared. The import is deferred so this module stays importable on a bare core install.
        """
        if not hasattr(it_cfg, "sae_cfgs"):
            return
        from interpretune.adapters.sae_lens.config import SAELensFromPretrainedConfig

        it_cfg.sae_cfgs = [
            SAELensFromPretrainedConfig(
                release=spec.release, sae_id=spec.sae_id, device=self.device_type, dtype=self.precision
            )
            for spec in self.latent_models_for_model()
        ]

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
        ct = getattr(it_cfg, "circuit_tracer_cfg", None)
        if ct is not None:
            import torch

            # the suite fixes precision; the seed example uploads to neuronpedia, which a conformance run never does
            ct.dtype = getattr(torch, self.precision)
            for name, value in (("use_neuronpedia", False), ("verbose", False)):
                if hasattr(ct, name):
                    setattr(ct, name, value)
            np_cfg = getattr(it_cfg, "neuronpedia_cfg", None)
            if np_cfg is not None and hasattr(np_cfg, "enabled"):
                np_cfg.enabled = False

    def runner_kwargs(self) -> dict[str, Any]:
        """The runner settings every case shares.

        ``max_epochs`` is explicit because the runner's default of
        ``-1`` makes the analysis generator iterate zero epochs and yield an empty store.
        """
        return dict(
            limit_analysis_batches=self.limit_batches,
            max_epochs=self.max_epochs,
            ignore_manual=True,
            cache_dir=str(self._ensure_workdir() / "cache"),
            op_output_dataset_path=str(self._ensure_workdir() / "out"),
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
        datamodule_flavour: which seed data pipeline to start from. ``"hf"`` (the default) is adapter-free and
            hydrates on a bare core install; ``"bridge"`` and ``"nnsight"`` are the bundled adapters' seed
            configs and need those adapters installed.
        batch_size: rows per batch for this target; ``1`` for a backend that takes one prompt at a time.
    """

    composition: tuple[str, ...]
    session_cfg_factory: Callable[[ConformanceInputs], Any] | None = None
    forward_family: str = "hf_native"
    load: Callable[[], Any] | None = None
    module_cfg_extras: dict[str, Any] | None = None
    """Settings the default factory supplies on the seed module config, each a field the composed config class
    declares; the always-on composition case checks every one reached the composed config as the same object."""
    datamodule_flavour: str = "hf"
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
        return inputs.session_cfg(
            self.composition, flavour=self.datamodule_flavour, module_cfg_extras=self.module_cfg_extras
        )
