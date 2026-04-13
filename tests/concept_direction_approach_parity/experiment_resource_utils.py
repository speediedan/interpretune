from __future__ import annotations

import tempfile
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Literal, cast

import torch

from interpretune.config.nnsight import NNsightConfig
from interpretune.utils.resource_mgmt import cleanup_python_cuda, safe_clean_cuda  # – re-export
from tests import load_dotenv
from tests.analysis_resource_utils import clear_nnsight_test_state, serial_test_cleanup
from tests.configuration import config_modules
from tests.conftest import FixtPhase, session_fixture_hook_exec
from tests.core.cfg_aliases import CircuitTracerNNsightGemma2, CircuitTracerNNsightGemma3


@dataclass(frozen=True)
class ModelSpec:
    family: str
    variant: str
    model_name: str
    transcoder_set: str
    neuronpedia_model: str
    neuronpedia_set: str
    use_chat_template: bool
    hf_model_head: str | None = None


MODEL_SPECS: dict[tuple[str, str], ModelSpec] = {
    ("gemma2", "base"): ModelSpec(
        family="gemma2",
        variant="base",
        model_name="google/gemma-2-2b",
        transcoder_set="gemma",
        neuronpedia_model="gemma-2-2b",
        neuronpedia_set="gemmascope-transcoder-16k",
        use_chat_template=False,
    ),
    ("gemma2", "it"): ModelSpec(
        family="gemma2",
        variant="it",
        model_name="google/gemma-2-2b-it",
        transcoder_set="gemma",
        neuronpedia_model="gemma-2-2b",
        neuronpedia_set="gemmascope-transcoder-16k",
        use_chat_template=True,
    ),
    ("gemma3", "pt"): ModelSpec(
        family="gemma3",
        variant="pt",
        model_name="google/gemma-3-1b-pt",
        transcoder_set="mwhanna/gemma-scope-2-1b-pt/transcoder_all/width_16k_l0_small_affine",
        neuronpedia_model="gemma-3-1b",
        neuronpedia_set="gemmascope-2-res-16k",
        use_chat_template=False,
    ),
    ("gemma3", "1b_it"): ModelSpec(
        family="gemma3",
        variant="1b_it",
        model_name="google/gemma-3-1b-it",
        transcoder_set="mwhanna/gemma-scope-2-1b-it/transcoder_all/width_16k_l0_small_affine",
        neuronpedia_model="gemma-3-1b-it",
        neuronpedia_set="gemmascope-2-transcoder-16k",
        use_chat_template=True,
    ),
    ("gemma3", "4b_pt"): ModelSpec(
        family="gemma3",
        variant="4b_pt",
        model_name="google/gemma-3-4b-pt",
        transcoder_set="mwhanna/gemma-scope-2-4b-pt/transcoder_all/width_16k_l0_small_affine",
        neuronpedia_model="gemma-3-4b",
        neuronpedia_set="gemmascope-2-res-16k",
        use_chat_template=False,
        hf_model_head="transformers.Gemma3ForConditionalGeneration",
    ),
    ("gemma3", "4b_it"): ModelSpec(
        family="gemma3",
        variant="4b_it",
        model_name="google/gemma-3-4b-it",
        transcoder_set="mwhanna/gemma-scope-2-4b-it/transcoder_all/width_16k_l0_small_affine",
        neuronpedia_model="gemma-3-4b-it",
        neuronpedia_set="gemmascope-2-transcoder-16k",
        use_chat_template=True,
        hf_model_head="transformers.Gemma3ForConditionalGeneration",
    ),
    ("gemma3", "4b_262k_pt"): ModelSpec(
        family="gemma3",
        variant="4b_262k_pt",
        model_name="google/gemma-3-4b-pt",
        transcoder_set="mwhanna/gemma-scope-2-4b-pt/transcoder_all/width_262k_l0_small_affine",
        neuronpedia_model="gemma-3-4b",
        neuronpedia_set="gemmascope-2-res-262k",
        use_chat_template=False,
        hf_model_head="transformers.Gemma3ForConditionalGeneration",
    ),
    ("gemma3", "4b_262k_it"): ModelSpec(
        family="gemma3",
        variant="4b_262k_it",
        model_name="google/gemma-3-4b-it",
        transcoder_set="mwhanna/gemma-scope-2-4b-it/transcoder_all/width_262k_l0_small_affine",
        neuronpedia_model="gemma-3-4b-it",
        neuronpedia_set="gemmascope-2-transcoder-262k",
        use_chat_template=True,
        hf_model_head="transformers.Gemma3ForConditionalGeneration",
    ),
}


DebugSessionSurfacePreset = Literal["notebook_default", "parity_surface"]


def _apply_debug_session_surface_preset(
    cfg: CircuitTracerNNsightGemma2 | CircuitTracerNNsightGemma3,
    *,
    preset: DebugSessionSurfacePreset,
) -> None:
    if preset == "notebook_default":
        return
    if preset != "parity_surface":
        raise ValueError(f"Unsupported debug session surface preset: {preset}")

    nnsight_cfg = getattr(cfg, "nnsight_cfg", None)
    if nnsight_cfg is not None:
        nnsight_cfg.attn_implementation = "eager"
        nnsight_cfg.torch_dtype = "float32"

    circuit_tracer_cfg = getattr(cfg, "circuit_tracer_cfg", None)
    if circuit_tracer_cfg is not None:
        circuit_tracer_cfg.dtype = torch.float32
        circuit_tracer_cfg.analysis_target_tokens = None
        circuit_tracer_cfg.target_token_ids = None
        circuit_tracer_cfg.offload = "cpu"
        circuit_tracer_cfg.verbose = False


def create_work_root(base_dir: str | None, experiment_name: str) -> Path:
    if base_dir:
        work_root = Path(base_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        return work_root
    return Path(tempfile.mkdtemp(prefix=f"concept_dir_{experiment_name}_"))


def resolve_model_spec(
    model_family: str,
    model_variant: str,
    *,
    model_name_override: str | None = None,
    transcoder_set_override: str | None = None,
    neuronpedia_model_override: str | None = None,
    neuronpedia_set_override: str | None = None,
    use_chat_template_override: bool | None = None,
) -> ModelSpec:
    try:
        spec = MODEL_SPECS[(model_family, model_variant)]
    except KeyError as exc:
        supported = ", ".join(f"{family}:{variant}" for family, variant in sorted(MODEL_SPECS))
        raise ValueError(f"Unsupported model selection {model_family}:{model_variant}. Supported: {supported}") from exc

    return replace(
        spec,
        model_name=model_name_override or spec.model_name,
        transcoder_set=transcoder_set_override or spec.transcoder_set,
        neuronpedia_model=neuronpedia_model_override or spec.neuronpedia_model,
        neuronpedia_set=neuronpedia_set_override or spec.neuronpedia_set,
        use_chat_template=spec.use_chat_template if use_chat_template_override is None else use_chat_template_override,
    )


def build_test_cfg(
    model_family: str,
    *,
    model_name: str,
    transcoder_set: str,
    force_device: str | None = None,
    hf_model_head: str | None = None,
    batch_size: int | None = None,
    max_feature_nodes: int | None = None,
    debug_session_surface_preset: DebugSessionSurfacePreset = "notebook_default",
) -> CircuitTracerNNsightGemma2 | CircuitTracerNNsightGemma3:
    device_type = force_device or ("cuda" if torch.cuda.is_available() else "cpu")

    if model_family == "gemma2":
        cfg_gemma2: CircuitTracerNNsightGemma2 = CircuitTracerNNsightGemma2(phase="test", device_type=device_type)
        assert cfg_gemma2.circuit_tracer_cfg is not None
        assert cfg_gemma2.nnsight_cfg is not None
        cfg_gemma2.circuit_tracer_cfg.model_name = model_name
        cfg_gemma2.circuit_tracer_cfg.transcoder_set = transcoder_set
        if batch_size is not None:
            cfg_gemma2.circuit_tracer_cfg.batch_size = batch_size
        if max_feature_nodes is not None:
            cfg_gemma2.circuit_tracer_cfg.max_feature_nodes = max_feature_nodes
        cfg_gemma2.nnsight_cfg.model_name = model_name
        cfg_gemma2.nnsight_cfg.device_map = device_type
        _apply_debug_session_surface_preset(cfg_gemma2, preset=debug_session_surface_preset)
        return cfg_gemma2

    if model_family == "gemma3":
        cfg_gemma3: CircuitTracerNNsightGemma3 = CircuitTracerNNsightGemma3(
            phase="test",
            device_type=device_type,
        )
        assert cfg_gemma3.circuit_tracer_cfg is not None
        cfg_gemma3.circuit_tracer_cfg.model_name = model_name
        cfg_gemma3.circuit_tracer_cfg.transcoder_set = transcoder_set
        if batch_size is not None:
            cfg_gemma3.circuit_tracer_cfg.batch_size = batch_size
        if max_feature_nodes is not None:
            cfg_gemma3.circuit_tracer_cfg.max_feature_nodes = max_feature_nodes
        nnsight_kwargs: dict[str, Any] = {
            "model_name": model_name,
            "device_map": device_type,
            "torch_dtype": "float32",
            "dispatch": True,
            "tokenizer_kwargs": {"padding_side": "left", "add_bos_token": True},
        }
        cfg_gemma3.nnsight_cfg = NNsightConfig(**nnsight_kwargs)
        _apply_debug_session_surface_preset(cfg_gemma3, preset=debug_session_surface_preset)
        return cfg_gemma3

    raise ValueError(f"Unsupported model family: {model_family}")


@contextmanager
def experiment_session(
    work_root: Path,
    run_name: str,
    *,
    model_family: str,
    model_name: str,
    transcoder_set: str,
    force_device: str | None = None,
    use_cuda_cleanup: bool = True,
    hf_model_head: str | None = None,
    batch_size: int | None = None,
    max_feature_nodes: int | None = None,
    debug_session_surface_preset: DebugSessionSurfacePreset = "notebook_default",
) -> Iterator[tuple[Any, Any, Any]]:
    session_dir = work_root / run_name
    session_dir.mkdir(parents=True, exist_ok=True)

    clear_nnsight_test_state(None)
    cleanup_python_cuda()
    load_dotenv()

    cfg = build_test_cfg(
        model_family,
        model_name=model_name,
        transcoder_set=transcoder_set,
        force_device=force_device,
        hf_model_head=hf_model_head,
        batch_size=batch_size,
        max_feature_nodes=max_feature_nodes,
        debug_session_surface_preset=debug_session_surface_preset,
    )
    it_session = config_modules(cfg, run_name, {}, session_dir, {}, False)
    session_fixture_hook_exec(it_session, cast(FixtPhase, FixtPhase.setup))

    module = it_session.module
    assert module is not None
    replacement_model = cast(Any, module).replacement_model
    tokenizer = replacement_model.tokenizer

    cuda_target = module if hasattr(module, "to") else getattr(module, "model", None)
    cuda_cleanup = (
        safe_clean_cuda(cuda_target)
        if use_cuda_cleanup and torch.cuda.is_available() and cuda_target is not None
        else nullcontext()
    )

    try:
        with serial_test_cleanup(it_session, module, replacement_model, clear_cuda=not use_cuda_cleanup):
            with cuda_cleanup:
                yield it_session, module, tokenizer
    finally:
        clear_nnsight_test_state(it_session)
        cleanup_python_cuda()


def tensor_to_cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().cpu().to(torch.float32)


def feature_ids_to_tuples(feature_ids: Any) -> list[tuple[int, ...]]:
    return [tuple(feature.tolist()) for feature in feature_ids]


def scalar_tensor_list(values: list[float] | tuple[float, ...], *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(list(values), dtype=dtype)
