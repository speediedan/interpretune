from __future__ import annotations

import gc
import tempfile
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, cast

import torch

from interpretune.config.nnsight import NNsightConfig
from interpretune.utils.resource_mgmt import cleanup_python_cuda, safe_clean_cuda  # noqa: F401 – re-export
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
    ("gemma3", "it"): ModelSpec(
        family="gemma3",
        variant="it",
        model_name="google/gemma-3-1b-it",
        transcoder_set="mwhanna/gemma-scope-2-1b-it/transcoder_all/width_16k_l0_small_affine",
        neuronpedia_model="gemma-3-1b-it",
        neuronpedia_set="gemmascope-2-res-16k",
        use_chat_template=True,
    ),
}


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
) -> CircuitTracerNNsightGemma2 | CircuitTracerNNsightGemma3:
    device_type = force_device or ("cuda" if torch.cuda.is_available() else "cpu")

    if model_family == "gemma2":
        cfg = CircuitTracerNNsightGemma2(phase="test", device_type=device_type)
        assert cfg.circuit_tracer_cfg is not None
        assert cfg.nnsight_cfg is not None
        cfg.circuit_tracer_cfg.model_name = model_name
        cfg.circuit_tracer_cfg.transcoder_set = transcoder_set
        cfg.nnsight_cfg.model_name = model_name
        cfg.nnsight_cfg.device_map = device_type
        return cfg

    if model_family == "gemma3":
        cfg = CircuitTracerNNsightGemma3(phase="test", device_type=device_type)
        assert cfg.circuit_tracer_cfg is not None
        cfg.circuit_tracer_cfg.model_name = model_name
        cfg.circuit_tracer_cfg.transcoder_set = transcoder_set
        cfg.nnsight_cfg = NNsightConfig(
            model_name=model_name,
            device_map=device_type,
            torch_dtype="float32",
            dispatch=True,
            tokenizer_kwargs={"padding_side": "left", "add_bos_token": True},
        )
        return cfg

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
