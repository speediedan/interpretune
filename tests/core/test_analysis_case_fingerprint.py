from __future__ import annotations

import dataclasses
import enum

import pytest
import torch

from interpretune.runners.analysis import (
    analysis_case_fingerprint,
    analysis_cfg_fingerprint,
    prune_generator_cache,
)
from interpretune.analysis.ops.base import AnalysisOp, ColCfg, OpSchema
from interpretune.config import AnalysisCfg
from interpretune.testing.conformance.inputs import (
    ConformanceInputs,
    ConformanceTarget,
    shared_generator_cache,
    target_cache_part,
)


def _base_kwargs(**overrides):
    kwargs = dict(target_part="target|flavour|batch", op="model_fwd_intervention", package_version="v1")
    kwargs.update(overrides)
    return kwargs


class TestCaseDiscrimination:
    """The #618 failure, as a unit test: same op, different run_inputs must key apart."""

    def test_stable(self):
        kwargs = _base_kwargs(run_inputs={"mode": "add", "scale": 1.0})
        assert analysis_case_fingerprint(**kwargs) == analysis_case_fingerprint(**kwargs)

    def test_intervention_axes_discriminate(self):
        base = dict(target_part="t", op="model_fwd_intervention", package_version="v1")
        keys = {
            analysis_case_fingerprint(
                **base, run_inputs={"mode": mode, "scale": scale, "scope": scope, "vector": vector}
            )
            for mode in ("add", "clamp")
            for scale in (0.0, 1.0)
            for scope in ("all", "single")
            for vector in ("v1", "v2")
        }
        assert len(keys) == 2 * 2 * 2 * 2

    def test_tensor_values_matter(self):
        kwargs = dict(target_part="t", op="op", package_version="v1")
        same_a = analysis_case_fingerprint(**kwargs, run_inputs={"v": torch.ones(4)})
        same_b = analysis_case_fingerprint(**kwargs, run_inputs={"v": torch.ones(4)})
        different = analysis_case_fingerprint(**kwargs, run_inputs={"v": torch.zeros(4)})
        assert same_a == same_b
        assert same_a != different

    def test_names_filter_matters(self):
        kwargs = dict(target_part="t", op="op", run_inputs={}, package_version="v1")
        assert analysis_case_fingerprint(**kwargs, names_filter=["a"]) != analysis_case_fingerprint(
            **kwargs, names_filter=["b"]
        )

    def test_target_and_version_matter(self):
        assert analysis_case_fingerprint(**_base_kwargs()) != analysis_case_fingerprint(
            **_base_kwargs(target_part="other")
        )
        assert analysis_case_fingerprint(**_base_kwargs()) != analysis_case_fingerprint(
            **_base_kwargs(package_version="v2")
        )


class TestNormalization:
    def test_op_objects_key_by_name_and_definition(self):
        schema = OpSchema(col=ColCfg(datasets_dtype="float32"))

        def make_op(description):
            return AnalysisOp(name="op", description=description, output_schema=schema)

        kwargs = dict(target_part="t", run_inputs={}, names_filter=None, package_version="v1")
        assert analysis_case_fingerprint(**kwargs, op=make_op("same")) == analysis_case_fingerprint(
            **kwargs, op=make_op("same")
        )
        assert analysis_case_fingerprint(**kwargs, op=make_op("same")) != analysis_case_fingerprint(
            **kwargs, op=make_op("changed")
        )

    def test_composite_ops(self):
        kwargs = dict(target_part="t", run_inputs={}, names_filter=None, package_version="v1")
        assert analysis_case_fingerprint(**kwargs, op=["a", "b"]) != analysis_case_fingerprint(**kwargs, op=["a", "c"])

    def test_lambda_refused_by_name(self):
        with pytest.raises(TypeError, match="no stable identity"):
            analysis_case_fingerprint(**_base_kwargs(op="op", run_inputs={"fn": lambda x: x}))

    def test_unknown_type_refused_by_name(self):
        class Opaque:
            pass

        with pytest.raises(TypeError, match="no stable normalization.*Opaque"):
            analysis_case_fingerprint(**_base_kwargs(run_inputs={"x": Opaque()}))

    def test_dataclass_and_enum_inputs(self):
        @dataclasses.dataclass
        class Spec:
            mode: str
            scale: float

        class Mode(enum.Enum):
            ADD = "add"

        kwargs = dict(target_part="t", op="op", package_version="v1")
        key = analysis_case_fingerprint(**kwargs, run_inputs={"spec": Spec("add", 1.0), "mode": Mode.ADD})
        assert key == analysis_case_fingerprint(**kwargs, run_inputs={"spec": Spec("add", 1.0), "mode": Mode.ADD})
        assert key != analysis_case_fingerprint(**kwargs, run_inputs={"spec": Spec("add", 2.0), "mode": Mode.ADD})


class TestCfgWrapper:
    def test_wrapper_reads_cfg(self):
        cfg = AnalysisCfg(target_op="model_fwd_intervention", run_inputs={"mode": "add"}, names_filter=["h"])
        assert analysis_cfg_fingerprint(cfg, target_part="t", package_version="v1") == analysis_case_fingerprint(
            target_part="t",
            op=cfg.op if cfg.op is not None else cfg.target_op,
            run_inputs={"mode": "add", "__latent_targets__": cfg.latent_analysis_targets},
            names_filter=["h"],
            package_version="v1",
            extra_case_part={"step_fn": cfg.step_fn, "ignore_manual": cfg.ignore_manual},
        )


class TestPruneGeneratorCache:
    def test_evicts_oldest_first_over_cap(self, tmp_path):
        stems = []
        for i, size in enumerate((100, 200, 300)):
            stem = f"fp{i}"
            stems.append(stem)
            path = tmp_path / f"{stem}-abc.arrow"
            path.write_bytes(b"x" * size)
            import os
            import time

            mtime = time.time() - (3 - i)
            os.utime(path, (mtime, mtime))
        stats = prune_generator_cache(tmp_path, max_bytes=350)
        assert stats["evicted_bytes"] == 300
        assert stats["evicted_stems"] == 2
        assert stats["kept_stems"] == 1
        assert (tmp_path / "fp2-abc.arrow").exists()
        assert not (tmp_path / "fp0-abc.arrow").exists()

    def test_missing_dir_noop(self, tmp_path):
        assert prune_generator_cache(tmp_path / "nope", max_bytes=10) == {
            "kept_stems": 0,
            "evicted_stems": 0,
            "evicted_bytes": 0,
        }


class TestTargetPart:
    def _target(self, **overrides):
        kwargs = dict(composition=("core",))
        kwargs.update(overrides)
        return ConformanceTarget(**kwargs)

    def test_stable_and_sorted_composition(self):
        inputs = ConformanceInputs()
        target = self._target()
        assert target_cache_part(target, inputs) == target_cache_part(target, inputs)
        assert target_cache_part(self._target(composition=("b", "a")), inputs) == target_cache_part(
            self._target(composition=("a", "b")), inputs
        )

    def test_discriminates(self):
        inputs = ConformanceInputs()
        assert target_cache_part(self._target(), inputs) != target_cache_part(
            self._target(), ConformanceInputs(precision="bfloat16")
        )
        assert target_cache_part(self._target(), inputs) != target_cache_part(
            self._target(datamodule_flavour="bridge"), inputs
        )

    def test_workdir_excluded(self):
        inputs = ConformanceInputs()
        inputs._ensure_workdir()
        assert target_cache_part(self._target(), inputs) == target_cache_part(self._target(), ConformanceInputs())
        inputs.cleanup()


class TestSharedCacheEnv:
    def test_unset_is_noop(self, monkeypatch):
        monkeypatch.delenv("IT_CONFORMANCE_GENERATOR_CACHE_DIR", raising=False)
        assert shared_generator_cache() == (None, 0)

    def test_dir_and_cap(self, monkeypatch, tmp_path):
        monkeypatch.setenv("IT_CONFORMANCE_GENERATOR_CACHE_DIR", str(tmp_path))
        assert shared_generator_cache() == (tmp_path, 40_000_000_000)
        monkeypatch.setenv("IT_CONFORMANCE_GENERATOR_CACHE_MAX_BYTES", "123")
        assert shared_generator_cache() == (tmp_path, 123)

    def test_garbage_cap_refused(self, monkeypatch, tmp_path):
        monkeypatch.setenv("IT_CONFORMANCE_GENERATOR_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("IT_CONFORMANCE_GENERATOR_CACHE_MAX_BYTES", "lots")
        with pytest.raises(ValueError, match="not an integer"):
            shared_generator_cache()


class TestRunInjection:
    def test_per_case_key_set_on_opt_in(self, monkeypatch, tmp_path):
        from unittest.mock import MagicMock

        from interpretune.testing.conformance.session import ConformanceSession

        monkeypatch.setenv("IT_CONFORMANCE_GENERATOR_CACHE_DIR", str(tmp_path))
        runner = MagicMock()
        session = ConformanceSession(
            target=ConformanceTarget(composition=("core",)),
            inputs=ConformanceInputs(),
            session=None,
            runner=runner,
            capabilities=None,
        )
        cfg = AnalysisCfg(target_op="model_fwd_intervention", run_inputs={"mode": "add"})
        session.run(cfg)
        key = runner.run_cfg.dataset_fingerprint
        assert isinstance(key, str) and len(key) == 32
        runner.run_analysis.assert_called_once_with(analysis_cfgs=cfg)
        cfg2 = AnalysisCfg(target_op="model_fwd_intervention", run_inputs={"mode": "clamp"})
        session.run(cfg2)
        assert runner.run_cfg.dataset_fingerprint != key

    def test_no_opt_in_leaves_config_alone(self, monkeypatch):
        from unittest.mock import MagicMock

        from interpretune.testing.conformance.session import ConformanceSession

        monkeypatch.delenv("IT_CONFORMANCE_GENERATOR_CACHE_DIR", raising=False)
        runner = MagicMock()
        runner.run_cfg.dataset_fingerprint = None
        session = ConformanceSession(
            target=ConformanceTarget(composition=("core",)),
            inputs=ConformanceInputs(),
            session=None,
            runner=runner,
            capabilities=None,
        )
        session.run(AnalysisCfg(target_op="model_fwd_intervention"))
        assert runner.run_cfg.dataset_fingerprint is None
