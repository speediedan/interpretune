"""A compiled op cache is served only when it holds exactly the ops its sources declare.

The cache key covers the sources' content, not the process that compiled them. A process that could not resolve some ops
(an environment missing a dependency) used to save the rest under the same name, and every later session was served that
subset as the whole set.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import interpretune.analysis.backends.ops as tensor_ops
import interpretune.analysis.ops.compiler.cache_manager as cache_manager_module
from interpretune.analysis.ops.compiler.load_policy import IT_STRICT_OP_LOAD_ENV_VAR, OpLoadError
from interpretune.analysis.ops.dispatcher import AnalysisOpDispatcher

OPS_YAML = """
first_op:
  description: first
  implementation: tests.core.test_analysis_ops_base.op_impl_test
  aliases: [first_alias]
  input_schema: {}
  output_schema: {}
second-op:
  description: second
  implementation: tests.core.test_analysis_ops_base.op_impl_test
  required_ops: [first_op]
  input_schema: {}
  output_schema: {}
composite_operations:
  both_ops:
    composition: first_op.second-op
"""
DECLARED = {"first_op", "second_op", "both_ops"}


@pytest.fixture
def cache_dir(tmp_path) -> Path:
    path = tmp_path / "op_cache"
    path.mkdir()
    return path


@pytest.fixture
def op_dir(tmp_path) -> Path:
    path = tmp_path / "ops"
    path.mkdir()
    (path / "ops.yaml").write_text(OPS_YAML, encoding="utf-8")
    return path


def _dispatcher(cache_dir, *op_dirs) -> AnalysisOpDispatcher:
    dispatcher = AnalysisOpDispatcher(yaml_paths=list(op_dirs), enable_hub_ops=False)
    dispatcher._cache_manager.cache_dir = cache_dir
    return dispatcher


def _compiled(cache_dir, *op_dirs) -> AnalysisOpDispatcher:
    dispatcher = _dispatcher(cache_dir, *op_dirs)
    dispatcher.load_definitions()
    dispatcher.ensure_compiled()
    return dispatcher


def _canonical(dispatcher) -> set[str]:
    return {op_def.name for op_def in dispatcher._op_definitions.values()}


def _cache_files(cache_dir) -> list[Path]:
    return list(cache_dir.glob("op_definitions_*.py"))


def _cache_mismatch_warnings(recwarn) -> list[str]:
    return [str(w.message) for w in recwarn if "compiled op cache does not hold" in str(w.message)]


def _without_op(source: str, name: str) -> str:
    kept = [line for line in source.splitlines() if not line.startswith(f'    "{name}": ')]
    assert len(kept) == len(source.splitlines()) - 1, "the planted defect must remove exactly one op"
    return "\n".join(kept)


def test_an_intact_cache_is_served(cache_dir, op_dir, recwarn):
    """Control: the check must not turn a good cache into a recompile on every load."""
    _compiled(cache_dir, op_dir)
    assert len(_cache_files(cache_dir)) == 1

    warm = _dispatcher(cache_dir, op_dir)
    warm.load_definitions()

    assert warm._compiled, "a complete cache should be a hit, with nothing left to compile"
    assert DECLARED <= _canonical(warm)
    assert not _cache_mismatch_warnings(recwarn)


@pytest.mark.parametrize("removed", ["first_op", "second_op", "both_ops"])
def test_a_cache_missing_an_op_is_compiled_again(cache_dir, op_dir, removed):
    _compiled(cache_dir, op_dir)
    (cache_file,) = _cache_files(cache_dir)
    complete = cache_file.read_text(encoding="utf-8")
    cache_file.write_text(_without_op(complete, removed), encoding="utf-8")

    with pytest.warns(UserWarning, match=rf"compiled op cache does not hold.*missing \['{removed}'\]"):
        repaired = _compiled(cache_dir, op_dir)

    assert removed in _canonical(repaired)
    assert cache_file.read_text(encoding="utf-8") == complete, "the recompiled set should replace the subset"


def test_a_cache_holding_an_undeclared_op_is_compiled_again(cache_dir, op_dir):
    _compiled(cache_dir, op_dir)
    (cache_file,) = _cache_files(cache_dir)
    source = cache_file.read_text(encoding="utf-8")
    (first,) = [line for line in source.splitlines() if line.startswith('    "first_op": ')]
    planted = first.replace("first_op", "planted_op").replace("first_alias", "planted_alias")
    cache_file.write_text(source.replace(first, f"{first}\n{planted}"), encoding="utf-8")

    with pytest.warns(UserWarning, match=r"not declared by any source \['planted_op'\]"):
        repaired = _compiled(cache_dir, op_dir)

    assert "planted_op" not in _canonical(repaired)


def test_the_bundled_ops_round_trip_through_the_cache(cache_dir, recwarn):
    """Every bundled file, its tensor ops and its composites included, declares exactly what compiles."""
    cold = _compiled(cache_dir)
    assert len(_cache_files(cache_dir)) == 1

    warm = _dispatcher(cache_dir)
    warm.load_definitions()

    assert warm._compiled
    assert _canonical(warm) == _canonical(cold)
    assert not _cache_mismatch_warnings(recwarn)


class TestAnIncompleteCompileIsNotCached:
    @staticmethod
    def _without_venngram(monkeypatch):
        def missing():
            raise ModuleNotFoundError("No module named 'venngram'")

        monkeypatch.setattr(tensor_ops, "_venngram_ops", missing)

    def test_a_missing_dependency_leaves_no_cache_for_the_next_session(self, cache_dir, op_dir, monkeypatch):
        """The measured defect: the first op with `required_ops` raised, and the rest of its file went with it, logged
        at debug; what was left was saved as the whole set."""
        with monkeypatch.context() as broken:
            self._without_venngram(broken)
            with pytest.warns(UserWarning, match=r"Skipping 'second-op' in .*No module named 'venngram'"):
                partial = _compiled(cache_dir, op_dir)
        assert "first_op" in _canonical(partial), "only the op that failed is dropped, not the rest of its file"
        assert "second_op" not in _canonical(partial)
        assert not _cache_files(cache_dir)

        assert DECLARED <= _canonical(_compiled(cache_dir, op_dir))
        assert len(_cache_files(cache_dir)) == 1

    def test_strict_loading_raises_on_it(self, cache_dir, op_dir, monkeypatch):
        self._without_venngram(monkeypatch)
        monkeypatch.setenv(IT_STRICT_OP_LOAD_ENV_VAR, "1")
        with pytest.raises(OpLoadError, match="venngram"):
            _compiled(cache_dir, op_dir)

    def test_an_unreadable_file_is_reported_and_not_cached(self, cache_dir, op_dir):
        (op_dir / "broken.yaml").write_text("broken_op: [unclosed\n", encoding="utf-8")
        with pytest.warns(UserWarning, match=r"Failed to load op definitions from .*broken\.yaml"):
            dispatcher = _compiled(cache_dir, op_dir)
        assert DECLARED <= _canonical(dispatcher)
        assert not _cache_files(cache_dir)

    def test_an_unresolved_required_op_is_not_cached(self, cache_dir, op_dir):
        path = op_dir / "ops.yaml"
        path.write_text(
            path.read_text(encoding="utf-8").replace("required_ops: [first_op]", "required_ops: [absent_op]"),
            encoding="utf-8",
        )
        with pytest.warns(UserWarning, match="Required operation 'absent_op' not found"):
            _compiled(cache_dir, op_dir)
        assert not _cache_files(cache_dir)

    def test_a_skipped_collection_does_not_stop_caching(self, cache_dir, op_dir, recwarn):
        """A collection outside its window declares nothing here, so the set without it is the complete set."""
        (op_dir / "future.yaml").write_text(
            "collection:\n  name: future\n  version: 1.0.0\n  requires:\n    interpretune: '>=999'\n"
            "future_op:\n  implementation: tests.core.test_analysis_ops_base.op_impl_test\n"
            "  input_schema: {}\n  output_schema: {}\n",
            encoding="utf-8",
        )
        _compiled(cache_dir, op_dir)
        assert [w for w in recwarn if "Skipping op collection 'future'" in str(w.message)]
        assert len(_cache_files(cache_dir)) == 1

        warm = _dispatcher(cache_dir, op_dir)
        warm.load_definitions()
        assert warm._compiled
        assert not _cache_mismatch_warnings(recwarn)


class TestTheCacheIsReplacedWhole:
    def test_an_interrupted_write_leaves_the_previous_file(self, cache_dir, op_dir, monkeypatch):
        dispatcher = _compiled(cache_dir, op_dir)
        (cache_file,) = _cache_files(cache_dir)
        complete = cache_file.read_text(encoding="utf-8")
        write_text = Path.write_text

        def interrupted(self, data, *args, **kwargs):
            write_text(self, data[: len(data) // 2], *args, **kwargs)
            raise KeyboardInterrupt

        monkeypatch.setattr(Path, "write_text", interrupted)
        with pytest.raises(KeyboardInterrupt):
            dispatcher._cache_manager.save_cache(dispatcher._op_definitions)

        assert cache_file.read_text(encoding="utf-8") == complete
        assert [path.name for path in cache_dir.iterdir() if path.suffix == ".tmp"] == []

    def test_a_file_another_process_holds_open_is_reported(self, cache_dir, op_dir, monkeypatch):
        dispatcher = _compiled(cache_dir, op_dir)

        def denied(*args):
            raise PermissionError("in use")

        monkeypatch.setattr(cache_manager_module.os, "replace", denied)
        with pytest.warns(UserWarning, match=re.escape("Could not write the compiled op cache")):
            dispatcher._cache_manager.save_cache(dispatcher._op_definitions)
        assert [path.name for path in cache_dir.iterdir() if path.suffix == ".tmp"] == []
