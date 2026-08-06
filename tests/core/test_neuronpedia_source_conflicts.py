"""Tests for source-set collision detection and resolution.

The database is faked at the psycopg seam rather than mocked at the module's own functions: the
behaviour worth pinning is which SQL scope is used and what is done with the counts, and mocking
``describe_source_set`` would test the mock.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from interpretune.utils.neuronpedia_source_conflicts import (
    AUTOSUFFIX_MAX_ATTEMPTS,
    ConflictPolicy,
    ExplanationLossRefused,
    SourceSetConflictError,
    SourceSetOccupancy,
    describe_source_set,
    generate_autosuffix,
    render_conflict_report,
    resolve_source_set_conflict,
    suffix_source_set_id,
)

DB = "postgres://user:pw@127.0.0.1:5433/postgres"


class _FakeCursor:
    """Returns the queued counts in order, recording the SQL it was asked to run."""

    def __init__(self, counts: list[int], statements: list[tuple[str, tuple]]):
        # Shared by reference on purpose: each describe_source_set opens its own connection, and the
        # queued counts must advance across all of them rather than restarting per cursor.
        self._counts = counts
        self._statements = statements
        self.rowcount = 0

    def execute(self, sql, params=()):
        self._statements.append((" ".join(sql.split()), params))
        if sql.strip().upper().startswith("DELETE"):
            self.rowcount = self._counts.pop(0) if self._counts else 0
        return None

    def fetchone(self):
        return [self._counts.pop(0) if self._counts else 0]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConnection:
    def __init__(self, counts, statements):
        self._counts = counts
        self._statements = statements
        self.committed = False

    def cursor(self):
        return _FakeCursor(self._counts, self._statements)

    def commit(self):
        self.committed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patched_connect(counts: list[int], statements: list):
    """Patch the module's connect seam, not psycopg itself."""
    return patch(
        "interpretune.utils.neuronpedia_source_conflicts._connect",
        lambda *a, **k: _FakeConnection(counts, statements),
    )


class TestDescribe:
    def test_counts_sources_neurons_and_explanations(self) -> None:
        statements: list = []
        with _patched_connect([26, 425984, 15], statements):
            occupancy = describe_source_set(DB, model_id="m", source_set_id="s")

        assert occupancy == SourceSetOccupancy(
            model_id="m", source_set_id="s", source_count=26, neuron_count=425984, explanation_count=15
        )
        assert occupancy.occupied

    def test_neurons_are_scoped_by_source_set_name(self) -> None:
        """The `sourceSetName` scope does not require Source rows, so a half-imported set still counts.

        Verified equal to the join-through-Source form at 425,984 rows on a full 26-layer set.
        """
        statements: list = []
        with _patched_connect([0, 5, 0], statements):
            describe_source_set(DB, model_id="m", source_set_id="s")

        neuron_sql = next(sql for sql, _ in statements if '"Neuron"' in sql and "Explanation" not in sql)
        assert '"sourceSetName" = %s' in neuron_sql

    def test_sources_without_neurons_still_count_as_occupied(self) -> None:
        """A partially-imported set collides too; reporting it empty would invite the silent no-op."""
        with _patched_connect([26, 0, 0], []):
            assert describe_source_set(DB, model_id="m", source_set_id="s").occupied

    def test_empty_set_is_not_occupied(self) -> None:
        with _patched_connect([0, 0, 0], []):
            assert not describe_source_set(DB, model_id="m", source_set_id="s").occupied


class TestSuffix:
    def test_double_underscore_separator(self) -> None:
        assert (
            suffix_source_set_id("gemmascope-2-transcoder-16k", "myvariant") == "gemmascope-2-transcoder-16k__myvariant"
        )

    def test_there_is_no_default_rename_suffix(self) -> None:
        """Regression: a fixed ``DEFAULT_RENAME_SUFFIX = "hub"`` used to supply this.

        It produced ids like ``…-rte__hub``, which say nothing about when an import happened or
        which of two variants is newer, and collided with themselves on the second run. The rename
        policy now has no default at all: the caller either names a suffix or gets a timestamp.
        """
        import interpretune.utils.neuronpedia_source_conflicts as conflicts

        assert not hasattr(conflicts, "DEFAULT_RENAME_SUFFIX")
        with pytest.raises(TypeError):
            suffix_source_set_id("s")  # type: ignore[call-arg]

    @pytest.mark.parametrize("bad", ["", "has space", "/slash", "-leading-dash"])
    def test_unusable_suffixes_fail_loudly(self, bad: str) -> None:
        with pytest.raises(ValueError, match="not usable"):
            suffix_source_set_id("s", bad)


class TestResolve:
    def test_empty_target_is_a_no_op(self) -> None:
        with _patched_connect([0, 0, 0], []):
            resolution = resolve_source_set_conflict(DB, model_id="m", source_set_id="s")
        assert resolution.effective_source_set_id == "s"
        assert not resolution.renamed

    def test_error_policy_refuses_and_names_both_ways_forward(self) -> None:
        with _patched_connect([26, 425984, 0], []):
            with pytest.raises(SourceSetConflictError) as excinfo:
                resolve_source_set_conflict(DB, model_id="m", source_set_id="s", policy=ConflictPolicy.ERROR)
        message = str(excinfo.value)
        assert "SILENT NO-OP" in message
        assert "--autosuffix-on-exists" in message and "--overwrite-existing" in message

    def test_autosuffix_leaves_the_resident_set_alone(self) -> None:
        statements: list = []
        # first describe: occupied; second describe (rename target): empty
        with _patched_connect([26, 425984, 0, 0, 0, 0], statements):
            resolution = resolve_source_set_conflict(DB, model_id="m", source_set_id="s", policy=ConflictPolicy.RENAME)

        assert resolution.renamed
        assert resolution.effective_source_set_id.startswith("s__")
        # A UTC timestamp, e.g. s__20260805T174530Z -- readable and sortable, not epoch seconds.
        stamp = resolution.effective_source_set_id.removeprefix("s__")
        assert re.fullmatch(r"\d{8}T\d{6}Z", stamp), stamp
        assert not any(sql.startswith("DELETE") for sql, _ in statements), "rename must not delete anything"

    def test_autosuffix_regenerates_rather_than_refusing_on_collision(self) -> None:
        """A generated name the caller never chose is not worth failing a run over.

        First stamp collides; the retry adds microseconds and lands.
        """
        with _patched_connect([26, 425984, 0, 26, 425984, 0, 0, 0, 0], []):
            resolution = resolve_source_set_conflict(DB, model_id="m", source_set_id="s", policy=ConflictPolicy.RENAME)

        stamp = resolution.effective_source_set_id.removeprefix("s__")
        assert re.fullmatch(r"\d{8}T\d{6}\d+Z", stamp), stamp

    def test_autosuffix_gives_up_after_a_bounded_number_of_attempts(self) -> None:
        """Bounded so a stuck clock cannot spin forever; reaching it means something is wrong."""
        with _patched_connect([26, 425984, 0] + [26, 425984, 0] * AUTOSUFFIX_MAX_ATTEMPTS, []):
            with pytest.raises(SourceSetConflictError, match="should be impossible"):
                resolve_source_set_conflict(DB, model_id="m", source_set_id="s", policy=ConflictPolicy.RENAME)

    def test_explicit_suffix_is_used_verbatim(self) -> None:
        with _patched_connect([26, 425984, 0, 0, 0, 0], []):
            resolution = resolve_source_set_conflict(
                DB, model_id="m", source_set_id="s", policy=ConflictPolicy.RENAME, rename_suffix="mine"
            )
        assert resolution.effective_source_set_id == "s__mine"

    def test_explicit_suffix_collision_is_an_error_not_a_regeneration(self) -> None:
        """The caller named this set, so routing around it would ignore what they asked for."""
        with _patched_connect([26, 425984, 0, 26, 425984, 0], []):
            with pytest.raises(SourceSetConflictError, match="named explicitly"):
                resolve_source_set_conflict(
                    DB, model_id="m", source_set_id="s", policy=ConflictPolicy.RENAME, rename_suffix="mine"
                )

    def test_overwrite_refuses_when_explanations_would_cascade(self) -> None:
        """Activations regenerate from a corpus; explanations do not.

        That asymmetry is the whole rule.
        """
        with _patched_connect([26, 425984, 15], []):
            with pytest.raises(ExplanationLossRefused, match="15 explanations"):
                resolve_source_set_conflict(DB, model_id="m", source_set_id="s", policy=ConflictPolicy.OVERWRITE)

    def test_overwrite_proceeds_when_no_explanations_exist(self) -> None:
        statements: list = []
        with _patched_connect([26, 425984, 0, 425984], statements):
            resolution = resolve_source_set_conflict(
                DB, model_id="m", source_set_id="s", policy=ConflictPolicy.OVERWRITE
            )
        assert resolution.deleted_neurons == 425984
        assert resolution.effective_source_set_id == "s"
        deletes = [sql for sql, _ in statements if sql.startswith("DELETE")]
        assert any('"Neuron"' in sql for sql in deletes)
        assert any('"Source"' in sql for sql in deletes)

    def test_overwrite_with_explicit_consent_deletes_despite_explanations(self) -> None:
        with _patched_connect([26, 425984, 15, 425984], []):
            resolution = resolve_source_set_conflict(
                DB,
                model_id="m",
                source_set_id="s",
                policy=ConflictPolicy.OVERWRITE,
                allow_explanation_loss=True,
            )
        assert resolution.deleted_explanations == 15

    def test_overwrite_dry_run_changes_nothing(self) -> None:
        statements: list = []
        with _patched_connect([26, 425984, 0], statements):
            resolution = resolve_source_set_conflict(
                DB, model_id="m", source_set_id="s", policy=ConflictPolicy.OVERWRITE, dry_run=True
            )
        assert resolution.deleted_neurons == 0
        assert not any(sql.startswith("DELETE") for sql, _ in statements)


class TestReport:
    def test_explanation_warning_only_when_explanations_exist(self) -> None:
        with_explanations = render_conflict_report(
            SourceSetOccupancy("m", "s", source_count=26, neuron_count=10, explanation_count=15)
        )
        without = render_conflict_report(
            SourceSetOccupancy("m", "s", source_count=26, neuron_count=10, explanation_count=0)
        )
        assert "CASCADE" in with_explanations and "--allow-explanation-loss" in with_explanations
        assert "CASCADE" not in without


class TestPipelineIntegration:
    """The pipeline must apply the policy BEFORE generating, and must not re-key from the sidecar."""

    def test_rename_overrides_the_import_set_without_moving_the_run_directory(self, tmp_path) -> None:
        from interpretune.utils import neuronpedia_dashboard_pipeline as pipeline

        config = SimpleNamespace(
            import_to_local_db=True,
            local_db_url=DB,
            model_name="gemma-3-1b-it",
            neuronpedia_source_set_id="s",
            on_existing_source_set="rename",
            source_set_rename_suffix="myvariant",
            allow_explanation_loss=False,
            neuronpedia_import_source_set_id=None,
        )
        with _patched_connect([26, 425984, 0, 0, 0, 0], []):
            pipeline._apply_source_set_conflict_policy(config)

        assert config.neuronpedia_import_source_set_id == "s__myvariant"
        # run_directory is derived from neuronpedia_source_set_id, which must be untouched: renaming
        # it would relocate the corpus out from under an --import-only run.
        assert config.neuronpedia_source_set_id == "s"

    def test_unreachable_db_does_not_block_the_run(self, caplog) -> None:
        from interpretune.utils import neuronpedia_dashboard_pipeline as pipeline

        config = SimpleNamespace(
            import_to_local_db=True,
            local_db_url=DB,
            model_name="m",
            neuronpedia_source_set_id="s",
            on_existing_source_set="error",
            source_set_rename_suffix="myvariant",
            allow_explanation_loss=False,
            neuronpedia_import_source_set_id=None,
        )

        def _boom(*a, **k):
            raise OSError("connection refused")

        with patch("interpretune.utils.neuronpedia_source_conflicts._connect", _boom):
            pipeline._apply_source_set_conflict_policy(config)  # must not raise

        assert config.neuronpedia_import_source_set_id is None

    def test_renamed_import_ignores_the_corpus_declared_source_ids(self, tmp_path) -> None:
        """The sidecar states the corpus's ORIGINAL ids; honoring it would undo the rename."""
        from interpretune.utils import neuronpedia_dashboard_pipeline as pipeline

        run_root = tmp_path / "gemma-3-1b-it_s"
        layer_dir = run_root / "layer_0"
        layer_dir.mkdir(parents=True)
        pipeline.write_source_ids_sidecar(run_root, {0: "0-s"}, source_set_id="s")

        assert pipeline._resolve_source_id(layer_dir, 0, "s") == "0-s"
        assert (
            pipeline._resolve_source_id(layer_dir, 0, "s__myvariant", ignore_declared_source_ids=True)
            == "0-s__myvariant"
        )


class TestAutosuffixGeneration:
    def test_shape_is_readable_and_sortable(self) -> None:
        """Chosen over epoch seconds because these ids appear in URLs and in every DB row."""
        stamp = generate_autosuffix()
        assert re.fullmatch(r"\d{8}T\d{6}Z", stamp), stamp
        assert generate_autosuffix() >= stamp  # lexicographic order == chronological order

    def test_retry_adds_finer_granularity(self) -> None:
        assert len(generate_autosuffix(1)) > len(generate_autosuffix(0))

    def test_generated_suffixes_are_valid_source_set_components(self) -> None:
        """The generator must not be able to produce an id the validator would reject."""
        for attempt in (0, 1):
            assert suffix_source_set_id("s", generate_autosuffix(attempt)).startswith("s__")
