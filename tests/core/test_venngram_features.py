"""Feature targets and table-sourced values (#887, slice 3b): the two feature paths and the transform pipeline."""

import pytest
import torch

pytest.importorskip("venngram", reason="the venngram extra is not installed")

from interpretune.testing.venngram_books import treatment_book
from interpretune.analysis.backends.capabilities import (
    FeatureTargetSupport,
    InterventionSupport,
    PositionScope,
)
from interpretune.analysis.backends.interventions import apply_intervention
from interpretune.analysis.venngram import (
    check_recipe,
    load_recipe,
    recipe_feature_interventions,
    recipe_interventions,
    require_supported,
    support_mapping,
    venngram_feature_subsets,
    venngram_feature_support,
    venngram_subsets,
    venngram_support,
)
from interpretune.utils import MisconfigurationException

SITE = "hook_resid_post"
HOOKED = f"{SITE}.hook_sae_acts_post"
TRANSCODER = FeatureTargetSupport(
    value_sources=frozenset({"top_feature_scores", "top_feature_activation_values", "constant"})
)
CAPTURING = FeatureTargetSupport(
    value_sources=frozenset({"top_feature_scores", "top_feature_activation_values", "constant"}),
    captures_clean_activations=True,
)


def _feature_record(detail: FeatureTargetSupport) -> InterventionSupport:
    """An analysis backend's intervention record whose ``features`` target carries ``detail``."""
    return InterventionSupport(
        ops=frozenset({"replace"}),
        position_scopes=frozenset({PositionScope.POSITIONS}),
        targets={"features": detail},
    )


def _recipe(ops, *, site=SITE, tokens="last", sets=None):
    operands = {"v": {"shape": [2], "values": [1.0, 2.0], "basis": "b"}}
    sets = sets or {"top": {"features": "3,7", "layers": [2, 5]}}
    return load_recipe(
        treatment_book(
            {
                "name": "feature-treatment",
                "model": {"id": "m", "revision": "r1"},
                "book": {
                    "schema_version": 1,
                    "name": "features",
                    "portability": "checkpoint",
                    "applies_to": {"models": [{"id": "m", "revision": "r1"}]},
                    "sets": sets,
                },
                "operands": operands,
                "stages": [{"site": site, "tokens": tokens, "ops": ops}],
            }
        )
    ).recipe


def _from(column="score", transform=(), table="graph"):
    return {"from": {"table": table, "column": column}, "transform": list(transform)}


def _op(op="add", **value):
    return {"op": op, "target": {"set": "top"}, "value": value or {"dose": 0.5}}


def _table(**columns):
    return {"graph": columns}


SCORES = {(2, 3): 2.0, (2, 7): -4.0, (5, 3): 1.0, (5, 7): 0.0}


class TestModelBackendPath:
    def test_an_add_scatters_at_the_latent_sub_hook_and_applies(self):
        payload = recipe_interventions(_recipe([_op(dose=0.5)]), site_widths={HOOKED: 16})
        assert sorted(payload) == [f"blocks.2.{HOOKED}", f"blocks.5.{HOOKED}"]
        (spec,) = payload[f"blocks.2.{HOOKED}"]
        assert spec.op == "add" and spec.position_scope is PositionScope.LAST_TOKEN
        assert spec.intervention_tensor.shape == (16,) and torch.count_nonzero(spec.intervention_tensor) == 2
        out = apply_intervention(torch.zeros(1, 3, 16), spec, last_pos=2)
        assert out[0, 2, 3] == 0.5 and out[0, 2, 7] == 0.5 and out[0, 0, 3] == 0.0

    def test_a_scale_leaves_the_unnamed_features_at_one(self):
        (spec,) = recipe_interventions(_recipe([_op("scale", dose=0.0)]), site_widths={HOOKED: 16})[
            f"blocks.2.{HOOKED}"
        ]
        assert spec.op == "scale"
        assert spec.intervention_tensor[3] == 0.0 and spec.intervention_tensor.sum() == 14.0

    def test_a_latent_name_lists_its_features_in_units(self):
        """Venngram's `target_kind` reads this as units; it must not scatter into the site's own activation."""
        sets = {"top": {"features": "sae", "point": SITE, "layers": [5], "units": [1, 9]}}
        payload = recipe_interventions(_recipe([_op(dose=2.0)], sets=sets), site_widths={HOOKED: 16, SITE: 8})
        (spec,) = payload[f"blocks.5.{HOOKED}"]
        assert list(payload) == [f"blocks.5.{HOOKED}"] and spec.intervention_tensor[9] == 2.0

    def test_a_per_layer_doses_table_may_leave_features_undosed(self):
        sets = {"top": {"features": "sae", "layers": [2, 5], "units": [3, 7]}}
        payload = recipe_interventions(_recipe([_op(doses={2: {3: 1.5}})], sets=sets), site_widths={HOOKED: 16})
        assert list(payload) == [f"blocks.2.{HOOKED}"]
        assert torch.count_nonzero(payload[f"blocks.2.{HOOKED}"][0].intervention_tensor) == 1

    @pytest.mark.parametrize(
        "ops, kwargs, match",
        [
            pytest.param([_op("replace", dose=1.0)], {}, "has no native translation at a sub-hook", id="replace"),
            pytest.param([_op(dose=0.5)], {"site": f"blocks.2.{SITE}"}, "stage names the layer-free", id="layered"),
            pytest.param([_op(dose=0.5)], {"site": HOOKED}, "`hook_sae_acts_post` is implied", id="sub-hook-site"),
            pytest.param(
                [_op(dose=0.5)],
                {"sets": {"top": {"features": "sae", "layers": [2]}}},
                "names the latent space `sae` but no features",
                id="whole-space",
            ),
            pytest.param(
                [_op(dose=0.5)],
                {"sets": {"top": {"features": "1,2", "layers": "2:4"}}},
                "does not name its layers explicitly",
                id="layer-slice",
            ),
            pytest.param(
                [_op(dose=0.5)],
                {"sets": {"top": {"features": "sae", "layers": [2], "units": "0:3"}}},
                "names a slice of units",
                id="unit-slice",
            ),
            pytest.param(
                [{"op": "add", "value": {"dose": 1.0, "transform": ["abs"]}}],
                {"site": f"blocks.2.{SITE}"},
                "a whole-activation op's dose takes no transform",
                id="whole-activation-dose-transform",
            ),
        ],
    )
    def test_what_it_cannot_translate_is_refused_by_name(self, ops, kwargs, match):
        with pytest.raises(MisconfigurationException, match=match):
            recipe_interventions(_recipe(ops, **kwargs), site_widths={HOOKED: 16})

    def test_a_dose_runs_its_transforms_too(self):
        """Venngram's transforms apply to a dose or a doses table, not only to a table (venngram's conformance
        vectors include both)."""
        value = {"dose": -0.5, "transform": ["abs", {"scale": {"factor": 4.0}}]}
        (spec,) = recipe_interventions(_recipe([_op(**value)]), site_widths={HOOKED: 16})[f"blocks.2.{HOOKED}"]
        assert spec.intervention_tensor[3] == 2.0 and spec.intervention_tensor[7] == 2.0
        doses = {"doses": {2: {3: 3.0}, 5: {7: -1.5}}, "transform": [{"statistic": {"name": "t"}}]}
        sets = {"top": {"features": "sae", "layers": [2, 5], "units": [3, 7]}}
        payload = recipe_interventions(
            _recipe([_op(**doses)], sets=sets), site_widths={HOOKED: 16}, statistics={"t": {(2, 3): 2.0, (5, 7): 4.0}}
        )
        assert payload[f"blocks.2.{HOOKED}"][0].intervention_tensor[3] == 6.0
        assert payload[f"blocks.5.{HOOKED}"][0].intervention_tensor[7] == -6.0

    def test_the_latent_width_is_required_and_bounds_the_features(self):
        with pytest.raises(MisconfigurationException, match="no width was supplied for this site"):
            recipe_interventions(_recipe([_op(dose=0.5)]), site_widths={SITE: 16})
        with pytest.raises(MisconfigurationException, match="unit 7 at layer 2 is outside the site width 4"):
            recipe_interventions(_recipe([_op(dose=0.5)]), site_widths={HOOKED: 4})


class TestEvidenceValues:
    def _values(self, *transform, column="score", tables=None, statistics=None):
        payload = recipe_interventions(
            _recipe([_op(**_from(column, transform))]),
            site_widths={HOOKED: 8},
            tables=tables or _table(score=SCORES, activation={k: 3.0 for k in SCORES}),
            statistics=statistics,
        )
        return {(layer, f): payload[f"blocks.{layer}.{HOOKED}"][0].intervention_tensor[f].item() for layer, f in SCORES}

    def test_a_column_reads_as_written_with_no_transform(self):
        assert self._values() == SCORES

    @pytest.mark.parametrize(
        "transform, expected",
        [
            pytest.param(
                [{"scale": {"factor": 2.0}}], {(2, 3): 4.0, (2, 7): -8.0, (5, 3): 2.0, (5, 7): 0.0}, id="scale"
            ),
            pytest.param(["abs"], {(2, 3): 2.0, (2, 7): 4.0, (5, 3): 1.0, (5, 7): 0.0}, id="abs"),
            pytest.param(
                [{"normalize": {"by": "max_abs"}}], {(2, 3): 0.5, (2, 7): -1.0, (5, 3): 0.25, (5, 7): 0.0}, id="max-abs"
            ),
            pytest.param(
                ["abs", {"sign_aware_scale": {"factor": 2.0}}],
                {(2, 3): 4.0, (2, 7): -8.0, (5, 3): 2.0, (5, 7): 0.0},
                id="sign-aware-restores-the-score-sign",
            ),
            pytest.param(
                [{"scale": {"factor": -1.0}}, {"sign_of": {"column": "score"}}],
                {(2, 3): 2.0, (2, 7): -4.0, (5, 3): 1.0, (5, 7): 0.0},
                id="sign-of",
            ),
        ],
    )
    def test_each_transform_runs_in_order(self, transform, expected):
        assert self._values(*transform) == expected

    def test_max_abs_of_all_zeros_is_zero_rather_than_a_division(self):
        zeros = {k: 0.0 for k in SCORES}
        values = self._values({"normalize": {"by": "max_abs"}}, tables=_table(score=zeros))
        assert set(values.values()) == {0.0}

    @pytest.mark.parametrize(
        "statistics",
        [pytest.param({"median": 4.0}, id="scalar"), pytest.param({"median": {2: 4.0, 5: 8.0}}, id="per-layer")],
    )
    def test_statistic_multiplies_and_normalize_by_it_divides_at_each_layer(self, statistics):
        """`statistic` is the inverse of `normalize` by that statistic (venngram#58's draft): a dose written in
        units of a layer statistic becomes an activation-space dose by multiplying."""
        scaled = self._values({"statistic": {"name": "median"}}, statistics=statistics)
        normalized = self._values({"normalize": {"by": "median"}}, statistics=statistics)
        at_5 = statistics["median"][5] if isinstance(statistics["median"], dict) else statistics["median"]
        assert scaled[(2, 3)] == 8.0 and normalized[(2, 3)] == 0.5
        assert scaled[(5, 3)] == at_5 and normalized[(5, 3)] == 1.0 / at_5
        roundtrip = self._values(
            {"statistic": {"name": "median"}}, {"normalize": {"by": "median"}}, statistics=statistics
        )
        assert roundtrip == SCORES

    @pytest.mark.parametrize(
        "value, tables, statistics, match",
        [
            pytest.param(_from(), {}, None, "reads table `graph`, which neither the book", id="no-table"),
            pytest.param(_from("gone"), _table(score=SCORES), None, "no column `gone`", id="no-column"),
            pytest.param(
                _from(), _table(score={(2, 3): 1.0}), None, r"no value for \(layer, unit\) \[\(2, 7\)", id="no-member"
            ),
            pytest.param(
                _from(transform=[{"statistic": {"name": "median"}}]),
                _table(score=SCORES),
                None,
                "no statistic `median` was supplied",
                id="no-statistic",
            ),
            pytest.param(
                _from(transform=[{"statistic": {"name": "median"}}]),
                _table(score=SCORES),
                {"median": {2: 1.0}},
                "statistic `median` has no value at layer 5",
                id="no-statistic-layer",
            ),
            pytest.param(
                _from(transform=[{"normalize": {"by": "median"}}]),
                _table(score=SCORES),
                {"median": 0.0},
                r"statistic `median` is zero at \(layer, unit\) \(2, 3\)",
                id="zero-statistic",
            ),
            pytest.param(
                _from(transform=[{"sign_of": {"column": "gone"}}]),
                _table(score=SCORES),
                None,
                "no column `gone`",
                id="no-sign-column",
            ),
        ],
    )
    def test_what_the_caller_did_not_supply_is_refused_by_name(self, value, tables, statistics, match):
        with pytest.raises(MisconfigurationException, match=match):
            recipe_interventions(_recipe([_op(**value)]), site_widths={HOOKED: 8}, tables=tables, statistics=statistics)

    def test_a_statistic_may_be_given_per_member(self):
        per_member = {"median": {(2, 3): 2.0, (2, 7): 4.0, (5, 3): 0.5, (5, 7): 1.0}}
        values = self._values({"statistic": {"name": "median"}}, statistics=per_member)
        assert values == {(2, 3): 4.0, (2, 7): -16.0, (5, 3): 0.5, (5, 7): 0.0}

    @pytest.mark.parametrize(
        "statistic, match",
        [
            pytest.param({(2, 3): 1.0, 5: 1.0}, "mixes per-layer and per-\\(layer, unit\\) keys", id="mixed"),
            pytest.param({(2, 3): 1.0}, r"has no value at \(layer, unit\) \(2, 7\)", id="missing-member"),
        ],
    )
    def test_a_per_member_statistic_names_every_member(self, statistic, match):
        with pytest.raises(MisconfigurationException, match=match):
            self._values({"statistic": {"name": "median"}}, statistics={"median": statistic})

    def test_a_unit_target_reads_a_table_too(self):
        sets = {"top": {"point": SITE, "layers": [2], "units": [3]}}
        payload = recipe_interventions(
            _recipe([_op(**_from())], sets=sets), site_widths={SITE: 8}, tables=_table(score=SCORES)
        )
        assert payload[f"blocks.2.{SITE}"][0].intervention_tensor[3] == 2.0

    def test_a_whole_activation_op_still_takes_a_single_dose(self):
        op = {"op": "add", "value": _from()}
        with pytest.raises(MisconfigurationException, match="takes a single `dose`"):
            recipe_interventions(_recipe([op], site=f"blocks.2.{SITE}"), site_widths={}, tables=_table(score=SCORES))


class TestVenngramConformance:
    """Interpretune's evaluator agrees with venngram's own transform meanings, measured by venngram's vectors."""

    @staticmethod
    def _evaluate(case):
        """One conformance vector through the real path: a recipe whose feature set names one member per vector
        member, its value and transforms as the vector writes them, evaluated by ``member_values``."""
        from interpretune.analysis.venngram.values import member_values

        n = len(case.members)
        if case.source == "dose":
            value = {"dose": case.members[0].start}
        elif case.source == "doses":
            value = {"doses": {0: {i: m.start for i, m in enumerate(case.members)}}}
        else:
            value = {"from": {"table": "e", "column": case.column}}
        value["transform"] = [{t.name: dict(t.params)} if t.params else t.name for t in case.transform]
        sets = {"top": {"features": "sae", "layers": [0], "units": list(range(n))}}
        recipe = _recipe([{"op": "add", "target": {"set": "top"}, "value": value}], sets=sets)
        columns = {c for m in case.members for c in m.columns}
        tables = {
            "e": {c: {(0, i): m.columns[c] for i, m in enumerate(case.members) if c in m.columns} for c in columns}
        }
        names = {s for m in case.members for s in m.statistics}
        statistics = {s: {(0, i): m.statistics[s] for i, m in enumerate(case.members)} for s in names}
        values = member_values(recipe, recipe.stages[0].ops[0], tables=tables, statistics=statistics)
        return [values[(0, i)] for i in range(n)]

    def test_every_transform_vector_agrees(self):
        vg_ops = pytest.importorskip("venngram.ops")
        cases = vg_ops.transform_cases()
        assert {t.name for c in cases for t in c.transform} >= set(vg_ops.TRANSFORMS), "a core transform has no vector"
        assert vg_ops.check_transforms(self._evaluate) == []

    def test_a_disagreeing_evaluator_is_caught(self):
        """Positive control: the check can fail, so its empty answer above is evidence."""
        vg_ops = pytest.importorskip("venngram.ops")
        assert vg_ops.check_transforms(lambda case: [v + 1.0 for v in self._evaluate(case)])

    def test_the_table_transforms_agree_with_venngrams(self):
        """Venngram refuses a table-reading transform on a dose when it reads the recipe; the evaluator refuses the
        same set."""
        from interpretune.analysis.venngram.values import EVALUATED_TRANSFORMS, TABLE_TRANSFORMS

        vg_ops = pytest.importorskip("venngram.ops")
        evaluated = EVALUATED_TRANSFORMS | {"statistic"}
        assert TABLE_TRANSFORMS == {name for name in vg_ops.TABLE_TRANSFORMS if name in evaluated}
        from venngram.errors import AddressError

        with pytest.raises(AddressError, match="read the `from` table"):
            _recipe([_op(dose=1.0, transform=[{"sign_of": {"column": "score"}}])])


class TestAnalysisBackendPath:
    TOKENS = {"positions": [4, 6]}
    SITE = "transcoder.features"

    def _recipe(self, **value):
        return _recipe([_op("replace", **value)], site=self.SITE, tokens=self.TOKENS)

    def test_a_replace_becomes_one_tuple_per_member_and_position(self):
        tuples = recipe_feature_interventions(self._recipe(dose=2.5), support=TRANSCODER)
        assert sorted(tuples) == sorted(
            (layer, position, feature, 2.5) for layer in (2, 5) for feature in (3, 7) for position in (4, 6)
        )

    @pytest.mark.parametrize(
        "settings, value",
        [
            pytest.param(
                {"value": None, "value_source": "top_feature_scores", "sign_aware_scale": True},
                _from(transform=[{"sign_aware_scale": {"factor": 3.0}}]),
                id="score-scaled",
            ),
            pytest.param(
                {"value": None, "value_source": "top_feature_activation_values", "sign_aware_scale": True},
                _from("activation", [{"sign_of": {"column": "score"}}, {"scale": {"factor": 3.0}}]),
                id="activation-signed-by-score",
            ),
            pytest.param(
                {"value": 5.0, "value_source": "constant", "sign_aware_scale": True, "max_influence_norm_scale": True},
                _from(transform=[{"normalize": {"by": "max_abs"}}, {"scale": {"factor": 15.0}}]),
                id="constant-under-max-influence",
            ),
        ],
    )
    def test_the_tuples_match_circuit_tracers_own(self, settings, value):
        """The transforms mean what circuit-tracer's settings mean: the two routes reach one set of tuples."""
        backends = pytest.importorskip("interpretune.adapters.circuit_tracer.backends")
        activations = {(2, 3): 0.5, (2, 7): 1.5, (5, 3): 2.5, (5, 7): 3.5}
        rows = sorted(SCORES)
        theirs, _ = backends.CircuitTracerAnalysisBackend().build_feature_interventions(
            {
                "top_feature_ids": [[layer, 4, f] for layer, f in rows],
                "top_feature_scores": [SCORES[k] for k in rows],
                "top_feature_activation_values": [activations[k] for k in rows],
            },
            {
                "scale_factor": 3.0,
                "max_influence_norm_scale": False,
                "constrained_layers": None,
                "freeze_attention": None,
                "apply_activation_function": None,
                "sparse": False,
                "return_activations": False,
                "freeze_norms": False,
                **settings,
            },
        )
        ours = recipe_feature_interventions(
            _recipe([_op("replace", **value)], site=self.SITE, tokens={"positions": [4]}),
            support=TRANSCODER,
            tables=_table(score=SCORES, activation=activations),
        )
        ours, theirs = sorted(ours), sorted(theirs)
        assert [t[:3] for t in ours] == [t[:3] for t in theirs]
        assert [t[3] for t in ours] == pytest.approx([t[3] for t in theirs])

    @pytest.mark.parametrize(
        "ops, tokens, match",
        [
            pytest.param([_op("add", dose=1.0)], TOKENS, "declares no clean capture", id="add-without-a-capture"),
            pytest.param([_op("scale", dose=1.0)], TOKENS, "declares no clean capture", id="scale-without-a-capture"),
            pytest.param([_op("replace", dose=1.0)], "last", "name their positions", id="scope"),
            pytest.param(
                [{"op": "replace", "target": {"set": "top"}, "operand": "v"}], TOKENS, "takes a value", id="operand"
            ),
            pytest.param(
                [{"op": "replace", "value": {"dose": 1.0}}], TOKENS, "act on a feature target only", id="whole"
            ),
        ],
    )
    def test_what_tuples_cannot_express_is_refused_by_name(self, ops, tokens, match):
        recipe = _recipe(ops, site=self.SITE, tokens=tokens)
        with pytest.raises(MisconfigurationException, match=match):
            recipe_feature_interventions(recipe, support=TRANSCODER)

    def test_a_backend_not_honouring_constant_cannot_take_computed_values(self):
        scores_only = FeatureTargetSupport(value_sources=frozenset({"top_feature_scores"}))
        with pytest.raises(MisconfigurationException, match="cannot take a recipe's computed values"):
            recipe_feature_interventions(self._recipe(dose=1.0), support=scores_only)
        assert venngram_feature_support(_feature_record(scores_only)).values == frozenset()

    @pytest.mark.parametrize("op, combine", [("add", lambda a, d: a + d), ("scale", lambda a, d: a * d)])
    def test_add_and_scale_are_relative_to_the_clean_capture(self, op, combine):
        clean = torch.arange(6 * 8 * 8, dtype=torch.float32).reshape(6, 8, 8) / 10
        recipe = _recipe([_op(op, dose=1.5)], site=self.SITE, tokens=self.TOKENS)
        tuples = recipe_feature_interventions(recipe, support=CAPTURING, clean_activations=clean)
        assert sorted(tuples) == sorted(
            (layer, position, feature, pytest.approx(combine(float(clean[layer, position, feature]), 1.5)))
            for layer in (2, 5)
            for feature in (3, 7)
            for position in (4, 6)
        )

    def test_a_relative_op_needs_the_capture_passed(self):
        recipe = _recipe([_op("add", dose=1.0)], site=self.SITE, tokens=self.TOKENS)
        with pytest.raises(MisconfigurationException, match="pass `clean_activations`"):
            recipe_feature_interventions(recipe, support=CAPTURING)
        with pytest.raises(
            MisconfigurationException, match=r"of shape \(6, 5, 8\).*no activation at layer 2, position 6"
        ):
            recipe_feature_interventions(recipe, support=CAPTURING, clean_activations=torch.zeros(6, 5, 8))

    def test_a_replace_needs_no_capture(self):
        """The capture is read only for a relative op, so a replace-only recipe translates as it did."""
        assert recipe_feature_interventions(self._recipe(dose=2.0), support=CAPTURING, clean_activations=None)

    def test_add_and_scale_are_declared_only_with_a_capture(self):
        recipe = _recipe([_op("add", dose=1.0)], site=self.SITE, tokens=self.TOKENS)
        plain, capturing = (venngram_feature_support(_feature_record(d)) for d in (TRANSCODER, CAPTURING))
        assert plain.ops == frozenset({"replace"})
        assert capturing.ops == frozenset({"replace", "add", "scale"})
        refusals = check_recipe(recipe, plain, analysis_backend=True)
        assert any("`add`): the op is not supported" in r for r in refusals), refusals
        assert check_recipe(recipe, capturing, analysis_backend=True) == []


class TestSupport:
    INTERVENTION = InterventionSupport.every()

    def test_feature_targets_are_declared_only_with_a_latent_model(self):
        recipe = _recipe([_op(dose=0.5)])
        assert any("target `features`" in r for r in check_recipe(recipe, venngram_support(self.INTERVENTION)))
        require_supported(recipe, venngram_support(self.INTERVENTION, latent_models=True))

    def test_a_statistic_is_declared_only_when_the_caller_names_one(self):
        recipe = _recipe([_op(**_from(transform=[{"statistic": {"name": "median"}}]))])
        declared = venngram_support(self.INTERVENTION, latent_models=True, statistics=frozenset({"median"}))
        assert any(
            "transform `statistic`" in r
            for r in check_recipe(recipe, venngram_support(self.INTERVENTION, latent_models=True))
        )
        assert check_recipe(recipe, declared) == []

    def test_the_analysis_path_checks_with_its_own_refusals(self):
        recipe = _recipe([_op("replace", dose=1.0)], site="transcoder.features", tokens={"positions": [1]})
        support = venngram_feature_support(_feature_record(TRANSCODER))
        assert check_recipe(recipe, support, analysis_backend=True) == []
        require_supported(recipe, support, analysis_backend=True)
        assert any("no native translation at a sub-hook" in r for r in check_recipe(recipe, support))

    @pytest.mark.parametrize(
        "support",
        [
            pytest.param(venngram_support(InterventionSupport.every(), latent_models=True), id="model-backend"),
            pytest.param(venngram_feature_support(_feature_record(TRANSCODER)), id="analysis-backend"),
        ],
    )
    def test_both_declarations_publish_as_support_records(self, support):
        steering = support_mapping(support)["sections"]["treatments"]
        assert "features" in steering["targets"] and "from" in steering["values"]
        assert "sign_aware_scale" in steering["transforms"]


GRAPH_SITE = "transcoder.features"
GRAPH_MODEL = {"id": "m", "revision": "r1"}
GRAPH_TOKENIZER = {"id": "m-tokenizer", "revision": "t1"}


def _node(node_id, feature, anchor, activation, baseline=0.0, space="tc"):
    index = {"feature": feature} if space == "tc" else {"block": feature}
    return {
        "id": node_id,
        "kind": "feature",
        "site": GRAPH_SITE,
        "space": space,
        "layer": 1,
        "position": 0,
        "index": index,
        "activation": activation,
        "baseline": baseline,
        "anchor": anchor,
    }


#: Two transcoder nodes: one on the prompt's last token, one on the token id 22 wherever it falls.
NODES = [
    _node("f0", 7, {"kind": "from_end", "offset": 0}, 3.0, 1.0),
    _node("f1", 8, {"kind": "token_match", "token_ids": [22]}, 2.0, 0.5),
]


def _graph_recipe(tmp_path, ops, nodes=NODES, value_shape=None, tc_kind="transcoder", universe=None):
    """A book carrying one graph subset and a treatment applying ``ops`` to it, read as that treatment's recipe."""
    import hashlib

    import yaml
    from venngram.book import AddressBook

    spaces = {"tc": {"id": "dict/transcoders", "revision": "p", "kind": tc_kind}}
    if value_shape is not None:
        spaces["bsf"] = {"id": "dict/bsf", "revision": "p", "kind": "block_sparse", "index": ["block"]}
        spaces["bsf"]["value_shape"] = value_shape
    graph = {
        "schema_version": 1,
        "kind": "graph_subset",
        "name": "reasoning",
        "model": GRAPH_MODEL,
        "spaces": spaces,
        "source": {"tokenizer": GRAPH_TOKENIZER, "token_ids": [11, 22, 33]},
        "nodes": nodes,
    }
    path = tmp_path / "reasoning.yaml"
    path.write_text(yaml.safe_dump(graph), encoding="utf-8")
    book = {
        "schema_version": 5,
        "name": "transfer",
        "sets": {"universe": universe or {"features": "tc", "layers": [1], "units": [7, 8, 9]}},
        "graphs": {"reasoning": {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
        "treatments": {"transfer": {"model": GRAPH_MODEL, "stages": [{"site": GRAPH_SITE, "ops": ops}]}},
    }
    return AddressBook.from_mapping(book, base_dir=tmp_path).treatments["transfer"]


def _graph_op(op="replace", column="activation", baseline=None, transform=()):
    source = {"graph": "reasoning", "column": column, **({"baseline": baseline} if baseline else {})}
    return {"op": op, "target": {"graph": "reasoning"}, "value": {"from": source, "transform": list(transform)}}


RESTRICTION = {
    "op": "replace",
    "target": {"graph": "reasoning", "outside": {"set": "universe", "tokens": "prompt"}},
    "value": {"dose": 0.0},
}

#: The destination prompt: token 22 falls at position 1, and the last token is at position 3.
PROMPT = [5, 22, 9, 4]


class TestGraphSubsets:
    """#966: a graph subset distilled from one prompt, placed by its anchors in another as feature tuples."""

    def test_a_replace_places_each_node_by_its_anchor_with_its_column(self, tmp_path):
        recipe = _graph_recipe(tmp_path, [_graph_op()])
        tuples = recipe_feature_interventions(recipe, support=TRANSCODER, token_ids=PROMPT, tokenizer=GRAPH_TOKENIZER)
        assert tuples == [(1, 3, 7, 3.0), (1, 1, 8, 2.0)]

    def test_an_add_of_a_baseline_delta_is_relative_to_the_clean_activation(self, tmp_path):
        recipe = _graph_recipe(tmp_path, [_graph_op("add", baseline="baseline", transform=[{"scale": {"factor": 2}}])])
        clean = torch.arange(2 * 4 * 10, dtype=torch.float32).reshape(2, 4, 10)
        tuples = recipe_feature_interventions(
            recipe, support=CAPTURING, clean_activations=clean, token_ids=PROMPT, tokenizer=GRAPH_TOKENIZER
        )
        assert tuples == [(1, 3, 7, float(clean[1, 3, 7]) + 4.0), (1, 1, 8, float(clean[1, 1, 8]) + 3.0)]

    def test_anchors_that_need_no_tokenizer_resolve_without_one(self, tmp_path):
        nodes = [_node("f0", 7, {"kind": "absolute", "position": 2}, 1.5), _node("f1", 8, {"kind": "all"}, -1.0)]
        tuples = recipe_feature_interventions(
            _graph_recipe(tmp_path, [_graph_op()], nodes=nodes), support=TRANSCODER, token_ids=[1, 2, 3]
        )
        assert tuples == [(1, 2, 7, 1.5)] + [(1, p, 8, -1.0) for p in range(3)]

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            pytest.param({}, "pass `token_ids`", id="no-prompt"),
            pytest.param({"token_ids": [5, 9, 4]}, "node `f1`", id="unresolved-anchor"),
            pytest.param({"token_ids": PROMPT}, "pinned tokenizer", id="no-tokenizer-for-a-token-match"),
        ],
    )
    def test_a_graph_that_cannot_be_placed_is_refused_whole(self, tmp_path, kwargs, match):
        recipe = _graph_recipe(tmp_path, [_graph_op()])
        kwargs = {"tokenizer": GRAPH_TOKENIZER, **kwargs} if match != "pinned tokenizer" else kwargs
        with pytest.raises(MisconfigurationException, match=match):
            recipe_feature_interventions(recipe, support=TRANSCODER, **kwargs)

    @pytest.mark.parametrize(
        "ops, support, match, value_shape",
        [
            pytest.param([RESTRICTION], TRANSCODER, "would enumerate each one", None, id="restriction"),
            pytest.param([_graph_op("add")], TRANSCODER, "clean capture", None, id="add-without-a-capture"),
            pytest.param(
                [_graph_op(transform=[{"statistic": {"name": "median"}}])],
                TRANSCODER,
                r"\['statistic'\] are not applied",
                None,
                id="statistic",
            ),
            pytest.param([_graph_op()], TRANSCODER, "`block_sparse` featurizer", [2], id="block-sparse"),
        ],
    )
    def test_what_feature_tuples_cannot_express_is_refused_by_name(self, tmp_path, ops, support, match, value_shape):
        nodes = NODES
        if value_shape is not None:
            nodes = NODES + [{**_node("b0", 4, {"kind": "absolute", "position": 0}, [0.5, -1.0], space="bsf")}]
            nodes[-1]["baseline"] = [0.0, 0.0]
        recipe = _graph_recipe(tmp_path, ops, nodes=nodes, value_shape=value_shape)
        with pytest.raises(MisconfigurationException, match=match):
            recipe_feature_interventions(recipe, support=support, token_ids=PROMPT, tokenizer=GRAPH_TOKENIZER)

    @pytest.mark.parametrize(
        "detail, ops", [(TRANSCODER, {"replace"}), (CAPTURING, {"replace", "add"})], ids=["plain", "capturing"]
    )
    def test_the_subset_declaration_follows_the_feature_record(self, detail, ops):
        from venngram.graphs import ANCHORS

        subsets = venngram_feature_subsets(_feature_record(detail))
        assert subsets.ops == ops and subsets.anchors == frozenset(ANCHORS) and not subsets.complement
        assert subsets.featurizers == {"transcoder", "clt"} and subsets.values == {"scalar"}

    def test_a_backend_without_constants_declares_no_subset_ops(self):
        scores_only = FeatureTargetSupport(value_sources=frozenset({"top_feature_scores"}))
        assert venngram_feature_subsets(_feature_record(scores_only)).ops == frozenset()

    @pytest.mark.parametrize(
        "ops, refused",
        [([_graph_op()], False), ([RESTRICTION], True), ([_graph_op("add", baseline="baseline")], True)],
        ids=["replace", "restriction", "add"],
    )
    def test_the_declaration_and_the_apply_path_agree(self, tmp_path, ops, refused):
        """Venngram's check of the declaration refuses what the apply path refuses, so a recipe passing one passes
        the other."""
        recipe = _graph_recipe(tmp_path, ops)
        declared = venngram_feature_subsets(_feature_record(TRANSCODER)).check_treatment(recipe)
        assert bool(declared) is refused, declared
        checked = check_recipe(recipe, venngram_feature_support(_feature_record(TRANSCODER)), analysis_backend=True)
        assert checked == declared
        if refused:
            with pytest.raises(MisconfigurationException):
                recipe_feature_interventions(recipe, support=TRANSCODER, token_ids=PROMPT, tokenizer=GRAPH_TOKENIZER)
        else:
            recipe_feature_interventions(recipe, support=TRANSCODER, token_ids=PROMPT, tokenizer=GRAPH_TOKENIZER)


GRAPH_HOOKED = f"{GRAPH_SITE}.hook_sae_acts_post"


def _block(node_id, block, anchor, activation):
    node = _node(node_id, block, anchor, activation, space="bsf")
    node["baseline"] = [0.0] * len(activation)
    return node


def _apply(payload, value):
    for spec in payload[f"blocks.1.{GRAPH_HOOKED}"]:
        value = apply_intervention(value, spec, last_pos=value.shape[1] - 1)
    return value


class TestGraphSubsetsOnTheModelBackend:
    """#966: the same graph subset written at an attached latent model's sub-hook, one spec per layer and
    position."""

    def _payload(self, recipe, width=16, **kwargs):
        kwargs = {"token_ids": PROMPT, "tokenizer": GRAPH_TOKENIZER, **kwargs}
        return recipe_interventions(recipe, site_widths={GRAPH_HOOKED: width}, **kwargs)

    def test_a_replace_writes_each_node_at_its_anchored_position_alone(self, tmp_path):
        payload = self._payload(_graph_recipe(tmp_path, [_graph_op()]))
        assert list(payload) == [f"blocks.1.{GRAPH_HOOKED}"]
        specs = payload[f"blocks.1.{GRAPH_HOOKED}"]
        assert [(s.op, s.positions, s.unit_indices) for s in specs] == [
            ("replace", (1,), (8,)),
            ("replace", (3,), (7,)),
        ]
        assert all(s.position_scope is PositionScope.POSITIONS for s in specs)
        out = _apply(payload, torch.ones(1, 4, 16))
        assert out[0, 3, 7] == 3.0 and out[0, 1, 8] == 2.0
        assert out.sum() == 4 * 16 - 2 + 5.0  # every other feature and position untouched

    def test_an_add_adds_the_transformed_delta_with_no_clean_capture(self, tmp_path):
        op = _graph_op("add", baseline="baseline", transform=[{"scale": {"factor": 2}}])
        out = _apply(self._payload(_graph_recipe(tmp_path, [op])), torch.ones(1, 4, 16))
        assert out[0, 3, 7] == 1.0 + 4.0 and out[0, 1, 8] == 1.0 + 3.0

    def test_nodes_sharing_a_position_share_its_spec(self, tmp_path):
        nodes = [_node("f0", 7, {"kind": "absolute", "position": 2}, 1.5), _node("f1", 8, {"kind": "all"}, -1.0)]
        specs = self._payload(_graph_recipe(tmp_path, [_graph_op()], nodes=nodes), token_ids=[1, 2, 3])[
            f"blocks.1.{GRAPH_HOOKED}"
        ]
        assert [(s.positions, s.unit_indices) for s in specs] == [((0,), (8,)), ((1,), (8,)), ((2,), (7, 8))]
        assert specs[2].intervention_tensor.tolist() == [[1.5, -1.0]]

    def test_a_vector_node_is_its_block_of_columns(self, tmp_path):
        nodes = [
            _block("b0", 2, {"kind": "absolute", "position": 0}, [0.5, -1.0]),
            _block("b1", 0, {"kind": "from_end", "offset": 0}, [1.0, 2.0]),
        ]
        payload = self._payload(_graph_recipe(tmp_path, [_graph_op()], nodes=nodes, value_shape=[2]), width=8)
        specs = payload[f"blocks.1.{GRAPH_HOOKED}"]
        assert [(s.positions, s.unit_indices) for s in specs] == [((0,), (4, 5)), ((3,), (0, 1))]
        out = _apply(payload, torch.zeros(1, 4, 8))
        assert out[0, 0].tolist() == [0, 0, 0, 0, 0.5, -1.0, 0, 0] and out[0, 3, :2].tolist() == [1.0, 2.0]

    @pytest.mark.parametrize(
        "ops, nodes, kwargs, match",
        [
            pytest.param(
                [_graph_op(transform=[{"statistic": {"name": "median"}}])],
                NODES,
                {},
                r"\['statistic'\] are not applied",
                id="statistic",
            ),
            pytest.param(
                [_graph_op()],
                NODES + [_block("b0", 1, {"kind": "absolute", "position": 0}, [0.5, -1.0])],
                {},
                "one latent model is attached at a hook",
                id="two-spaces-at-one-hook",
            ),
            pytest.param([_graph_op()], NODES, {"token_ids": None}, "pass `token_ids`", id="no-prompt"),
            pytest.param([_graph_op()], NODES, {"width": 8}, "feature 8|unit 8 .* outside the site width", id="width"),
            pytest.param([_graph_op()], NODES, {"flow": "grad"}, "acts on the activation value", id="grad-flow"),
        ],
    )
    def test_what_one_latent_model_cannot_take_is_refused_by_name(self, tmp_path, ops, nodes, kwargs, match):
        recipe = _graph_recipe(tmp_path, ops, nodes=nodes, value_shape=[2])
        with pytest.raises(MisconfigurationException, match=match):
            self._payload(recipe, **kwargs)

    def test_the_latent_width_is_required(self, tmp_path):
        with pytest.raises(MisconfigurationException, match="no width was supplied"):
            recipe_interventions(_graph_recipe(tmp_path, [_graph_op()]), site_widths={}, token_ids=PROMPT)

    def test_a_cross_layer_transcoder_is_not_placed(self, tmp_path):
        recipe = _graph_recipe(tmp_path, [_graph_op()], tc_kind="clt")
        with pytest.raises(MisconfigurationException, match="`clt` featurizer, which a latent sub-hook"):
            self._payload(recipe)

    def test_a_restriction_writes_its_dose_on_the_complement_at_every_position(self, tmp_path):
        payload = self._payload(_graph_recipe(tmp_path, [RESTRICTION]))
        specs = payload[f"blocks.1.{GRAPH_HOOKED}"]
        # the universe is features 7, 8 and 9; the graph keeps 8 at position 1 and 7 at position 3
        expected = [((0,), (7, 8, 9)), ((1,), (7, 9)), ((2,), (7, 8, 9)), ((3,), (8, 9))]
        assert [(s.positions, s.unit_indices) for s in specs] == expected
        assert all(s.op == "replace" and not s.intervention_tensor.any() for s in specs)
        out = _apply(payload, torch.ones(1, 4, 16))
        assert out[0, 1, 8] == 1.0 and out[0, 3, 7] == 1.0 and out[0, :, 10:].all()
        assert out[0, :, 7:10].sum() == 2.0  # only the kept coordinates survive within the universe

    def test_a_restriction_and_a_replace_compose_into_the_subset_alone(self, tmp_path):
        out = _apply(self._payload(_graph_recipe(tmp_path, [RESTRICTION, _graph_op()])), torch.ones(1, 4, 16))
        assert out[0, 3, 7] == 3.0 and out[0, 1, 8] == 2.0 and out[0, :, 7:10].sum() == 5.0

    def test_a_whole_space_universe_takes_the_latent_width(self, tmp_path):
        recipe = _graph_recipe(tmp_path, [RESTRICTION], universe={"features": "tc", "layers": [1]})
        specs = self._payload(recipe, width=10)[f"blocks.1.{GRAPH_HOOKED}"]
        assert [len(s.unit_indices) for s in specs] == [10, 9, 10, 9]
        assert 8 not in specs[1].unit_indices and 7 not in specs[3].unit_indices

    def test_a_block_restriction_covers_each_blocks_columns(self, tmp_path):
        nodes = [_block("b0", 1, {"kind": "absolute", "position": 0}, [0.5, -1.0])]
        universe = {"features": "bsf", "layers": [1], "units": [0, 1]}
        recipe = _graph_recipe(tmp_path, [RESTRICTION], nodes=nodes, value_shape=[2], universe=universe)
        specs = self._payload(recipe, width=4, token_ids=[5, 6])[f"blocks.1.{GRAPH_HOOKED}"]
        assert [(s.positions, s.unit_indices) for s in specs] == [((0,), (0, 1)), ((1,), (0, 1, 2, 3))]

    @pytest.mark.parametrize(
        "universe, match",
        [
            pytest.param({"features": "bsf", "layers": [1], "units": [7, 8]}, "is space `bsf`", id="other-space"),
            pytest.param({"features": "tc", "units": [7, 8, 9]}, "does not name its layers", id="no-layers"),
        ],
    )
    def test_a_universe_the_graph_cannot_be_restricted_in_is_refused_by_name(self, tmp_path, universe, match):
        recipe = _graph_recipe(tmp_path, [RESTRICTION], universe=universe, value_shape=[2])
        assert any(
            match in r for r in check_recipe(recipe, venngram_support(InterventionSupport.every(), latent_models=True))
        )
        with pytest.raises(MisconfigurationException, match=match):
            self._payload(recipe)

    def test_the_declaration_needs_an_attached_latent_model(self):
        from venngram.graphs import ANCHORS

        subsets = venngram_subsets(InterventionSupport.every(), latent_models=True)
        assert subsets.ops == {"replace", "add"} and subsets.anchors == frozenset(ANCHORS) and subsets.complement
        assert subsets.featurizers == {"sae", "transcoder", "block_sparse"} and subsets.values == {"scalar", "vector"}
        assert venngram_subsets(InterventionSupport.every()).ops == frozenset()
        assert not venngram_subsets(InterventionSupport.every()).complement

    @pytest.mark.parametrize(
        "ops, nodes, latent_models, refused",
        [
            ([_graph_op()], NODES, True, False),
            ([_graph_op("add", baseline="baseline")], NODES, True, False),
            ([RESTRICTION], NODES, True, False),
            ([_graph_op()], NODES, False, True),
            ([_graph_op()], NODES + [_block("b0", 1, {"kind": "absolute", "position": 0}, [0.5, -1.0])], True, True),
        ],
        ids=["replace", "add", "restriction", "no-latent-model", "two-spaces"],
    )
    def test_the_check_and_the_apply_path_agree(self, tmp_path, ops, nodes, latent_models, refused):
        """A recipe the model-backend check passes is one the apply path writes, and one it refuses is refused."""
        recipe = _graph_recipe(tmp_path, ops, nodes=nodes, value_shape=[2])
        checked = check_recipe(recipe, venngram_support(InterventionSupport.every(), latent_models=latent_models))
        assert bool(checked) is refused, checked
        if refused and latent_models:
            with pytest.raises(MisconfigurationException):
                self._payload(recipe)
        elif not refused:
            self._payload(recipe)
