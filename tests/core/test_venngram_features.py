"""Feature targets and evidence-sourced values (#887, slice 3b): the two feature paths and the transform
pipeline."""

import pytest
import torch

pytest.importorskip("venngram", reason="the venngram extra is not installed")

from interpretune.analysis.backends.capabilities import (
    FeatureInterventionSupport,
    InterventionMode,
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
    venngram_feature_support,
    venngram_support,
)
from interpretune.utils import MisconfigurationException

SITE = "hook_resid_post"
HOOKED = f"{SITE}.hook_sae_acts_post"
TRANSCODER = FeatureInterventionSupport(
    value_sources=frozenset({"top_feature_scores", "top_feature_activation_values", "constant"})
)
CAPTURING = FeatureInterventionSupport(
    value_sources=frozenset({"top_feature_scores", "top_feature_activation_values", "constant"}),
    captures_clean_activations=True,
)


def _recipe(ops, *, site=SITE, tokens="last", sets=None):
    operands = {"v": {"shape": [2], "values": [1.0, 2.0], "basis": "b"}}
    sets = sets or {"top": {"features": "3,7", "layers": [2, 5]}}
    return load_recipe(
        {
            "schema_version": 2,
            "kind": "recipe",
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
    ).recipe


def _from(column="score", transform=(), evidence="graph"):
    return {"from": {"evidence": evidence, "column": column}, "transform": list(transform)}


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
        assert spec.mode is InterventionMode.ADD and spec.position_scope is PositionScope.LAST_TOKEN
        assert spec.intervention_tensor.shape == (16,) and torch.count_nonzero(spec.intervention_tensor) == 2
        out = apply_intervention(torch.zeros(1, 3, 16), spec, last_pos=2)
        assert out[0, 2, 3] == 0.5 and out[0, 2, 7] == 0.5 and out[0, 0, 3] == 0.0

    def test_a_scale_leaves_the_unnamed_features_at_one(self):
        (spec,) = recipe_interventions(_recipe([_op("scale", dose=0.0)]), site_widths={HOOKED: 16})[
            f"blocks.2.{HOOKED}"
        ]
        assert spec.mode is InterventionMode.SCALE
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
            pytest.param([_op("replace", dose=1.0)], {}, "has no native mode at a sub-hook", id="replace"),
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
        """Venngram's transforms apply to a dose or a doses table, not only to evidence (venngram's conformance
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
            evidence=tables or _table(score=SCORES, activation={k: 3.0 for k in SCORES}),
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
            pytest.param(_from(), {}, None, "reads evidence `graph`, which the caller did not supply", id="no-table"),
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
            recipe_interventions(
                _recipe([_op(**value)]), site_widths={HOOKED: 8}, evidence=tables, statistics=statistics
            )

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

    def test_a_unit_target_reads_evidence_too(self):
        sets = {"top": {"point": SITE, "layers": [2], "units": [3]}}
        payload = recipe_interventions(
            _recipe([_op(**_from())], sets=sets), site_widths={SITE: 8}, evidence=_table(score=SCORES)
        )
        assert payload[f"blocks.2.{SITE}"][0].intervention_tensor[3] == 2.0

    def test_a_whole_activation_op_still_takes_a_single_dose(self):
        op = {"op": "add", "value": _from()}
        with pytest.raises(MisconfigurationException, match="takes a single `dose`"):
            recipe_interventions(_recipe([op], site=f"blocks.2.{SITE}"), site_widths={}, evidence=_table(score=SCORES))


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
            value = {"from": {"evidence": "e", "column": case.column}}
        value["transform"] = [{t.name: dict(t.params)} if t.params else t.name for t in case.transform]
        sets = {"top": {"features": "sae", "layers": [0], "units": list(range(n))}}
        recipe = _recipe([{"op": "add", "target": {"set": "top"}, "value": value}], sets=sets)
        columns = {c for m in case.members for c in m.columns}
        evidence = {
            "e": {c: {(0, i): m.columns[c] for i, m in enumerate(case.members) if c in m.columns} for c in columns}
        }
        names = {s for m in case.members for s in m.statistics}
        statistics = {s: {(0, i): m.statistics[s] for i, m in enumerate(case.members)} for s in names}
        values = member_values(recipe, recipe.stages[0].ops[0], evidence=evidence, statistics=statistics)
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

    def test_the_evidence_transforms_agree_with_venngrams(self):
        """Venngram refuses an evidence-reading transform on a dose when it reads the recipe; the evaluator refuses
        the same set."""
        from interpretune.analysis.venngram.values import EVALUATED_TRANSFORMS, EVIDENCE_TRANSFORMS

        vg_ops = pytest.importorskip("venngram.ops")
        evaluated = EVALUATED_TRANSFORMS | {"statistic"}
        assert EVIDENCE_TRANSFORMS == {name for name in vg_ops.EVIDENCE_TRANSFORMS if name in evaluated}
        from venngram.errors import AddressError

        with pytest.raises(AddressError, match="read the evidence table"):
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
            evidence=_table(score=SCORES, activation=activations),
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
        scores_only = FeatureInterventionSupport(value_sources=frozenset({"top_feature_scores"}))
        with pytest.raises(MisconfigurationException, match="cannot take a recipe's computed values"):
            recipe_feature_interventions(self._recipe(dose=1.0), support=scores_only)
        assert venngram_feature_support(scores_only).values == frozenset()

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
        assert venngram_feature_support(TRANSCODER).ops == frozenset({"replace"})
        assert venngram_feature_support(CAPTURING).ops == frozenset({"replace", "add", "scale"})
        refusals = check_recipe(recipe, venngram_feature_support(TRANSCODER), analysis_backend=True)
        assert any("`add`): the op is not supported" in r for r in refusals), refusals
        assert check_recipe(recipe, venngram_feature_support(CAPTURING), analysis_backend=True) == []


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
        support = venngram_feature_support(TRANSCODER)
        assert check_recipe(recipe, support, analysis_backend=True) == []
        require_supported(recipe, support, analysis_backend=True)
        assert any("no native mode at a sub-hook" in r for r in check_recipe(recipe, support))

    @pytest.mark.parametrize(
        "support",
        [
            pytest.param(venngram_support(InterventionSupport.every(), latent_models=True), id="model-backend"),
            pytest.param(venngram_feature_support(TRANSCODER), id="analysis-backend"),
        ],
    )
    def test_both_declarations_publish_as_support_records(self, support):
        steering = support_mapping(support)["sections"]["steering"]
        assert "features" in steering["targets"] and "from" in steering["values"]
        assert "sign_aware_scale" in steering["transforms"]
