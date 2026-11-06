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
                [{"op": "add", "target": {"set": "top"}, "value": {"dose": 1.0, "transform": ["abs"]}}],
                {},
                "a transform applies to an evidence-sourced value",
                id="dose-transform",
            ),
        ],
    )
    def test_what_it_cannot_translate_is_refused_by_name(self, ops, kwargs, match):
        with pytest.raises(MisconfigurationException, match=match):
            recipe_interventions(_recipe(ops, **kwargs), site_widths={HOOKED: 16})

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
        "statistics, expected",
        [
            pytest.param({"median": 2.0}, 1.0, id="scalar"),
            pytest.param({"median": {2: 2.0, 5: 4.0}}, 1.0, id="per-layer"),
        ],
    )
    def test_a_supplied_statistic_divides_at_each_layer(self, statistics, expected):
        for transform in ({"statistic": {"name": "median"}}, {"normalize": {"by": "median"}}):
            assert self._values(transform, statistics=statistics)[(2, 3)] == expected

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
                "statistic `median` is zero at layer 2",
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
        assert sorted(ours) == pytest.approx(sorted(theirs))

    @pytest.mark.parametrize(
        "ops, tokens, match",
        [
            pytest.param([_op("add", dose=1.0)], TOKENS, "replace a feature's value, not `add`", id="add"),
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
