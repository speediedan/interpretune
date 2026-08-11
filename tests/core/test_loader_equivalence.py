"""4a loader-equivalence harness: baseline leg (hub design v3 §11.4).

Captures every CLI experiment config through the CURRENT jsonargparse session surface and asserts the
capture works. When ``load_session_cfg`` lands, each case gains the second leg: the unified loader's
output must be ``session_spec``-identical to this baseline BEFORE the old path is removed. The
equivalence set also includes a registry/examples configuration exercising ``AutoCompConfig``
``make_dataclass`` synthesis — none of the CLI configs uses AutoComp (they bind explicit
``RTEBoolq*Config`` classes), so that acceptance case cannot come from this parametrization alone.
"""

from __future__ import annotations

import pytest

import json
from pathlib import Path

from tests.core.loader_equivalence import (
    cli_experiment_configs,
    session_spec,
)

CONFIGS = cli_experiment_configs()
BASELINE_SPECS = json.loads((Path(__file__).parent / "fixtures" / "cli_session_baseline_specs.json").read_text())


def _baseline_for(config_path):
    rel = str(config_path.relative_to(config_path.parents[3]))
    return BASELINE_SPECS[rel]


def test_all_cli_experiment_configs_discovered():
    """The harness must cover the full experiment-config surface; a moved/removed config shrinks it loudly."""
    assert len(CONFIGS) == 15, sorted(str(c) for c in CONFIGS)


from tests.core.loader_equivalence import (  # noqa: E402
    capture_registered_cfg_via_factories,
    example_configuration_files,
    registered_spec,
)

EXAMPLE_CONFIGS = example_configuration_files()


def test_all_example_configurations_discovered():
    assert len(EXAMPLE_CONFIGS) == 6, sorted(str(c) for c in EXAMPLE_CONFIGS)


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS, ids=lambda p: p.stem)
def test_factory_path_capture_baseline(config_path):
    """Every examples/ configuration instantiates via the factory path; AutoComp synthesis included."""
    key, registered = capture_registered_cfg_via_factories(config_path)
    spec = registered_spec(registered)
    assert spec["datamodule_cfg"]["__class__"] and spec["module_cfg"]["__class__"]


def test_autocomp_synthesis_baseline():
    """The AutoComp acceptance case (umbrella 4a note 4): a synthesized module_cfg class, pinned.

    ``rte_demo.gemma2.circuit_tracer`` declares ``auto_comp_cfg`` (module_cfg_name RTEBoolqConfig +
    entailment-mapping mixin), so its concrete config class exists only as ``make_dataclass`` output.
    The unified loader must route through the factories so this synthesis keeps happening; this pin
    fails if a reimplementation constructs a plain ITConfig instead.
    """
    target = next(c for c in EXAMPLE_CONFIGS if c.stem == "rte_demo.gemma2.circuit_tracer")
    _, registered = capture_registered_cfg_via_factories(target)
    cfg_cls = type(registered.module_cfg)
    assert cfg_cls.__module__ != "interpretune.config.module" or cfg_cls.__qualname__ != "ITConfig", (
        "expected an AutoComp-synthesized config class, got plain ITConfig"
    )
    # the synthesized class must carry the entailment-mapping surface the mixin composes in
    assert hasattr(registered.module_cfg, "entailment_mapping")


# ---------------------------------------------------------------------------------------------------
# Second (new-path) legs: the unified loader must reproduce each baseline exactly (4a acceptance)
# ---------------------------------------------------------------------------------------------------

import yaml  # noqa: E402

from interpretune.config.loading import load_session_cfg, session_body_from_cli_mapping  # noqa: E402


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_unified_loader_matches_frozen_baseline(config_path):
    """The loader must keep producing what the (now-removed) jsonargparse path produced.

    45/45 live equivalence was proven BEFORE the old path was removed; its final captures are frozen as a committed
    fixture, so post-removal regressions in the loader (or accidental semantic edits to the experiment configs) still
    fail against the validated baseline.
    """
    mapping = yaml.safe_load(config_path.read_text(encoding="utf-8"))["session_cfg"]
    loaded = load_session_cfg(session_body_from_cli_mapping(mapping))
    assert json.loads(json.dumps(session_spec(loaded), default=str)) == _baseline_for(config_path)


@pytest.mark.parametrize("config_path", EXAMPLE_CONFIGS, ids=lambda p: p.stem)
def test_unified_loader_matches_factory_core(config_path):
    """For hub-shaped bodies the loader must equal the factory primitive (defaults fns excluded).

    The example REGISTRY additionally applies example_{datamodule,itmodule}_defaults via its register func; where those
    defaults live post-centralization is a 4c design item, so this leg compares the loader against the same merge-site
    primitive it wraps (instantiate_or_import), which is also what guarantees AutoComp synthesis parity.
    """
    from interpretune.registry import instantiate_or_import
    from tests.core.loader_equivalence import normalize
    from it_examples.example_module_registry import load_config_file

    # fresh body per leg: instantiate_or_import mutates nested class_path nodes in place
    key, factory_body = load_config_file(config_path)
    dm_cfg, m_cfg, dm_cls, m_cls = instantiate_or_import(
        dict(factory_body["registered_cfg"]), dict(factory_body["shared_config"]), None, None, None, None
    )
    _, loader_body = load_config_file(config_path)
    loaded = load_session_cfg(loader_body, expected_key=key)
    assert normalize(loaded.datamodule_cfg) == normalize(dm_cfg)
    assert normalize(loaded.module_cfg) == normalize(m_cfg)
    assert normalize(loaded.datamodule_cls) == normalize(dm_cls)
    assert normalize(loaded.module_cls) == normalize(m_cls)
    # AutoComp parity travels with the factories: synthesized classes must match by name on both paths
    assert type(loaded.module_cfg).__qualname__ == type(m_cfg).__qualname__


class TestOneGrammarRecursionRule:
    """Directional pins for the type-aware recursion skip (umbrella ruling on the defaults fork).

    Neither regression direction may go silent: a dict-typed field carrying a class_path-shaped dict
    must NOT instantiate (declarative — the optimizer_init case), and a class-typed field MUST.
    """

    def test_dict_typed_field_stays_declarative(self):
        from interpretune.registry import it_cfg_factory

        cfg = it_cfg_factory(
            {
                "class_path": "interpretune.config.module.ITConfig",
                "init_args": {
                    "optimizer_init": {"class_path": "torch.optim.AdamW", "init_args": {"lr": 1.0e-3}},
                },
            },
            {"model_name_or_path": "gpt2", "task_name": "rte"},
        )
        assert isinstance(cfg.optimizer_init, dict), "declarative dict-typed field was instantiated"
        assert cfg.optimizer_init["class_path"] == "torch.optim.AdamW"

    def test_class_typed_field_still_instantiates(self):
        from interpretune.config.mixins import HFFromPretrainedConfig
        from interpretune.registry import it_cfg_factory

        cfg = it_cfg_factory(
            {
                "class_path": "interpretune.config.module.ITConfig",
                "init_args": {
                    "hf_from_pretrained_cfg": {
                        "class_path": "interpretune.config.mixins.HFFromPretrainedConfig",
                        "init_args": {"pretrained_kwargs": {"device_map": "cpu"}},
                    },
                },
            },
            {"model_name_or_path": "gpt2", "task_name": "rte"},
        )
        assert isinstance(cfg.hf_from_pretrained_cfg, HFFromPretrainedConfig), (
            "class-typed field was left as a raw dict"
        )


def test_materialized_defaults_present_and_declarative():
    """The former example defaults are MATERIALIZED into every published configuration (v3 ruling).

    Pins both halves of the migration: the values are present in each published body's loader output
    (per-config self-containment — no load-time injection exists to supply them anymore), and the
    optimizer/scheduler entries remain DECLARATIVE dicts (the one-grammar recursion rule), ready for
    configure_optimizers-time instantiation.
    """
    from interpretune.config.loading import load_session_cfg
    from it_examples.example_module_registry import load_config_file

    for config_path in EXAMPLE_CONFIGS:
        key, body = load_config_file(config_path)
        loaded = load_session_cfg(body, expected_key=key)
        assert loaded.datamodule_cfg.prepare_data_map_cfg == {"batched": True}, key
        assert isinstance(loaded.module_cfg.optimizer_init, dict), key
        assert loaded.module_cfg.optimizer_init["class_path"] == "torch.optim.AdamW", key
        assert loaded.module_cfg.lr_scheduler_init["init_args"]["T_mult"] == 2, key
