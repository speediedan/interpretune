"""Core as the first consumer of the tuning conformance cases (#760): finetuning-scheduler's view contract, run
against bundled compositions through the same ``ConformanceTarget`` an adapter repository declares."""

from __future__ import annotations

import pytest

from interpretune.testing.conformance import ConformanceTarget, TuningConformance
from interpretune.utils.import_utils import package_available


class TestBridgeSplicedSAETuning(TuningConformance):
    """gpt2 over the TransformerBridge with an SAE spliced in: a trainable view with a `latent:sae` role."""

    target = ConformanceTarget(
        composition=("core", "sae_lens"), datamodule_flavour="bridge", module_cfg_extras={"add_saes_on_init": True}
    )


@pytest.mark.skipif(not package_available("nnsight"), reason="nnsight is not installed")
class TestNNsightTuning(TuningConformance):
    """A local nnsight composition registers its HF module, so its identity view runs the full cases (#774)."""

    target = ConformanceTarget(
        composition=("core", "nnsight", "sae_lens"), forward_family="hf_native", datamodule_flavour="nnsight"
    )

    def test_view_is_trainable(self, probe):
        """A refusing view would pass the cases above with only the refusal case run, so require the real view."""
        assert probe.support().trainable, probe.support().reason


def test_a_frozen_spliced_sae_is_an_instrument(request):
    """An SAE configured ``frozen`` is spliced and applied but declared ``instrument``; the others stay
    ``latent:sae`` (#812)."""
    from copy import deepcopy
    from dataclasses import replace

    from interpretune.session import ITSession

    cfg = deepcopy(request.getfixturevalue("get_it_session_cfg__sl_br_gpt2"))
    sae_cfgs = list(cfg.module_cfg.sae_cfgs)
    assert len(sae_cfgs) >= 2, "the fixture needs two SAEs to show one frozen beside one not"
    cfg.module_cfg.sae_cfgs = [replace(sae_cfgs[0], frozen=True), *sae_cfgs[1:]]
    cfg.module_cfg.add_saes_on_init = True
    module = ITSession(cfg).module
    roles = module.tunable_roles()
    frozen_key, other_key = (module.attached_sae_key(s.sae_id) for s in sae_cfgs[:2])
    by_key = {k: {r for n, r in roles.items() if n.startswith(f"attached_saes.{k}.")} for k in (frozen_key, other_key)}
    assert by_key == {frozen_key: {"instrument"}, other_key: {"latent:sae"}}
