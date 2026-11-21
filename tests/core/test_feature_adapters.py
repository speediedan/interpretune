"""Unit adapters over a latent model's feature blocks (#868), and the roles of a weight edit on a latent model's
tensor (#1098)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytest.importorskip("venngram.tuning", reason="venngram, which holds the tunable view protocol, is not installed")

from venngram.tuning import CanonicalView  # noqa: E402

from interpretune.analysis.venngram.latents import LatentBinding, LatentCatalog, SubspaceBinding  # noqa: E402
from interpretune.base.modules import BaseITModule  # noqa: E402
from interpretune.tuning.latents import LatentSite, attach_latent_model  # noqa: E402
from interpretune.tuning.weights import attach_feature_adapter  # noqa: E402

D, BLOCK, BLOCKS = 4, 3, 2  # site width, coordinates per block, blocks
WIDTH = BLOCK * BLOCKS


class Featurizer(nn.Module):
    """A block-sparse featurizer stand-in: ``BLOCKS`` blocks of ``BLOCK`` latent coordinates."""

    def __init__(self, d: int = D, width: int = WIDTH) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.enc, self.dec = nn.Linear(d, width), nn.Linear(width, d)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.enc(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)


class Root(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.ModuleDict({"blocks": nn.ModuleList([nn.Linear(D, D), nn.Linear(D, D)])})


def _spliced(frozen: bool = False, featurizer: nn.Module | None = None):
    root, featurizer = Root(), featurizer or Featurizer()
    attach_latent_model(root, "bsf", featurizer, LatentSite("model.blocks.1"), frozen=frozen)
    catalog = LatentCatalog([LatentBinding("bsf", 1, "hook_resid_post", featurizer, featurizer.dec.weight.T, BLOCK)])
    return root, featurizer, catalog


def _roles(root: nn.Module) -> dict:
    """The roles of ``root``'s view, declared as every interpretune module declares them."""
    return dict(CanonicalView(root, roles=BaseITModule.tunable_roles(root)).roles())  # type: ignore[arg-type]


class TestFeatureAdapters:
    def test_a_block_adapts_its_coordinates_on_each_tensor_with_a_latent_axis(self):
        root, featurizer, catalog = _spliced()
        handles = attach_feature_adapter(root, catalog, "features:bsf & units:[1]", name="blk", n_layers=2)
        edited = {h.target: h.edit for h in handles}
        # the encoder's rows and bias and the decoder's columns hold the latent axis; the decoder's bias does not
        assert set(edited) == {f"latent_models.bsf.{t}" for t in ("enc.weight", "enc.bias", "dec.weight")}
        assert all(e.units.tolist() == [3, 4, 5] for e in edited.values())  # block 1 of blocks of 3
        assert edited["latent_models.bsf.dec.weight"].delta.shape == (D, BLOCK)

    def test_only_the_selected_blocks_coordinates_move(self):
        root, featurizer, catalog = _spliced()
        before = {"enc.weight": featurizer.enc.weight.detach().clone()}
        before["dec.weight"] = featurizer.dec.weight.detach().clone()
        handles = attach_feature_adapter(root, catalog, "features:bsf & units:[0]", name="blk", n_layers=2)
        torch.testing.assert_close(featurizer.enc.weight, before["enc.weight"])  # zero at first
        with torch.no_grad():
            for h in handles:
                h.edit.delta.add_(1.0)
        moved = (featurizer.enc.weight - before["enc.weight"]).abs().sum(dim=1) > 0
        assert moved.tolist() == [True] * BLOCK + [False] * BLOCK
        moved = (featurizer.dec.weight - before["dec.weight"]).abs().sum(dim=0) > 0
        assert moved.tolist() == [True] * BLOCK + [False] * BLOCK
        for h in handles:
            h.remove()
        torch.testing.assert_close(featurizer.enc.weight, before["enc.weight"])

    def test_the_adapter_takes_its_role_beside_a_frozen_featurizers(self):
        """A frozen featurizer stays ``artifact`` while a schedule thaws its adapter at the chosen blocks."""
        root, _, catalog = _spliced(frozen=True)
        attach_feature_adapter(root, catalog, "features:bsf & units:[1]", name="blk", n_layers=2)
        roles = _roles(root)
        adapters = {n for n, r in roles.items() if r == "adapter:blk"}
        assert adapters == {
            f"latent_models.bsf.{t}.parametrizations.{leaf}.0.delta"
            for t, leaf in (("enc", "weight"), ("enc", "bias"), ("dec", "weight"))
        }
        assert roles["latent_models.bsf.enc.parametrizations.weight.original"] == "artifact"
        assert roles["latent_models.bsf.dec.bias"] == "artifact"

    def test_an_explicit_axis_settles_a_featurizer_as_wide_as_its_site(self):
        root, featurizer, catalog = _spliced(featurizer=Featurizer(d=WIDTH))
        with pytest.raises(ValueError, match="several axes of the latent width"):
            attach_feature_adapter(root, catalog, "features:bsf & units:[1]", name="blk", n_layers=2)
        assert not hasattr(featurizer.enc, "parametrizations")  # nothing attached when any tensor is refused
        axes = {"enc.weight": 0, "enc.bias": 0, "dec.weight": 1}
        handles = attach_feature_adapter(root, catalog, "features:bsf & units:[1]", name="blk", n_layers=2, axes=axes)
        # only the tensors `axes` names: the decoder's bias is as long as the latent width here, and is left alone
        assert {h.target.removeprefix("latent_models.bsf.") for h in handles} == set(axes)

    def test_refusals_name_what_is_wrong(self):
        root, featurizer, catalog = _spliced()
        loose = LatentCatalog([LatentBinding("other", 1, "hook_resid_post", Featurizer(), featurizer.dec.weight.T, 3)])
        with pytest.raises(ValueError, match="not registered under Root"):
            attach_feature_adapter(root, loose, "features:other & units:[0]", name="blk", n_layers=2)
        plane = SubspaceBinding("plane", 1, "hook_resid_post", torch.eye(D)[:2])
        spans = LatentCatalog([], [plane])
        with pytest.raises(ValueError, match="subspace 'plane'"):
            attach_feature_adapter(root, spans, "subspace:plane", name="blk", n_layers=2)
        with pytest.raises(ValueError, match=r"has no tensor \['enc.W'\]"):
            attach_feature_adapter(root, catalog, "features:bsf & units:[0]", name="blk", n_layers=2, axes={"enc.W": 0})
        with pytest.raises(ValueError, match="has no latent axis 1 of width 6"):
            attach_feature_adapter(
                root, catalog, "features:bsf & units:[0]", name="blk", n_layers=2, axes={"enc.weight": 1}
            )


class TestEditRolesUnderALatentModel:
    """A weight edit on a latent model's own tensor keeps the edit's role (#1098)."""

    def test_the_edits_role_wins_over_the_latent_models_address(self):
        root, _, catalog = _spliced()
        attach_feature_adapter(root, catalog, "features:bsf & units:[0]", name="blk", n_layers=2)
        roles = _roles(root)  # refused as conflicting role addresses before #1098
        assert roles["latent_models.bsf.dec.parametrizations.weight.0.delta"] == "adapter:blk"
        assert roles["latent_models.bsf.dec.parametrizations.weight.original"] == "latent:bsf"
        assert roles["latent_models.bsf.dec.bias"] == "latent:bsf"
