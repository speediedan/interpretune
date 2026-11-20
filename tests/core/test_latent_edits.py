"""Native latent frame and chart edits at the latent sub-hook (#1091): each op moves a block's decoded values by
exactly the catalog's own edit, on a synthetic catalog and through both CPU backends' forward passes."""

from __future__ import annotations

import types

import pytest
import torch

pytest.importorskip("venngram", reason="the venngram extra is not installed")

from interpretune.analysis.backends import (  # noqa: E402
    InterventionSpec,
    PositionScope,
    apply_intervention,
    check_tensor_op,
    require_attached_latent_models,
)
from interpretune.analysis.backends import ops as op_registry  # noqa: E402
from interpretune.analysis.points.vocabulary import parse  # noqa: E402
from interpretune.analysis.venngram import (  # noqa: E402
    LATENT_EDIT_OPS,
    LatentBinding,
    LatentCatalog,
    LatentChart,
    LatentEdit,
    LatentFrame,
    LatentTarget,
    block_edit_spec,
    latent_edit_spec,
)
from interpretune.analysis.ops.bundled.latent import latent_ops  # noqa: E402
from interpretune.utils import MisconfigurationException  # noqa: E402

WIDTH, BLOCK, N_BLOCKS, UNIT = 6, 2, 4, 1
HOOK = "blocks.3.hook_resid_post"
COLUMNS = (UNIT * BLOCK, UNIT * BLOCK + 1)
ATOL = 1e-5
ALL = PositionScope.ALL_POSITIONS


def _handle(decoder: torch.Tensor, hook: str = HOOK):
    """A stand-in latent model: the hook it is attached at and its decoder rows, as a sae_lens SAE exposes them."""
    return types.SimpleNamespace(
        cfg=types.SimpleNamespace(metadata=types.SimpleNamespace(hook_name=hook)), W_dec=decoder
    )


def _catalog(decoder: torch.Tensor, handle, *, frame_rows: torch.Tensor | None = None):
    point = parse(handle.cfg.metadata.hook_name)
    binding = LatentBinding("sae", point.layer, point.base, handle, decoder, BLOCK)
    target = LatentTarget("feature", "sae", point.layer, binding.point, UNIT)
    block = decoder[UNIT * BLOCK : (UNIT + 1) * BLOCK].detach().to(torch.float64)
    rows = torch.linalg.qr(block.T).Q.T.contiguous() if frame_rows is None else frame_rows
    frame = LatentFrame("f", "1", target, rows)
    rank = rows.shape[0]
    ring = LatentChart(
        "ring",
        "1",
        "pca_plane",
        target,
        "f",
        "1",
        2,
        {1: 2 * torch.pi},
        {"origin": torch.tensor([0.1, -0.2], dtype=torch.float64), "axes": torch.eye(rank, dtype=torch.float64)},
    )
    # one dimension of a two-dimensional frame: the other frame coordinate is off this chart
    line = LatentChart(
        "line",
        "1",
        "affine",
        target,
        "f",
        "1",
        1,
        {},
        {"origin": torch.zeros(rank, dtype=torch.float64), "axes": torch.tensor([[0.6, 0.8]], dtype=torch.float64)},
    )
    return LatentCatalog([binding], frames=[frame], charts=[ring, line]), target


@pytest.fixture
def synthetic():
    generator = torch.Generator().manual_seed(1091)
    decoder = torch.randn(N_BLOCKS * BLOCK, WIDTH, generator=generator)
    handle = _handle(decoder)
    catalog, target = _catalog(decoder, handle)
    activations = torch.rand(2, 5, N_BLOCKS * BLOCK, generator=generator) + 0.5
    return types.SimpleNamespace(decoder=decoder, handle=handle, catalog=catalog, target=target, acts=activations)


EDITS = {
    "frame_replace": dict(name="f", coordinates=[0.75, -1.25]),
    "chart_set": dict(name="ring", value=[1.5, 0.5]),
    "chart_path": dict(name="ring", waypoints=[[1.0, 3.0], [2.0, -3.0]], t=0.5),
    "chart_rotate": dict(name="ring", by={1: 1.0}),
    "chart_scale_norm": dict(name="ring", factor=1.75),
}


def _catalog_edit(catalog, target, op, values, *, name, **args):
    """The functional result the native op must reproduce on the block's decoded values."""
    if op == "frame_replace":
        coordinates = torch.as_tensor(args["coordinates"], dtype=values.dtype).expand(*values.shape[:-1], 2)
        return catalog.replace_coordinates(target, values, coordinates, frame=name, version="1")
    return catalog.apply_chart(target, values, chart=name, version="1", op=LATENT_EDIT_OPS[op], **args)


def _decoded(acts: torch.Tensor, decoder: torch.Tensor) -> torch.Tensor:
    return acts[..., list(COLUMNS)] @ decoder[list(COLUMNS)]


class TestTheEditMovesTheDecodedBlockByTheCatalogsEdit:
    @pytest.mark.parametrize("op", sorted(EDITS))
    def test_each_op_matches_the_functional_result(self, synthetic, op):
        args = dict(EDITS[op])
        hook, spec = latent_edit_spec(
            synthetic.catalog, synthetic.target, op, version="1", position_scope=PositionScope.ALL_POSITIONS, **args
        )
        assert hook == f"{HOOK}.hook_sae_acts_post" and spec.unit_indices == COLUMNS
        edited = apply_intervention(synthetic.acts.clone(), spec, last_pos=4)
        values = _decoded(synthetic.acts, synthetic.decoder)
        expected = _catalog_edit(synthetic.catalog, synthetic.target, op, values, **args)
        assert not torch.allclose(expected, values, atol=1e-3), f"{op}: the oracle edit moves nothing"
        torch.testing.assert_close(_decoded(edited, synthetic.decoder), expected, atol=ATOL, rtol=0)
        others = [c for c in range(N_BLOCKS * BLOCK) if c not in COLUMNS]
        assert torch.equal(edited[..., others], synthetic.acts[..., others])

    def test_the_last_token_scope_leaves_every_other_position(self, synthetic):
        _, spec = latent_edit_spec(
            synthetic.catalog, synthetic.target, "chart_rotate", name="ring", version="1", by={1: 1.0}
        )
        edited = apply_intervention(synthetic.acts.clone(), spec, last_pos=4)
        assert torch.equal(edited[:, :4], synthetic.acts[:, :4])
        assert not torch.allclose(edited[:, 4], synthetic.acts[:, 4])

    def test_the_dose_interpolates_from_the_identity(self, synthetic):
        def run(dose):
            _, spec = latent_edit_spec(
                synthetic.catalog, synthetic.target, "chart_set", name="ring", version="1", dose=dose, value=[1.5, 0.5]
            )
            return apply_intervention(synthetic.acts.clone(), spec, last_pos=4)

        assert torch.equal(run(0.0), synthetic.acts)
        torch.testing.assert_close(run(0.5), (run(0.0) + run(1.0)) / 2, atol=ATOL, rtol=0)

    def test_a_rank_deficient_block_moves_by_the_minimum_norm_change(self):
        generator = torch.Generator().manual_seed(7)
        decoder = torch.randn(N_BLOCKS * BLOCK, WIDTH, generator=generator, dtype=torch.float64)
        decoder[COLUMNS[1]] = 2.0 * decoder[COLUMNS[0]]  # the block spans one direction
        direction = decoder[COLUMNS[0]] / decoder[COLUMNS[0]].norm()
        handle = _handle(decoder)
        catalog, target = _catalog_without_charts(decoder, handle, direction[None])
        acts = torch.rand(1, 3, N_BLOCKS * BLOCK, generator=generator, dtype=torch.float64)
        _, spec = latent_edit_spec(
            catalog, target, "frame_replace", name="f", version="1", position_scope=ALL, coordinates=[0.5]
        )
        edited = apply_intervention(acts.clone(), spec, last_pos=2)
        values = _decoded(acts, decoder)
        expected = catalog.replace_coordinates(
            target, values, torch.full((1, 3, 1), 0.5, dtype=values.dtype), frame="f", version="1"
        )
        torch.testing.assert_close(_decoded(edited, decoder), expected, atol=1e-10, rtol=0)
        change = (edited - acts)[..., list(COLUMNS)]
        # the minimum-norm change lies in the row space of the block's decoder Gram, here along (1, 2)
        torch.testing.assert_close(change[..., 1], 2.0 * change[..., 0], atol=1e-10, rtol=0)


def _catalog_without_charts(decoder, handle, frame_rows):
    point = parse(handle.cfg.metadata.hook_name)
    binding = LatentBinding("sae", point.layer, point.base, handle, decoder, BLOCK)
    target = LatentTarget("feature", "sae", point.layer, binding.point, UNIT)
    return LatentCatalog([binding], frames=[LatentFrame("f", "1", target, frame_rows)]), target


class TestPlantedDefectsFail:
    def test_a_wrong_frame_version_is_refused_by_name(self, synthetic):
        with pytest.raises(MisconfigurationException, match="unknown latent frame 'f' version '2'"):
            latent_edit_spec(
                synthetic.catalog, synthetic.target, "frame_replace", name="f", version="2", coordinates=[0, 0]
            )
        with pytest.raises(MisconfigurationException, match="unknown latent chart 'ring' version '0'"):
            latent_edit_spec(synthetic.catalog, synthetic.target, "chart_rotate", name="ring", version="0", by={1: 1.0})

    def test_dropping_the_off_chart_residual_breaks_the_match(self, synthetic, monkeypatch):
        """The positive control's comparison is sensitive: an edit writing the chart's point alone, dropping the
        frame coordinate off a one-dimensional chart, no longer matches the catalog's result."""
        args = dict(name="line", value=[1.25])
        _, spec = latent_edit_spec(
            synthetic.catalog, synthetic.target, "chart_set", version="1", position_scope=ALL, **args
        )
        values = _decoded(synthetic.acts, synthetic.decoder)
        expected = _catalog_edit(synthetic.catalog, synthetic.target, "chart_set", values, **args)
        torch.testing.assert_close(
            _decoded(apply_intervention(synthetic.acts.clone(), spec, last_pos=4), synthetic.decoder),
            expected,
            atol=ATOL,
            rtol=0,
        )

        def drops_the_residual(self, values):
            chart = self.catalog.chart(self.target, name=self.name, version=self.version)
            frame = self.catalog.frame(self.target, name=chart.frame, version=chart.frame_version)
            rows = frame.rows.to(values)
            point = torch.as_tensor(self.args["value"], dtype=values.dtype).expand(*values.shape[:-1], 1)
            return values + (chart.handler.from_chart(chart, point) - values @ rows.T) @ rows

        monkeypatch.setattr(LatentEdit, "edit_values", drops_the_residual)
        defective = _decoded(apply_intervention(synthetic.acts.clone(), spec, last_pos=4), synthetic.decoder)
        assert not torch.allclose(defective, expected, atol=1e-3)

    def test_a_chart_op_its_kind_lacks_is_refused_by_name(self, synthetic):
        _, spec = latent_edit_spec(
            synthetic.catalog, synthetic.target, "chart_rotate", name="line", version="1", by={0: 1.0}
        )
        with pytest.raises(MisconfigurationException, match="does not support 'rotate'"):
            apply_intervention(synthetic.acts.clone(), spec, last_pos=4)


class TestTheAttachedModelIsChecked:
    def _payload(self, synthetic):
        hook, spec = latent_edit_spec(
            synthetic.catalog, synthetic.target, "chart_rotate", name="ring", version="1", by={1: 1.0}
        )
        return {hook: spec}, {hook: [hook]}

    def test_the_resolved_model_attached_at_its_hook_passes(self, synthetic):
        payload, matches = self._payload(synthetic)
        require_attached_latent_models(payload, matches, [synthetic.handle], flow="value")

    def test_another_object_at_the_same_hook_is_refused(self, synthetic):
        payload, matches = self._payload(synthetic)
        impostor = _handle(synthetic.decoder.clone())
        with pytest.raises(ValueError, match="is not the loaded model the target resolved to"):
            require_attached_latent_models(payload, matches, [impostor], flow="value")

    def test_no_model_attached_there_is_refused(self, synthetic):
        payload, matches = self._payload(synthetic)
        elsewhere = _handle(synthetic.decoder, hook="blocks.1.hook_resid_post")
        with pytest.raises(ValueError, match="sub-hook of an attached latent model; attached at"):
            require_attached_latent_models(payload, matches, [elsewhere], flow="value")

    def test_changed_decoder_rows_are_refused(self, synthetic):
        payload, matches = self._payload(synthetic)
        synthetic.handle.W_dec = synthetic.decoder.clone()
        synthetic.handle.W_dec[COLUMNS[0], 0] += 1.0
        with pytest.raises(ValueError, match="decoder rows for the block differ"):
            require_attached_latent_models(payload, matches, [synthetic.handle], flow="value")

    def test_the_gradient_flow_is_refused(self, synthetic):
        payload, matches = self._payload(synthetic)
        with pytest.raises(ValueError, match="runs on the forward flow, not 'grad'"):
            require_attached_latent_models(payload, matches, [synthetic.handle], flow="grad")

    def test_a_model_attached_at_another_point_is_refused_when_compiled(self, synthetic):
        binding = synthetic.catalog.binding(synthetic.target)
        object.__setattr__(binding, "model", _handle(synthetic.decoder, hook="blocks.3.hook_resid_pre"))
        with pytest.raises(MisconfigurationException, match="is attached at 'blocks.3.hook_resid_pre'"):
            latent_edit_spec(synthetic.catalog, synthetic.target, "chart_rotate", name="ring", version="1", by={1: 1.0})


class TestBlockEdits:
    def test_set_ablate_and_scale_norm_are_core_ops_on_the_block(self, synthetic):
        values = _decoded(synthetic.acts, synthetic.decoder)
        source = torch.tensor([0.25, 2.0])
        for op, value, expected in (
            ("set", source, (source @ synthetic.decoder[list(COLUMNS)]).expand_as(values)),
            ("ablate", None, torch.zeros_like(values)),
            ("scale_norm", 1.5, 1.5 * values),
        ):
            hook, spec = block_edit_spec(
                synthetic.catalog, synthetic.target, op, value=value, position_scope=PositionScope.ALL_POSITIONS
            )
            assert hook == f"{HOOK}.hook_sae_acts_post" and spec.op in ("replace", "scale") and spec.params is None
            edited = apply_intervention(synthetic.acts.clone(), spec, last_pos=4)
            torch.testing.assert_close(_decoded(edited, synthetic.decoder), expected, atol=ATOL, rtol=0)

    @pytest.mark.parametrize(
        "op, value, match",
        [
            ("set", [1.0], "needs a value of 2 activations"),
            ("scale_norm", -1.0, "positive factor"),
            ("tilt", None, "unknown block edit"),
        ],
    )
    def test_a_malformed_block_edit_is_refused_by_name(self, synthetic, op, value, match):
        with pytest.raises(MisconfigurationException, match=match):
            block_edit_spec(synthetic.catalog, synthetic.target, op, value=value)


class TestTheOpsAreTensorOps:
    def test_the_bundled_collection_registers_every_op(self):
        from interpretune.analysis.ops.dispatcher import DISPATCHER

        DISPATCHER.load_definitions()
        for op in LATENT_EDIT_OPS:
            record = op_registry.executor_record(op)
            assert (record.source, record.collection) == ("bundled", "latent")
            assert record.executor is latent_ops.latent_edit_executor

    def test_the_executor_passes_the_contract_only_on_its_probe(self):
        check_tensor_op(
            "frame_replace",
            latent_ops.latent_edit_executor,
            operand_kinds=("none",),
            probe=latent_ops.probe_latent_edit,
        )
        with pytest.raises(ValueError, match="raised ValueError on a probe"):
            check_tensor_op("frame_replace", latent_ops.latent_edit_executor, operand_kinds=("none",))

    def test_a_spec_without_a_compiled_edit_is_refused_by_name(self, synthetic):
        spec = InterventionSpec(torch.tensor(1.0), op="chart_rotate", unit_indices=COLUMNS)
        with pytest.raises(ValueError, match="carries none in params"):
            apply_intervention(synthetic.acts.clone(), spec, last_pos=4)


# --------------------------------------------------------------------------------------------------------------------
# Both CPU backends: gpt2 with a real attached SAE
# --------------------------------------------------------------------------------------------------------------------

IDS = torch.tensor([[464, 3139, 286, 4881, 318]])  # "The capital of France is"


@pytest.mark.parametrize(
    "session_fixture, batch_key",
    [
        pytest.param("get_it_session__sl_br_gpt2__initonly", "input", id="transformer_lens"),
        pytest.param("get_it_session__sl_ns_gpt2__initonly", "input_ids", id="nnsight"),
    ],
)
class TestBothBackends:
    def _setup(self, request, session_fixture, batch_key):
        module = request.getfixturevalue(session_fixture).it_session.module
        handle = module.sae_handles[0]
        hook = f"{handle.cfg.metadata.hook_name}.hook_sae_acts_post"
        batch = {batch_key: IDS}
        backend = module.model_backend
        _, cache = backend.fwd_w_cache_and_latent_models(
            model=module.model, batch=batch, latent_model_handles=[handle], names_filter=hook
        )
        clean = torch.as_tensor(cache[hook])[0, -1].detach()
        unit = int(clean.argmax()) // BLOCK
        point = parse(handle.cfg.metadata.hook_name)
        binding = LatentBinding("sae", point.layer, point.base, handle, handle.W_dec, BLOCK)
        target = LatentTarget("feature", "sae", point.layer, binding.point, unit)
        block = handle.W_dec[unit * BLOCK : (unit + 1) * BLOCK].detach().to(torch.float64)
        frame = LatentFrame("f", "1", target, torch.linalg.qr(block.T).Q.T.contiguous())
        ring = LatentChart(
            "ring",
            "1",
            "pca_plane",
            target,
            "f",
            "1",
            2,
            {1: 2 * torch.pi},
            {"origin": torch.zeros(2, dtype=torch.float64), "axes": torch.eye(2, dtype=torch.float64)},
        )
        catalog = LatentCatalog([binding], frames=[frame], charts=[ring])
        return module, backend, handle, hook, batch, clean, catalog, target

    def test_a_chart_edit_runs_natively_as_its_replacement_does(self, request, session_fixture, batch_key):
        """The native edit's logits equal those of writing the block's oracle activations (the catalog's edit of
        its decoded values, written back by least squares) with a core ``replace``; and they differ from the clean
        run."""
        module, backend, handle, hook, batch, clean, catalog, target = self._setup(request, session_fixture, batch_key)
        _, spec = latent_edit_spec(catalog, target, "chart_rotate", name="ring", version="1", by={1: 2.0})
        columns = list(spec.unit_indices)
        rows = handle.W_dec[columns].detach().to(torch.float64)
        values = clean[columns].to(torch.float64) @ rows
        edited = catalog.apply_chart(target, values, chart="ring", version="1", op="rotate", by={1: 2.0})
        change = torch.linalg.lstsq(rows.T, (edited - values)[:, None]).solution[:, 0]
        oracle = InterventionSpec(
            (clean[columns].to(torch.float64) + change).to(clean.dtype), op="replace", unit_indices=spec.unit_indices
        )

        pre, post = backend.fwd_w_intervention(module.model, batch, {hook: spec}, latent_model_handles=[handle])
        _, reference = backend.fwd_w_intervention(module.model, batch, {hook: oracle}, latent_model_handles=[handle])
        assert not torch.allclose(post[:, -1], pre[:, -1], atol=1e-3)
        torch.testing.assert_close(post, reference, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(post[:, :-1], pre[:, :-1], atol=1e-5, rtol=0)

    def test_an_edit_for_another_loaded_model_is_refused_before_any_pass(self, request, session_fixture, batch_key):
        module, backend, handle, hook, batch, _, catalog, target = self._setup(request, session_fixture, batch_key)
        _, spec = latent_edit_spec(catalog, target, "chart_rotate", name="ring", version="1", by={1: 2.0})
        impostor = types.SimpleNamespace(cfg=handle.cfg, W_dec=handle.W_dec)
        with pytest.raises(ValueError, match="is not the loaded model the target resolved to"):
            backend.fwd_w_intervention(module.model, batch, {hook: spec}, latent_model_handles=[impostor])


class TestValidationRunsOnTheCheckedTensorsDevice:
    """A frame, chart or linear space checks its tensors on their own device, not the default one (#1096).

    A meta default device stands in for a GPU on a CPU-only host: every tensor here is on the CPU, so a check that
    builds its identity on the default device compares a CPU tensor with a meta one and fails.
    """

    def test_a_frame_a_chart_and_a_space_register_whatever_the_default_device(self, synthetic):
        from interpretune.analysis.venngram import LinearSpace

        cpu = dict(dtype=torch.float64, device="cpu")
        rows = torch.linalg.qr(synthetic.decoder[list(COLUMNS)].to(**cpu).T).Q.T.contiguous()
        matrix = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.5, 0.0]], **cpu)
        with torch.device("meta"):
            frame = LatentFrame("f", "2", synthetic.target, rows)
            chart = LatentChart(
                "ring",
                "2",
                "pca_plane",
                synthetic.target,
                "f",
                "2",
                2,
                {1: 2 * torch.pi},
                {"origin": torch.zeros(2, **cpu), "axes": torch.eye(2, **cpu)},
            )
            binding = synthetic.catalog.binding(synthetic.target)
            catalog = LatentCatalog([binding], frames=[frame], charts=[chart])
            space = LinearSpace("jlens", "1", matrix)
        assert catalog.chart(synthetic.target, name="ring", version="2").frame_version == "2"
        assert space.dim == 2
