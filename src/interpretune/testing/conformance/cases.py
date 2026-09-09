"""The model-backend conformance cases. A repository subclasses ``ModelBackendConformance`` and sets ``target``.

Every case goes through the runner with inputs constructed the way a caller constructs them, and asserts the declared
return type before reading a value. A case that must reach the backend directly says so in its docstring and is the
exception.
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

import pytest
import torch

from interpretune.analysis.backends import (
    AnalysisBackendCapability,
    BackendCapability,
    InterventionMode,
    PositionScope,
    SupportsGradients,
    SupportsIntervention,
    SupportsLatentModels,
)
from interpretune.analysis.optools import require_backend_capability

from .gates import UNDECLARED, conformance_case, gate_of
from .plugin import _REPORT_KEY
from .inputs import ConformanceInputs, ConformanceTarget
from .oracles import (
    expect_refusal,
    CHANGED_ATOL,
    CONVERGENCE_ATOL,
    STEER_SCALE,
    assert_non_degenerate,
    changed_positions,
    expected_positions,
    steering_vector,
)
from .ops import captured_points
from .reference import HFReference
from .session import ConformanceSession, build_conformance_session, tokenized_prompts

#: Tolerances for the real positions of a left-padded dataset batch, looser than the unpadded calibration
#: (1e-4) and RELATIVE as well as absolute. Measured: on Linux the bridge and a plain HF forward agree to 4.6e-4
#: absolute on gpt2's logits; on the Windows CPU runner the deepest points (the unembed input after twelve
#: blocks, and the logits) exceeded 1e-3 absolute while the shallow points passed, which is accumulated
#: platform drift rather than a wrong tensor. Logits sit at magnitude ~100, where 1e-3 absolute is 1e-5
#: relative. Pad positions are undefined and never compared.
PADDED_RTOL = 1e-3
PADDED_ATOL = 1e-3

_PROTOCOL_FOR = {
    BackendCapability.LATENT_MODELS: SupportsLatentModels,
    BackendCapability.GRADIENTS: SupportsGradients,
    BackendCapability.INTERVENTION: SupportsIntervention,
}


def _require_discriminating_length(attention_mask: torch.Tensor | None, seq_len: int, *, what: str) -> None:
    """The two scopes are indistinguishable below two real positions (both edit `{0}`), so a short input would pass
    a backend that ignores scope entirely.

    The discriminating power lives in the fixture, so the fixture is asserted before any set is compared. Measured on
    gpt2: n=1 gives `{0}` for both scopes.
    """
    real = seq_len if attention_mask is None else int(attention_mask.sum(dim=-1).min())
    assert real >= 2, (
        f"{what}: only {real} real position(s); the position scopes are indistinguishable below 2, so the exact-set "
        "cases would pass a backend that ignores scope entirely"
    )


def _assert_close_padded(got: torch.Tensor, ref: torch.Tensor, *, what: str) -> None:
    """Padded-batch convergence, with the measured differences IN the message so a platform failure reports
    numbers."""
    got = got.to(torch.float32)
    ref = ref.to(torch.float32)
    diff = (got - ref).abs()
    ok = diff <= PADDED_ATOL + PADDED_RTOL * ref.abs()
    if bool(ok.all()):
        return
    rel = (diff / ref.abs().clamp_min(1e-12)).max().item()
    raise AssertionError(
        f"{what} diverged from the HF forward (rtol={PADDED_RTOL}, atol={PADDED_ATOL}): greatest absolute difference "
        f"{diff.max().item():.3e}, greatest relative difference {rel:.3e}, {int((~ok).sum())} of {ok.numel()} elements"
    )


def _real_effect(effect: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Zero the effect at pad positions: they are undefined by contract, and a padded forward may not even be
    deterministic there, so a discriminator that counted them would flag noise as movement."""
    if attention_mask is None:
        return effect
    return effect * attention_mask.to(effect.dtype)


def _captured_under(captured: dict[str, torch.Tensor], name: str, cmap: Any) -> str | None:
    """The key ``name`` was captured under: itself, or its canonical spelling on ``cmap``'s architecture."""
    from interpretune.analysis.points import TensorRef, parse, resolve

    if name in captured:
        return name
    point = parse(name)
    ref = resolve(point, cmap)
    if not isinstance(ref, TensorRef):
        return None
    for key in captured:
        other = resolve(parse(key), cmap)
        if isinstance(other, TensorRef) and other.module_path == ref.module_path and other.io == ref.io:
            if getattr(other, "derived", False) == getattr(ref, "derived", False):
                return key
    return None


def _attribution_failure_context(suite, exc: BaseException) -> str:
    """What the two model handles are bound to, for a graph op that failed inside the analysis backend.

    Three probes, each on its own so one failing does not hide the others: the handles and their configured attention
    implementation; the provenance of the attention function the modeling module binds at its call site, since that is
    the source nnsight's recursive tracing parses, and a wrapper installed there by another library is what circuit-
    tracer's attention-pattern location fails on; and the layer-0 attention source nodes.
    """
    lines = [f"attribution graph op failed: {type(exc).__name__}: {str(exc).splitlines()[-1][:200]}"]
    module = suite.module
    for label in ("model", "replacement_model"):
        handle = getattr(module, label, None)
        inner = handle
        for attr in ("_model", "model", "hf_model"):
            inner = getattr(inner, attr, inner) if inner is not None else None
        cfg = getattr(inner, "config", None) or getattr(handle, "config", None)
        impl = getattr(cfg, "_attn_implementation", None)
        lines.append(
            f"  module.{label}: {type(handle).__name__} (inner {type(inner).__name__}), attn_implementation="
            f"{impl!r}, training={getattr(inner, 'training', None)}"
        )
    rm: Any = getattr(module, "replacement_model", None)
    try:
        import inspect

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        inner = rm.model if hasattr(rm, "model") else rm
        cfg = getattr(inner, "config", None)
        impl = getattr(cfg, "_attn_implementation", None)
        modeling = inspect.getmodule(type(inner))
        module_fn = getattr(modeling, "eager_attention_forward", None)
        bound = ALL_ATTENTION_FUNCTIONS.get_interface(str(impl), module_fn) if module_fn is not None else None
        for label, fn in (("modeling.eager_attention_forward", module_fn), ("selected interface", bound)):
            if fn is None:
                lines.append(f"  {label}: None")
                continue
            try:
                dropout = inspect.getsource(fn).count("dropout(")
            except (OSError, TypeError):
                dropout = "unreadable"
            own = modeling is not None and getattr(fn, "__module__", None) == modeling.__name__
            lines.append(
                f"  {label}: {getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', '?')} (the modeling"
                f" module's own: {own}; has __wrapped__: {hasattr(fn, '__wrapped__')}; dropout calls in source:"
                f" {dropout}; config impl {impl!r})"
            )
    except Exception as fn_exc:
        lines.append(f"  (attention function probe failed: {type(fn_exc).__name__}: {str(fn_exc)[:100]})")
    try:
        layer0 = rm.model.layers[0].self_attn  # the path circuit-tracer's attention pattern starts from
        names = [n for n in dir(layer0.source) if not n.startswith("_")]
        lines.append(f"  replacement_model layer 0 self_attn source nodes: {names}")
    except Exception as probe_exc:  # the probe must not hide the original failure
        lines.append(f"  (node probe failed: {type(probe_exc).__name__}: {str(probe_exc)[:120]})")
    return "\n".join(lines)


def _base_of(name: str) -> str:
    from interpretune.analysis.points.vocabulary import parse

    return parse(name).base


def _real_positions(attention_mask: torch.Tensor | None, tensor: torch.Tensor) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones(tensor.shape[:2], dtype=torch.bool)
    return attention_mask.bool()


class ModelBackendConformance:
    """Subclass, set ``target``, and pytest does the rest.

    Fixtures are class-scoped: one composed session and one HF reference per target class, and every case
    reads an extracted store rather than holding the live session.
    """

    target: ClassVar[ConformanceTarget]
    inputs: ClassVar[ConformanceInputs | None] = None

    # -- fixtures ----------------------------------------------------------------------------------

    @pytest.fixture(scope="class")
    def suite(self, request) -> ConformanceSession:
        """One composed session and runner per target class."""
        cls = request.cls
        inputs = cls.inputs or ConformanceInputs()
        return build_conformance_session(cls.target, inputs)

    @pytest.fixture(scope="class")
    def hf(self, suite) -> HFReference:
        """The library-independent HF reference, loaded once per class."""
        return HFReference(suite.inputs.model_id)

    @pytest.fixture(scope="class")
    def prompt_ids(self, hf, suite) -> list[torch.Tensor]:
        """The suite's prompts as unpadded id tensors."""
        return tokenized_prompts(hf.tokenizer, suite.inputs.prompts)

    @pytest.fixture(autouse=True)
    def _gate(self, request, suite):
        """Skip a case whose gate the target has not declared, with the reason the report keys on."""
        gate = gate_of(request.function)
        if gate is None:
            return
        report = request.config.stash.get(_REPORT_KEY, None)
        if report is not None and not report.declared:
            report.declared = sorted(c.name for c in suite.capabilities.model) + sorted(
                c.name for c in suite.capabilities.analysis
            )
        if report is not None:
            report.capture.setdefault(
                type(self).__name__,
                suite.capabilities.capture.describe() if suite.capabilities.capture else "undeclared",
            )
        if not gate.selects(suite.capabilities, family=suite.family, single_prompt=suite.target.single_prompt):
            pytest.skip(f"{UNDECLARED}: needs {gate.describe()}")

    # -- always-on ---------------------------------------------------------------------------------

    @conformance_case()
    def test_session_composes_and_declarations_are_coherent(self, suite):
        """A backend is attached, each declared surface satisfies its protocol, and records match declarations."""
        backend = suite.backend
        assert backend is not None, "the composed module attaches no model backend"
        assert isinstance(backend.capabilities, frozenset)
        for cap in backend.capabilities:
            assert isinstance(cap, BackendCapability), f"{cap!r} is not a BackendCapability member"
            protocol = _PROTOCOL_FOR[cap]
            assert isinstance(backend, protocol), (
                f"{suite.backend_name} declares {cap.name} but is not a {protocol.__name__}"
            )
        # the present-iff-declared invariant, from the live objects rather than the enum
        assert (suite.capabilities.intervention is not None) == (BackendCapability.INTERVENTION in backend.capabilities)
        assert (suite.capabilities.latent_models is not None) == (
            BackendCapability.LATENT_MODELS in backend.capabilities
        )

    @conformance_case()
    def test_undeclared_capabilities_are_refused_by_name(self, suite):
        """Every surface the backend does NOT claim must be refused by the shared gate, naming the backend and what
        it does claim.

        Checked at the gate rather than through an op, because the ops that need LATENT_MODELS and GRADIENTS also need
        latent models configured, which would fail first.
        """
        backend = suite.backend
        undeclared = [c for c in BackendCapability if c not in backend.capabilities]
        # A backend declaring every surface has nothing to refuse; that is a pass, not a skip, because the
        # claim "every surface is declared" is itself checked above and a skip here would count as "other".
        for cap in undeclared:
            with pytest.raises(ValueError, match=f"requires a model backend with {cap.name}"):
                require_backend_capability(backend, cap, "conformance")

    @conformance_case()
    def test_runner_produces_the_store_schema(self, suite):
        """`logit_diffs_base` yields the declared columns, one row per batch, with the declared dtypes."""
        import interpretune as it
        from interpretune import AnalysisCfg

        store = suite.run(AnalysisCfg(target_op=it.logit_diffs_base, save_tokens=True))
        n = suite.inputs.limit_batches
        for column in ("logit_diffs", "answer_logits", "preds", "orig_labels"):
            values = getattr(store, column)
            assert values is not None, f"{column} is absent from the store"
            assert len(values) == n, f"{column}: {len(values)} rows for {n} batches"
        for al in store.answer_logits:
            assert isinstance(al, torch.Tensor) and al.dtype == torch.float32, "answer_logits must be float32 tensors"
        for ld in store.logit_diffs:
            assert isinstance(ld, torch.Tensor), "logit_diffs must be tensors"

    # -- capture declaration ---------------------------------------------------------------------------
    #
    # A backend declares which vocabulary points it can capture on the model it wraps (`capture_support`), and the
    # capture cases select from that declaration rather than assuming every point. The declaration binds because a
    # point outside it is refused by name through the runner path, and because every point inside it is captured.

    @conformance_case()
    def test_the_backend_declares_what_it_can_capture(self, suite):
        """The attached backend declares a capture record for this model, and every suite capture point is decided
        by it (capturable or refused with a reason), never undeclared."""
        support = suite.capabilities.capture
        assert support is not None, (
            f"{suite.backend_name} declares no capture support: a model backend must say which vocabulary points "
            "it can capture on the model it wraps (a `capture_support(model)` method returning CaptureSupport)"
        )
        assert support.capturable, "the declaration names no capturable point"
        undeclared = [
            p for p in suite.inputs.capture_points if _base_of(p) not in support.capturable | set(support.uncapturable)
        ]
        assert not undeclared, f"suite capture points the declaration neither admits nor refuses: {undeclared}"

    @conformance_case()
    def test_every_declared_point_is_captured(self, suite):
        """Every base point the backend declares capturable is captured at the capture layer, as a non-degenerate
        tensor: the positive half of the declaration."""
        from interpretune import AnalysisCfg
        from interpretune.analysis.points import component_map_for
        from interpretune.analysis.points.inventory import spelled_at

        support = suite.capabilities.capture
        assert support is not None
        cmap = component_map_for(support.architecture)
        names = sorted(spelled_at(base, cmap, suite.inputs.capture_layer) for base in support.capturable)
        store = suite.run(AnalysisCfg(target_op="store_capture_points", names_filter=names, save_tokens=True))
        captured = captured_points(store, 0)
        # A backend may key the cache by the point's canonical spelling on this architecture rather than the one
        # asked for (a bridge answers `hook_mlp_out` as `mlp.hook_out` where no post-norm exists), which is the
        # same tensor by the vocabulary's own resolution; a name is missing only if neither spelling is there.
        found = {n: _captured_under(captured, n, cmap) for n in names}
        missing = [n for n, key in found.items() if key is None]
        assert not missing, f"declared capturable but not captured: {missing}; captured: {sorted(captured)}"
        for n, key in found.items():
            assert_non_degenerate(captured[cast(str, key)], what=n)

    @conformance_case()
    def test_a_point_outside_the_declaration_is_refused_by_name(self, suite):
        """The negative half: asking for a point the backend declares it cannot capture (or a layer the model lacks)
        is refused by name through the runner path, never answered with a cache that is silently short."""
        from interpretune import AnalysisCfg
        from interpretune.analysis.points import component_map_for
        from interpretune.analysis.points.inventory import spelled_at

        support = suite.capabilities.capture
        assert support is not None
        if support.uncapturable:
            base = sorted(support.uncapturable)[0]
            name = spelled_at(base, component_map_for(support.architecture), suite.inputs.capture_layer)
        else:
            name = f"blocks.{support.n_layers + 5}.hook_in"
        import re

        with expect_refusal(ValueError, match=re.escape(name)):
            suite.run(AnalysisCfg(target_op="store_capture_points", names_filter=[name], save_tokens=True))

    @conformance_case()
    def test_cache_op_stores_logits_and_every_requested_point(self, suite):
        """The cache path returns logits and every requested point reaches the store as a tensor."""
        from interpretune import AnalysisCfg

        points = suite.capturable_points()
        store = suite.run(AnalysisCfg(target_op="store_capture_points", names_filter=points, save_tokens=True))
        columns = list(store.dataset.column_names)
        for column in ("captured_values", "captured_shape", "captured_point_names"):
            assert column in columns, f"{column} is absent from the store; columns present: {columns}"
        assert store.answer_logits is not None and all(t is not None for t in store.answer_logits), (
            "answer_logits is None from the cache path; the backend's fwd_w_cache returned no logits"
        )
        for i in range(len(store["captured_values"])):
            captured = captured_points(store, i)
            missing = [p for p in points if p not in captured]
            assert not missing, f"cache lacks requested points {missing}"
            for p in points:
                assert isinstance(captured[p], torch.Tensor), f"{p} is not a tensor in the cache"

    @conformance_case()
    def test_block_output_is_the_next_block_input(self, suite):
        """`blocks.L.hook_out` and `blocks.L+1.hook_in` are the SAME residual tensor.

        Proves the layer index and the in/out slot mean what they say, with no reference at all.
        """
        from interpretune import AnalysisCfg

        layer = suite.inputs.capture_layer
        a, b = f"blocks.{layer}.hook_out", f"blocks.{layer + 1}.hook_in"
        store = suite.run(AnalysisCfg(target_op="store_capture_points", names_filter=[a, b, f"blocks.{layer}.hook_in"]))
        captured = captured_points(store, 0)
        torch.testing.assert_close(captured[a], captured[b], rtol=0, atol=0)
        assert not torch.allclose(captured[a], captured[f"blocks.{layer}.hook_in"]), "a block changed nothing"

    @conformance_case(family="hf_native")
    def test_capture_converges_on_the_forward(self, suite, hf):
        """Every captured point matches the HF module's tensor on the real positions of each batch."""
        from interpretune import AnalysisCfg

        points = suite.capturable_points()
        store = suite.run(AnalysisCfg(target_op="store_capture_points", names_filter=points, save_tokens=True))
        for i in range(len(store["captured_values"])):
            captured = captured_points(store, i)
            ids, mask = suite.batch_inputs(i)
            ref = hf.capture(ids, points, attention_mask=mask)
            real = _real_positions(mask, ref[points[0]])
            for p in points:
                assert_non_degenerate(captured[p], what=p)
                _assert_close_padded(captured[p][real], ref[p][real], what=f"{p} on the real positions of batch {i}")

    @conformance_case(family="hf_native")
    def test_answer_logits_converge_on_the_forward(self, suite, hf):
        """The store's logits match the HF forward on real positions."""
        from interpretune import AnalysisCfg

        store = suite.run(
            AnalysisCfg(target_op="store_capture_points", names_filter=[suite.capturable_points()[0]], save_tokens=True)
        )
        for i, al in enumerate(store.answer_logits):
            ids, mask = suite.batch_inputs(i)
            ref = hf.logits(ids, attention_mask=mask)
            real = _real_positions(mask, ref)
            _assert_close_padded(al[real], ref[real], what=f"logits on the real positions of batch {i}")

    # -- INTERVENTION ------------------------------------------------------------------------------

    def _intervene(
        self, suite, *, scope: str, mode: str = "add", scale: float = STEER_SCALE, vector=None, basis: bool = True
    ):
        """The caller's path: a raw payload in run_inputs, through model_fwd_intervention."""
        import interpretune as it
        from interpretune import AnalysisCfg

        if vector is None:
            vector = self._vector(suite)
        payload = {
            suite.inputs.intervention_point: {
                "intervention_tensor": vector,
                "mode": mode,
                "scale_factor": scale,
                "position_scope": scope,
                "use_intervention_tensor_as_basis": basis,
            }
        }
        return suite.run(
            AnalysisCfg(target_op=it.model_fwd_intervention, run_inputs={"interventions": payload}, save_tokens=True)
        )

    def _vector(self, suite) -> torch.Tensor:
        from interpretune import AnalysisCfg

        point = suite.inputs.intervention_point
        store = suite.run(AnalysisCfg(target_op="store_capture_points", names_filter=[point]))
        return steering_vector(captured_points(store, 0)[point])

    @conformance_case(
        capability=BackendCapability.INTERVENTION, scope=PositionScope.LAST_TOKEN, mode=InterventionMode.ADD
    )
    def test_last_token_scope_moves_exactly_the_final_position(self, suite):
        """The changed-position SET under `last_token` is exactly the final position."""
        store = self._intervene(suite, scope="last_token")
        for i, effect in enumerate(store["intervention_position_effect"]):
            _ids, mask = suite.batch_inputs(i)
            _require_discriminating_length(mask, effect.shape[-1], what=f"batch {i}")
            effect = _real_effect(effect, mask)
            seq = effect.shape[-1]
            got = changed_positions(torch.zeros_like(effect).unsqueeze(-1), effect.unsqueeze(-1))
            assert got == expected_positions("last_token", seq), f"batch {i}: changed {sorted(got)}"

    @conformance_case(
        capability=BackendCapability.INTERVENTION, scope=PositionScope.ALL_POSITIONS, mode=InterventionMode.ADD
    )
    def test_all_positions_scope_moves_every_real_position(self, suite):
        """Every real position is in the changed set under `all_positions`."""
        store = self._intervene(suite, scope="all_positions")
        for i, effect in enumerate(store["intervention_position_effect"]):
            _ids, mask = suite.batch_inputs(i)
            _require_discriminating_length(mask, effect.shape[-1], what=f"batch {i}")
            effect = _real_effect(effect, mask)
            got = changed_positions(torch.zeros_like(effect).unsqueeze(-1), effect.unsqueeze(-1))
            real = _real_positions(mask, effect.unsqueeze(-1))
            expected = {int(p) for p in torch.nonzero(real.any(dim=0)).flatten().tolist()}
            assert got == expected, f"batch {i}: changed {sorted(got)} but the real positions are {sorted(expected)}"

    @conformance_case(capability=BackendCapability.INTERVENTION, scope=PositionScope.ALL_POSITIONS, negative=True)
    def test_undeclared_all_positions_is_refused(self, suite):
        """A scope the backend did not declare is refused by name, never narrowed."""
        with expect_refusal(NotImplementedError, match="position_scope='all_positions'"):
            self._intervene(suite, scope="all_positions")

    @conformance_case(capability=BackendCapability.INTERVENTION, scope=PositionScope.LAST_TOKEN, negative=True)
    def test_undeclared_last_token_is_refused(self, suite):
        """A scope the backend did not declare is refused by name, never widened."""
        with expect_refusal(NotImplementedError, match="position_scope='last_token'"):
            self._intervene(suite, scope="last_token")

    @conformance_case(capability=BackendCapability.INTERVENTION)
    def test_undeclared_modes_are_refused_on_the_mode_axis(self, suite):
        """Each undeclared mode is refused naming the mode axis, never applied as another mode."""
        declared = {m.value for m in suite.capabilities.intervention.modes}
        undeclared = [m for m in InterventionMode if m.value not in declared]
        # Every mode declared means nothing to refuse: a pass, since the declaration itself is checked elsewhere.
        scope = next(iter(suite.capabilities.intervention.position_scopes)).value
        for mode in undeclared:
            with expect_refusal(NotImplementedError, match=f"mode='{mode.value}'"):
                self._intervene(suite, scope=scope, mode=mode.value)

    @conformance_case(capability=BackendCapability.INTERVENTION, mode=InterventionMode.ADD)
    def test_zero_intervention_is_identity(self, suite):
        """A zero-scale intervention leaves the logits unchanged."""
        scope = next(iter(suite.capabilities.intervention.position_scopes)).value
        store = self._intervene(suite, scope=scope, scale=0.0)
        for pre, post in zip(store["pre_intervention_logits"], store["post_intervention_logits"]):
            torch.testing.assert_close(
                post, pre, rtol=0, atol=CONVERGENCE_ATOL, msg="a zero-scale intervention changed the logits"
            )

    # -- per-mode invariants -----------------------------------------------------------------------
    #
    # Each declared mode has one algebraic property a caller can state without a reference implementation,
    # and each case pairs it with a positive control (the mode does SOMETHING), so an identity cannot pass
    # because the backend ignored the payload. The invariants are batch-safe by construction: an intervention
    # tensor broadcasts to every row, so "replace with the row's own activation" is not expressible here and
    # the replace invariant is scale-independence instead.

    def _second_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """A vector not collinear with ``vector`` and of the same shape: a rolled copy."""
        return torch.roll(vector, shifts=max(1, vector.numel() // 3), dims=-1)

    @staticmethod
    def _assert_moved(store, *, what: str) -> None:
        moved = any(
            not torch.allclose(post, pre, rtol=0, atol=CONVERGENCE_ATOL)
            for pre, post in zip(store["pre_intervention_logits"], store["post_intervention_logits"])
        )
        assert moved, f"{what} left every logit unchanged, so the invariant below would hold vacuously"

    @staticmethod
    def _assert_same_logits(a, b, *, what: str) -> None:
        """Two runs that are the same operation up to floating-point order agree to the padded tolerance.

        Mathematically equal payloads (a pair in either order, a basis and its negative multiple) reach the backend as
        different float32 arrays, and the pseudo-inverse or normalization then differs at roundoff. Six blocks later
        that is up to 1e-6 relative on logits of magnitude ~100, i.e. 1e-4 absolute, which is exactly the exact-identity
        tolerance; the relative tolerance is what these invariants need. The message keeps torch's own detail appended,
        so a failure reports the magnitude rather than only the claim.
        """
        for i, (x, y) in enumerate(zip(a["post_intervention_logits"], b["post_intervention_logits"])):
            torch.testing.assert_close(
                y, x, rtol=PADDED_RTOL, atol=PADDED_ATOL, msg=lambda detail, i=i: f"{what} (batch {i})\n{detail}"
            )

    @conformance_case(capability=BackendCapability.INTERVENTION, mode=InterventionMode.REPLACE)
    def test_replace_ignores_the_scale_factor(self, suite):
        """`replace` installs the tensor as-is: the result is independent of `scale_factor`, and it moves."""
        scope = next(iter(suite.capabilities.intervention.position_scopes)).value
        vector = self._vector(suite)
        unit = self._intervene(suite, scope=scope, mode="replace", scale=1.0, vector=vector)
        scaled = self._intervene(suite, scope=scope, mode="replace", scale=STEER_SCALE * 3, vector=vector)
        self._assert_moved(unit, what="replace")
        self._assert_same_logits(unit, scaled, what="replace changed with scale_factor, which it must ignore")

    @conformance_case(capability=BackendCapability.INTERVENTION, mode=InterventionMode.PATCH)
    def test_patch_of_a_pair_with_itself_is_identity(self, suite):
        """Swapping a concept's coordinates with its own leaves the activation, and the logits, unchanged."""
        scope = next(iter(suite.capabilities.intervention.position_scopes)).value
        vector = self._vector(suite)
        # scale_factor=1.0 is the pure exchange; the paper's alpha scales the swapped coordinates, so any other
        # value is a deliberate over- or under-steer and not an identity
        store = self._intervene(suite, scope=scope, mode="patch", scale=1.0, vector=torch.stack([vector, vector]))
        for i, (pre, post) in enumerate(zip(store["pre_intervention_logits"], store["post_intervention_logits"])):
            torch.testing.assert_close(
                post,
                pre,
                rtol=0,
                atol=CONVERGENCE_ATOL,
                msg=lambda detail, i=i: f"patch (v, v) changed the logits (batch {i})\n{detail}",
            )

    @conformance_case(capability=BackendCapability.INTERVENTION, mode=InterventionMode.PATCH)
    def test_patch_is_symmetric_in_pair_order(self, suite):
        """``h + V(sigma(c) - c)`` is the same update for ``(s, t)`` and ``(t, s)``, and it moves the logits.

        Both orders exchange the same two coordinates, so the pair is unordered; a backend that treated index 0
        as "from" and index 1 as "to" in some other sense would break this without breaking the identity case.
        """
        scope = next(iter(suite.capabilities.intervention.position_scopes)).value
        source = self._vector(suite)
        target = self._second_vector(source)
        forward = self._intervene(suite, scope=scope, mode="patch", scale=1.0, vector=torch.stack([source, target]))
        reverse = self._intervene(suite, scope=scope, mode="patch", scale=1.0, vector=torch.stack([target, source]))
        self._assert_moved(forward, what="patch (s, t)")
        self._assert_same_logits(forward, reverse, what="patch (s, t) and patch (t, s) differ")

    @conformance_case(capability=BackendCapability.INTERVENTION, mode=InterventionMode.PROJECT)
    def test_project_depends_only_on_the_basis_span(self, suite):
        """Projecting onto ``v`` and onto ``-2v`` is the same projection, and it moves the logits."""
        scope = next(iter(suite.capabilities.intervention.position_scopes)).value
        vector = self._vector(suite)
        onto_v = self._intervene(suite, scope=scope, mode="project", scale=1.0, vector=vector)
        onto_span = self._intervene(suite, scope=scope, mode="project", scale=1.0, vector=vector * -2.0)
        self._assert_moved(onto_v, what="project")
        self._assert_same_logits(onto_v, onto_span, what="project onto v and onto -2v differ")

    @conformance_case(capability=BackendCapability.INTERVENTION)
    def test_declared_modes_are_distinguishable(self, suite):
        """Every declared mode, given the same vector and point, yields logits distinguishable from every other
        declared mode and from the baseline.

        This is the case a backend that ignored ``mode`` and applied the one it has would fail, which is how the
        first hub adapter's backend once behaved: plausible logits for an intervention nobody requested.
        """
        scope = next(iter(suite.capabilities.intervention.position_scopes)).value
        vector = self._vector(suite)
        payloads: dict[str, tuple[torch.Tensor, float]] = {
            "add": (vector, STEER_SCALE),
            "replace": (vector, 1.0),
            "project": (vector, 1.0),
            "patch": (torch.stack([vector, self._second_vector(vector)]), 1.0),
            # `reject` removes the component in the span where `project` keeps it, so the same vector at
            # the same scale distinguishes them: they partition the activation rather than agreeing.
            "reject": (vector, 1.0),
        }
        declared = sorted(m.value for m in suite.capabilities.intervention.modes)
        results = {
            mode: self._intervene(suite, scope=scope, mode=mode, vector=payloads[mode][0], scale=payloads[mode][1])
            for mode in declared
        }
        for mode, store in results.items():
            self._assert_moved(store, what=f"mode {mode!r}")
        for i, a in enumerate(declared):
            for b in declared[i + 1 :]:
                same = all(
                    torch.allclose(x, y, rtol=0, atol=CONVERGENCE_ATOL)
                    for x, y in zip(results[a]["post_intervention_logits"], results[b]["post_intervention_logits"])
                )
                assert not same, f"modes {a!r} and {b!r} produced the same logits for the same vector and point"

    @conformance_case(capability=BackendCapability.INTERVENTION, mode=InterventionMode.ADD, family="hf_native")
    def test_baseline_is_an_unsteered_forward(self, suite, hf):
        """The pre-intervention half equals the plain forward."""
        scope = next(iter(suite.capabilities.intervention.position_scopes)).value
        store = self._intervene(suite, scope=scope)
        for i, pre in enumerate(store["pre_intervention_logits"]):
            ids, mask = suite.batch_inputs(i)
            # the op reports the FIRST ROW's last-token logits (`last_token_logits` takes `logits[0, -1]`)
            ref = hf.logits(ids, attention_mask=mask)[0, -1, :]
            _assert_close_padded(pre, ref, what=f"batch {i}: the baseline half against the plain forward")

    @conformance_case(
        capability=BackendCapability.INTERVENTION,
        scope=PositionScope.LAST_TOKEN,
        mode=InterventionMode.ADD,
        family="hf_native",
    )
    def test_steered_logits_converge_on_the_forward(self, suite, hf):
        """Adding a vector at the last token matches an HF hook doing the same."""
        vector = self._vector(suite)
        store = self._intervene(suite, scope="last_token", vector=vector)
        for i, post in enumerate(store["post_intervention_logits"]):
            ids, mask = suite.batch_inputs(i)
            ref = hf.steered(
                ids,
                suite.inputs.intervention_point,
                lambda t: t + vector * STEER_SCALE,
                scope="last_token",
                attention_mask=mask,
            )
            _assert_close_padded(
                post,
                ref["logits"][0, -1, :],
                what=f"batch {i}: steered last-token logits against the HF reference edit",
            )

    # -- INTERVENTION: mixed scopes -------------------------------------------------------------------

    @conformance_case(
        capability=BackendCapability.INTERVENTION,
        scopes=(PositionScope.LAST_TOKEN, PositionScope.ALL_POSITIONS),
        mode=InterventionMode.ADD,
        family="hf_native",
    )
    def test_mixed_scopes_produce_the_per_scope_result(self, suite, hf):
        """A payload naming two points under different scopes applies each point's own scope: it matches two HF
        hooks, one per point, and differs from either point's single-scope payload.

        The two edits are additive at different points, so their order does not matter and the reference applies
        them in any order. A non-commutative mode (``replace``, ``project``) at two points on one path would make
        the backend's application order part of the result, and this case does not claim anything about it.
        """
        import interpretune as it
        from interpretune import AnalysisCfg

        first = suite.inputs.intervention_point
        second = f"blocks.{suite.inputs.capture_layer + 1}.hook_in"
        vector = self._vector(suite)
        payload = {
            first: {
                "intervention_tensor": vector,
                "mode": "add",
                "scale_factor": STEER_SCALE,
                "position_scope": "last_token",
                "use_intervention_tensor_as_basis": True,
            },
            second: {
                "intervention_tensor": vector,
                "mode": "add",
                "scale_factor": STEER_SCALE,
                "position_scope": "all_positions",
                "use_intervention_tensor_as_basis": True,
            },
        }
        mixed = suite.run(
            AnalysisCfg(target_op=it.model_fwd_intervention, run_inputs={"interventions": payload}, save_tokens=True)
        )
        for i, post in enumerate(mixed["post_intervention_logits"]):
            ids, mask = suite.batch_inputs(i)
            ref = hf.steered_many(
                ids,
                [
                    (first, lambda t: t + vector * STEER_SCALE, "last_token"),
                    (second, lambda t: t + vector * STEER_SCALE, "all_positions"),
                ],
                attention_mask=mask,
            )
            _assert_close_padded(post, ref["logits"][0, -1, :], what=f"batch {i}: mixed-scope last-token logits")
        # positive control: the mixed payload is not either single-scope payload at the first point in disguise
        for scope in ("last_token", "all_positions"):
            alone = self._intervene(suite, scope=scope, vector=vector)
            same = all(
                torch.allclose(a, b, rtol=0, atol=CONVERGENCE_ATOL)
                for a, b in zip(alone["post_intervention_logits"], mixed["post_intervention_logits"])
            )
            assert not same, f"the mixed-scope payload gave the logits of a {scope!r}-only payload at {first!r}"

    # -- LATENT_MODELS ------------------------------------------------------------------------------
    #
    # These run over `inputs.latent_models`, which the target's session config attaches. The latent model is an
    # sae_lens handle today (`release` + `sae_id`, the activations cached under `<hook>.hook_sae_acts_post`); a
    # backend declaring LATENT_MODELS with no handle attached fails here by name rather than skipping, because a
    # skip would read as "undeclared" in the report and the declaration is exactly what was made.

    def _latent_hook(self, suite) -> tuple[Any, str]:
        """The first attached latent model and the point its activations are cached under."""
        handles = list(getattr(suite.module, "sae_handles", None) or [])
        if not handles and not suite.inputs.latent_models_for_model():
            # the suite's gap, not the backend's: nothing to attach for this model, so the case cannot decide
            pytest.skip(f"the suite carries no latent model for {suite.inputs.model_id!r}; the latent cases need one")
        assert handles, (
            f"{suite.backend_name} declares LATENT_MODELS but the session attached no latent model; a target "
            f"declaring it attaches inputs.latent_models ({[s.sae_id for s in suite.inputs.latent_models]}) in "
            "its session config"
        )
        handle = handles[0]
        return handle, f"{handle.cfg.metadata.hook_name}.hook_sae_acts_post"

    def _latent_store(self, suite, op_name: str, hook: str):
        """One memoized run of a latent composite over the attached model's activation point."""
        import interpretune as it
        from interpretune import AnalysisCfg

        op = getattr(it, op_name)
        return suite.run_once(f"{op_name}:{hook}", AnalysisCfg(target_op=op, names_filter=[hook], save_tokens=True))

    @staticmethod
    def _per_hook(store, column: str, index: int, hook: str):
        """One batch's value of a per-latent-hook column, refusing an absent entry by name."""
        row = store[column][index]
        assert isinstance(row, dict) and hook in row, (
            f"{column}[{index}] carries no entry for {hook!r}: {sorted(row) if isinstance(row, dict) else row!r}"
        )
        value = row[hook]
        assert value is not None, (
            f"{column}[{index}][{hook!r}] is None: the op wrote nothing for the latent model the target attached, "
            "which is what an activation point the backend never matched looks like"
        )
        return value

    def _latent_batch(self, suite, hook: str) -> tuple[int, torch.Tensor, list[int], int]:
        """The batch with the most correct rows, its answer positions, its two strongest alive latents at the
        answer position, and one dead latent; from the memoized latent run, so no case re-derives them."""
        store = self._latent_store(suite, "logit_diffs_latent", hook)
        best, best_rows = 0, -1
        for i in range(len(store.logit_diffs)):
            rows = torch.as_tensor(self._per_hook(store, "correct_activations", i, hook)).shape[0]
            if rows > best_rows:
                best, best_rows = i, rows
        assert best_rows > 0, "no batch has a correct row, so no answer-position activation is available"
        acts = torch.as_tensor(self._per_hook(store, "correct_activations", best, hook))
        alive = {int(a) for a in self._per_hook(store, "alive_latents", best, hook)}
        strongest = [int(j) for j in acts.abs().amax(dim=0).topk(2).indices.tolist()]
        dead = next(j for j in range(acts.shape[-1]) if j not in alive)
        answer_positions = torch.as_tensor(store.answer_indices[best]).reshape(-1)
        return best, answer_positions, strongest, dead

    def _hook_batch(self, suite, index: int) -> dict[str, torch.Tensor]:
        """The runner's batch as the backend's forward takes it: model inputs only."""
        return {k: v for k, v in suite.batches[index].items() if k != "labels"}

    @staticmethod
    def _ablate(latent: int, positions: torch.Tensor):
        """The bundled ablation hook, bound to one latent at the answer positions: the hook signature every
        LATENT_MODELS backend must honour, taken from the op that uses it rather than restated here."""
        from functools import partial

        from interpretune.analysis.ops.bundled.sae.sae_ops import ablate_sae_latent

        return partial(ablate_sae_latent, latent_idx=latent, seq_pos=positions)

    def _forward_with_hooks(self, suite, index: int, fwd_hooks: list) -> torch.Tensor:
        """The backend's hooked forward with the attached latent models, no gradient."""
        with torch.no_grad():
            return suite.backend.fwd_w_hooks_and_latent_models(
                model=suite.module.model,
                batch=self._hook_batch(suite, index),
                latent_model_handles=list(suite.module.sae_handles),
                fwd_hooks=fwd_hooks,
            )

    @conformance_case(capability=BackendCapability.LATENT_MODELS)
    def test_latent_op_stores_the_declared_schema(self, suite):
        """`logit_diffs_latent` yields, per batch and per attached model, a non-empty alive-latent set and the
        answer-position activations of the correct rows, `[rows, d_sae]`."""
        handle, hook = self._latent_hook(suite)
        store = self._latent_store(suite, "logit_diffs_latent", hook)
        n = suite.inputs.limit_batches
        assert len(store.logit_diffs) == n, f"{len(store.logit_diffs)} rows for {n} batches"
        for i in range(n):
            alive = self._per_hook(store, "alive_latents", i, hook)
            assert len(alive) > 0, (
                f"batch {i}: no alive latent at the answer position; the latent cases would be vacuous"
            )
            acts = torch.as_tensor(self._per_hook(store, "correct_activations", i, hook))
            assert acts.ndim == 2 and acts.shape[-1] == handle.cfg.d_sae, f"batch {i}: {tuple(acts.shape)}"
            assert torch.isfinite(acts).all(), f"batch {i}: non-finite activations"

    @conformance_case(capability=BackendCapability.LATENT_MODELS)
    def test_alive_latents_are_the_positive_latents_at_the_answer(self, suite):
        """Internal consistency, with no reference: every latent positive in a correct row's answer-position
        activation is in the batch's alive set, and some batch has such a latent (else the claim is vacuous)."""
        _handle, hook = self._latent_hook(suite)
        store = self._latent_store(suite, "logit_diffs_latent", hook)
        checked = 0
        for i in range(len(store.logit_diffs)):
            alive = {int(a) for a in self._per_hook(store, "alive_latents", i, hook)}
            acts = torch.as_tensor(self._per_hook(store, "correct_activations", i, hook))
            positive = {int(j) for j in torch.nonzero(acts > 0)[:, -1].tolist()}
            assert positive <= alive, f"batch {i}: {len(positive - alive)} latents positive at the answer but not alive"
            checked += len(positive)
        assert checked > 0, "no correct row had a positive latent, so the subset claim held vacuously"

    @conformance_case(capability=BackendCapability.LATENT_MODELS)
    def test_batched_hooks_agree_with_sequential(self, suite):
        """`fwd_w_hooks_batched` returns, per config, what one `fwd_w_hooks_and_latent_models` call returns.

        Reaches the backend directly, by design: the claim is on the method pair. `batched_hooks` says only
        whether the backend fuses the configs into one execution, which a caller cannot observe, so the equality
        is the contract under either declaration. Positive control: the two configs differ from each other.
        """
        _handle, hook = self._latent_hook(suite)
        index, positions, (first, second), _dead = self._latent_batch(suite, hook)
        configs = [[(hook, self._ablate(first, positions))], [(hook, self._ablate(second, positions))]]
        with torch.no_grad():
            batched = suite.backend.fwd_w_hooks_batched(
                model=suite.module.model,
                batch=self._hook_batch(suite, index),
                latent_model_handles=list(suite.module.sae_handles),
                hook_configs=configs,
            )
        assert len(batched) == len(configs), f"{len(batched)} results for {len(configs)} hook configs"
        # The padded (relative) tolerance rather than the exact one: a backend that fuses the configs runs a larger
        # batch through different kernels, and the same arithmetic in another order reaches the logits as roundoff.
        # Measured on the macOS runner with the nnsight backend: 3 of 26,334,668 elements at 1.07e-4 absolute and
        # 1.5e-6 relative, on logits of magnitude ~100; Linux agreed exactly. The positive control below is what
        # keeps this from being satisfiable by two runs that ignored their hooks.
        for k, config in enumerate(configs):
            sequential = self._forward_with_hooks(suite, index, config)
            torch.testing.assert_close(
                batched[k],
                sequential,
                rtol=PADDED_RTOL,
                atol=PADDED_ATOL,
                msg=lambda detail, k=k: f"config {k}: batched and sequential logits differ\n{detail}",
            )
        assert not torch.allclose(batched[0], batched[1], rtol=0, atol=CONVERGENCE_ATOL), (
            f"ablating latent {first} and latent {second} gave the same logits, so the agreement above is vacuous"
        )

    @conformance_case(capability=BackendCapability.LATENT_MODELS)
    def test_ablating_a_dead_latent_is_identity(self, suite):
        """Zeroing a latent that is already zero at the answer position leaves the logits unchanged, and zeroing
        the strongest alive one moves them (positive control).

        Reaches the backend directly, by design.
        """
        _handle, hook = self._latent_hook(suite)
        index, positions, (strongest, _), dead = self._latent_batch(suite, hook)
        base = self._forward_with_hooks(suite, index, [])
        dead_run = self._forward_with_hooks(suite, index, [(hook, self._ablate(dead, positions))])
        torch.testing.assert_close(
            dead_run, base, rtol=0, atol=CONVERGENCE_ATOL, msg=f"ablating dead latent {dead} changed the logits"
        )
        alive_run = self._forward_with_hooks(suite, index, [(hook, self._ablate(strongest, positions))])
        assert not torch.allclose(alive_run, base, rtol=0, atol=CONVERGENCE_ATOL), (
            f"ablating the strongest alive latent {strongest} changed nothing; the edit never reached the forward"
        )

    # -- GRADIENTS ----------------------------------------------------------------------------------

    @conformance_case(capability=BackendCapability.GRADIENTS)
    def test_gradient_op_stores_the_declared_schema(self, suite):
        """`logit_diffs_attr_grad` yields per-hook attribution values `[rows, d_sae]`, finite, zero off the alive
        set, and non-zero somewhere (positive control)."""
        handle, hook = self._latent_hook(suite)
        store = self._latent_store(suite, "logit_diffs_attr_grad", hook)
        nonzero = 0
        for i in range(len(store.logit_diffs)):
            attr = torch.as_tensor(self._per_hook(store, "attribution_values", i, hook))
            rows = suite.batch_inputs(i)[0].shape[0]
            assert attr.shape == (rows, handle.cfg.d_sae), f"batch {i}: {tuple(attr.shape)}"
            assert torch.isfinite(attr).all(), f"batch {i}: non-finite attribution"
            alive = {int(a) for a in self._per_hook(store, "alive_latents", i, hook)}
            off = {int(j) for j in torch.nonzero(attr != 0)[:, 1].tolist()}
            assert off <= alive, f"batch {i}: attribution on {len(off - alive)} latents that are not alive"
            nonzero += len(off)
        assert nonzero > 0, "every attribution is zero; the gradient never reached the latent activations"

    def _logit_diff_sum(self, suite, store, index: int, fwd_hooks: list) -> float:
        """The scalar the gradient op backpropagates (the summed logit difference), recomputed from a hooked
        forward with the labels and answer positions the op itself stored."""
        from interpretune.analysis.ops.base import AnalysisBatch
        from interpretune.analysis.optools import boolean_logits_to_avg_logit_diff, get_loss_preds_diffs

        logits = self._forward_with_hooks(suite, index, fwd_hooks)
        positions = torch.as_tensor(store.answer_indices[index]).reshape(-1)
        answer_logits = logits[torch.arange(logits.shape[0]), positions]
        batch = AnalysisBatch(label_ids=store.label_ids[index], orig_labels=store.orig_labels[index])
        # typed for the module's analysis-batch protocol; a bare AnalysisBatch carries the two fields it reads
        _loss, logit_diffs, _preds, _ = get_loss_preds_diffs(
            suite.module, cast(Any, batch), answer_logits, boolean_logits_to_avg_logit_diff
        )
        return float(logit_diffs.sum())

    @conformance_case(capability=BackendCapability.GRADIENTS)
    def test_gradient_predicts_a_small_perturbation_to_first_order(self, suite):
        """Scaling the strongest latent by `1 + eps` at the answer position moves the summed logit difference by
        `eps` times its stored attribution, to first order.

        The attribution the op stores is activation times gradient, so it IS the first-order coefficient of a relative
        perturbation. Reaches the backend directly for the perturbed forward, by design: the op path has no slot for a
        caller's edit. The unperturbed sum is recomputed the same way and checked against the op's own logit differences
        first, so the two paths are known to see one forward before the prediction is scored.
        """
        _handle, hook = self._latent_hook(suite)
        store = self._latent_store(suite, "logit_diffs_attr_grad", hook)
        totals = {
            i: torch.as_tensor(self._per_hook(store, "attribution_values", i, hook)).sum(dim=0)
            for i in range(len(store.logit_diffs))
        }
        index = max(totals, key=lambda i: float(totals[i].abs().max()))
        latent = int(totals[index].abs().argmax())
        coefficient = float(totals[index][latent])
        assert abs(coefficient) > CHANGED_ATOL, "no latent carries a first-order effect above the noise floor"
        positions = torch.as_tensor(store.answer_indices[index]).reshape(-1)

        def scale(acts: torch.Tensor, hook: Any, *, _eps: float) -> torch.Tensor:
            acts[torch.arange(acts.shape[0]), positions, latent] *= 1.0 + _eps
            return acts

        eps = 0.02
        unperturbed = self._logit_diff_sum(suite, store, index, [])
        torch.testing.assert_close(
            torch.tensor(unperturbed),
            torch.as_tensor(store.logit_diffs[index]).sum().to(torch.float32),
            rtol=PADDED_RTOL,
            atol=PADDED_ATOL,
            msg="the recomputed logit-difference sum differs from the op's own; the two paths see different forwards",
        )
        perturbed = self._logit_diff_sum(suite, store, index, [(hook, lambda acts, hook: scale(acts, hook, _eps=eps))])
        measured = perturbed - unperturbed
        predicted = eps * coefficient
        assert abs(measured - predicted) <= 0.1 * abs(predicted) + CONVERGENCE_ATOL, (
            f"first-order prediction {predicted:.4e} vs measured {measured:.4e} for latent {latent} (eps={eps})"
        )

    # -- analysis backend: ATTRIBUTION_GRAPH / FEATURE_INTERVENTION -------------------------------------
    #
    # The graph ops take a prompt rather than a datamodule batch, so these cases call the ops the way every caller
    # does (an analysis batch carrying the prompt, no dataloader batch), which is the ops' own entry rather than a
    # detour around it. Everything compared is read off op outputs: no case reaches the analysis backend's model.

    def _attribution_prompt(self, suite) -> str:
        prompt = suite.inputs.attribution_prompt
        if prompt is None:
            pytest.skip(f"the suite carries no attribution prompt for {suite.inputs.model_id!r}")
        return prompt

    def _graph(self, suite):
        """The attribution graph op's output for the suite prompt, computed once per target class."""
        import interpretune as it
        from interpretune.analysis.ops.base import AnalysisBatch

        prompt = self._attribution_prompt(suite)
        key = f"attribution_graph:{prompt}"
        if key not in suite.memo:
            try:
                suite.memo[key] = it.compute_attribution_graph(
                    suite.module, AnalysisBatch(prompts=[prompt]), batch=cast(Any, None), batch_idx=0
                )
            except Exception as exc:
                # A source-node lookup that fails names what was expected and not what was met; the facts that
                # decide the cause are which attention function each model handle is bound to (the eager one is
                # the only one whose source calls dropout), the owning module's mode, and the node set the
                # accessor actually holds. Report them beside the error rather than leaving them to a bisect.
                raise AssertionError(_attribution_failure_context(suite, exc)) from exc
        return suite.memo[key]

    @conformance_case(capability=AnalysisBackendCapability.ATTRIBUTION_GRAPH)
    def test_attribution_graph_round_trips_through_the_store(self, suite):
        """The graph the op writes as store columns hydrates back to the graph it computed: same adjacency, same
        active features, and the selected features index into them."""
        from interpretune.analysis.backends import require_analysis_backend

        result = self._graph(suite)
        backend = require_analysis_backend(suite.module)
        graph = backend.hydrate_graph_from_batch(result)
        adjacency = torch.as_tensor(result.adjacency_matrix, dtype=torch.float32)
        active = torch.as_tensor(result.active_features)
        assert adjacency.ndim == 2 and adjacency.shape[0] == adjacency.shape[1], tuple(adjacency.shape)
        assert active.ndim == 2 and active.shape[1] == 3 and active.shape[0] > 0, tuple(active.shape)
        assert_non_degenerate(adjacency, what="adjacency matrix")
        torch.testing.assert_close(torch.as_tensor(graph.adjacency_matrix, dtype=torch.float32).cpu(), adjacency)
        assert torch.equal(torch.as_tensor(graph.active_features).cpu(), active)
        selected = torch.as_tensor(graph.selected_features).cpu()
        assert selected.numel() > 0 and int(selected.max()) < int(active.shape[0])

    @conformance_case(capability=AnalysisBackendCapability.ATTRIBUTION_GRAPH)
    def test_pruning_is_monotone(self, suite):
        """A stricter node threshold keeps a subset of what a looser one keeps, and keeps strictly fewer at some
        pair (positive control: a pruning that ignored its threshold would pass the subset claim vacuously)."""
        import interpretune as it
        from interpretune.analysis.ops.base import AnalysisBatch

        result = self._graph(suite)
        kept: dict[float, set[int]] = {}
        for threshold in (0.3, 0.6, 0.9):
            pruned = it.graph_prune(
                suite.module,
                AnalysisBatch(**dict(result)),
                batch=cast(Any, None),
                batch_idx=0,
                node_threshold=threshold,
            )
            kept[threshold] = {int(i) for i in torch.as_tensor(pruned.selected_features).flatten().tolist()}
        assert kept[0.3] <= kept[0.6] <= kept[0.9], {t: len(v) for t, v in kept.items()}
        assert len(kept[0.3]) < len(kept[0.9]), f"pruning kept {len(kept[0.9])} nodes at every threshold"

    @conformance_case(capability=AnalysisBackendCapability.FEATURE_INTERVENTION)
    def test_the_edge_predicts_the_measured_feature_intervention(self, suite):
        """Scaling a top feature's activation moves the other active features and the target logits by the graph's
        edge weights into that feature, times the scale: the graph is a linear model of the intervention it
        describes, checked on the feature intervention op's own outputs."""
        import interpretune as it
        from interpretune.analysis.backends import require_analysis_backend
        from interpretune.analysis.ops.base import AnalysisBatch

        result = self._graph(suite)
        backend = require_analysis_backend(suite.module)
        graph = backend.hydrate_graph_from_batch(result)
        # The graph is a linear model of an intervention only when every layer is constrained (downstream features
        # frozen at their baseline rather than re-activated nonlinearly), so the case states that requirement on
        # the module's settings rather than assuming the target set it: measured unconstrained on gemma-3-1b-it, the
        # active features moved off the edges by up to 126 where the edges predicted under 1e-3.
        support = suite.capabilities.capture
        n_layers = support.n_layers if support is not None else None
        ct_cfg = getattr(suite.module, "circuit_tracer_cfg", None)
        if ct_cfg is not None and n_layers is not None:
            ct_cfg.intervention_constrained_layers = list(range(n_layers))
            ct_cfg.intervention_apply_activation_function = False
        influence = it.graph_node_influence(suite.module, result, batch=cast(Any, None), batch_idx=0)
        payload = dict(result)
        payload.update(dict(influence))
        top = it.extract_top_features(
            suite.module,
            AnalysisBatch(**payload),
            batch=cast(Any, None),
            batch_idx=0,
            top_n=suite.inputs.attribution_top_n,
        )
        feature_rows = torch.as_tensor(top.top_feature_ids, dtype=torch.long)
        assert feature_rows.shape[0] > 0, "no top feature to intervene on"
        active = torch.as_tensor(graph.active_features).cpu()
        adjacency = torch.as_tensor(graph.adjacency_matrix, dtype=torch.float32).cpu()
        logit_tokens = torch.as_tensor(graph.logit_tokens).long().cpu()
        baseline_acts = torch.as_tensor(result.activation_values, dtype=torch.float32).cpu()
        assert baseline_acts.shape[0] == active.shape[0], "one baseline activation per active feature"
        checked = 0
        for index, row in enumerate(feature_rows):
            layer, position, feature = (int(v) for v in row.tolist())
            node = int(((active[:, 0] == layer) & (active[:, 1] == position) & (active[:, 2] == feature)).nonzero()[0])
            baseline = float(baseline_acts[node])
            if abs(baseline) < 1e-12:
                continue
            single = AnalysisBatch(
                prompts=[self._attribution_prompt(suite)],
                top_feature_ids=row.unsqueeze(0),
                top_feature_scores=torch.as_tensor(top.top_feature_scores, dtype=torch.float32)[index : index + 1],
                top_feature_activation_values=torch.as_tensor(top.top_feature_activation_values, dtype=torch.float32)[
                    index : index + 1
                ],
                logit_target_ids=logit_tokens,
            )
            out = it.feature_intervention_forward(
                suite.module, single, batch=cast(Any, None), batch_idx=0, intervention_return_activations=True
            )
            new_value = float(torch.as_tensor(out.intervention_values, dtype=torch.float32)[0])
            scale = (new_value - baseline) / baseline
            expected = adjacency[:, node] * scale
            cache = torch.as_tensor(out.intervention_activation_cache, dtype=torch.float32).cpu()
            measured_acts = cache[active[:, 0], active[:, 1], active[:, 2]]
            torch.testing.assert_close(
                measured_acts,
                baseline_acts + expected[: active.shape[0]],
                rtol=1e-5,
                atol=1e-3,
                msg=lambda d, f=(layer, position, feature): f"feature {f}: active features moved off the edges\n{d}",
            )
            pre = torch.as_tensor(out.pre_intervention_logits, dtype=torch.float32).cpu().reshape(-1)
            post = torch.as_tensor(out.post_intervention_logits, dtype=torch.float32).cpu().reshape(-1)
            pre_d = pre[logit_tokens] - pre.mean()
            post_d = post[logit_tokens] - post.mean()
            torch.testing.assert_close(
                post_d,
                pre_d + expected[-len(logit_tokens) :],
                rtol=1e-3,
                atol=1e-5,
                msg=lambda d, f=(layer, position, feature): f"feature {f}: target logits moved off the edges\n{d}",
            )
            checked += 1
        assert checked > 0, "every top feature had a zero baseline activation; nothing was checked"

    # -- single-prompt backends ----------------------------------------------------------------------

    @conformance_case(single_prompt=True)
    def test_a_batch_above_the_declared_limit_is_refused_by_name(self, suite):
        """A target that declared one prompt at a time must REFUSE a larger batch, naming the limit.

        Reaches the backend directly, by design: the runner never builds a batch above the declared size for this
        target, so the refusal is only observable by asking. Silently processing row 0, or looping and re-padding
        without saying so, are the substitutions this case exists to catch.
        """
        ids, mask = suite.batch_inputs(0)
        assert ids.shape[0] == 1, f"a single-prompt target must run batches of one row, got {ids.shape[0]}"
        batch: dict[str, Any] = {"input": ids.repeat(2, 1)}
        if mask is not None:
            batch["attention_mask"] = mask.repeat(2, 1)
        with pytest.raises(Exception, match=r"(?i)one prompt|batch|single"):
            suite.backend.fwd(model=suite.module.model, batch=batch)

    # -- calibration: the discriminator, on the reference alone ------------------------------------

    @conformance_case(family="hf_native")
    def test_the_scope_discriminator_tells_the_scopes_apart(self, suite, hf, prompt_ids):
        """Positive control.

        Without it a scope case could pass because the instrument always says yes.
        """
        ids = prompt_ids[0]
        _require_discriminating_length(None, ids.shape[1], what="the calibration prompt")
        base = hf.capture(ids, [suite.inputs.observe_point])[suite.inputs.observe_point]
        vec = steering_vector(hf.capture(ids, [suite.inputs.intervention_point])[suite.inputs.intervention_point])
        for scope in ("last_token", "all_positions"):
            out = hf.steered(
                ids,
                suite.inputs.intervention_point,
                lambda t: t + vec * STEER_SCALE,
                scope=scope,
                observe=[suite.inputs.observe_point],
            )
            assert changed_positions(base, out[suite.inputs.observe_point]) == expected_positions(scope, ids.shape[1])
