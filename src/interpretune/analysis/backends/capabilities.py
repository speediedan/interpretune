"""The capability namespace, its support records, module capability aggregation, and the named-backend registry.

Part of the sanctioned :mod:`interpretune.analysis.backends` seam that op implementations (bundled,
local, or hub) may import. Ops should ask for capabilities rather than branching on backend or
adapter class names.
"""

from __future__ import annotations

from collections.abc import Mapping

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from interpretune.analysis.backends.protocols import AnalysisBackend, ModelBackend


class Capability(Enum):
    """The capabilities a composition's components provide: one namespace, whichever component provides them.

    A composition has components (a model backend that executes the base model, and may have an analysis backend such
    as circuit-tracer's replacement model), and each declares which of these it provides. The gates ask "who provides
    capability X", never "does the model backend or the analysis backend declare it", because which capabilities a
    component offers does not depend on its kind.

    A member answers "is this surface provided"; it never answers "which configurations of the surface". Those are a
    different kind of fact with a different shape: the typed support record the providing component attaches
    (:class:`InterventionSupport`, :class:`LatentModelSupport`, :class:`CaptureSupport`,
    :class:`AttributionGraphSupport`). Keeping the two apart is what lets a component truthfully claim a surface while
    declaring exactly which configurations of it it honours, and lets a gate refuse the rest by name. What a component
    needs ANOTHER component to provide is its ``requires`` set, checked when the module's capabilities are aggregated
    (:func:`get_module_capabilities`).
    """

    INTERVENTION = "intervention"
    """An edit applied during a forward pass, on an activation, a unit set or a latent model's features.

    The record
    (:class:`InterventionSupport`) declares which ops, targets, position scopes, value sources, bases and flows the
    provider honours; two components providing one target is legitimate and is resolved by naming the provider.
    """

    LATENT_MODELS = "latent_models"
    """``SupportsLatentModels``: ``fwd_w_cache_and_latent_models``, ``fwd_w_hooks_and_latent_models``,
    ``fwd_w_hooks_batched``."""

    GRADIENTS = "gradients"
    """``SupportsGradients``: forward + backward with gradient caching."""

    CAPTURE = "capture"
    """Caching activations at vocabulary points: the read surface (a latent site's ``observe`` is a capture). Its
    record, :class:`CaptureSupport`, depends on the model at hand, so the provider attaches it as
    ``capture_support(model)``."""

    ATTRIBUTION_GRAPH = "attribution_graph"
    """``SupportsAttributionGraph``: attribution graph construction, storage and scoring."""

    HEAD_ATTRIBUTION = "head_attribution"
    """``SupportsHeadAttribution``: credit at the level of attention heads."""


#: The component slots of a composition, in the order capabilities are aggregated. A provider is named by its slot.
COMPONENTS: tuple[str, ...] = ("model_backend", "analysis_backend")

#: Intervention targets, as venngram names them (``venngram.ops.TARGETS``): a whole activation, a set of its units, or
#: a latent model's features.
INTERVENTION_TARGETS: tuple[str, ...] = ("activation", "units", "features")

#: Value sources, as venngram names them (``venngram.ops.VALUE_SOURCES``).
VALUE_SOURCES: tuple[str, ...] = ("dose", "doses", "from")

#: What an op may act on: the forward value, or the gradient flowing back through the site.
FLOWS: tuple[str, ...] = ("value", "grad")


class PositionScope(str, Enum):
    """Which positions an intervention edits.

    A ``str`` enum so a spec built from YAML or a notebook can carry the plain string and still
    compare equal to the member.

    **Both scopes are legitimate operations, not a correct one and a broken one.** Steering the final
    token is the right shape for "change the next prediction"; steering every position is the right
    shape for "make the model read the whole prompt differently". Interpretune previously had a name
    for only the first, which is what made a backend implementing the second look like a defect
    rather than like a capability we could not express.
    """

    LAST_TOKEN = "last_token"
    ALL_POSITIONS = "all_positions"
    POSITIONS = "positions"
    """Explicit positions the payload names itself (circuit-tracer's feature tuples carry their own position).

    A hook spec carries no positions, so a backend applying hook specs never declares this scope.
    """

    @classmethod
    def hook_scopes(cls) -> frozenset[PositionScope]:
        """The scopes a hook spec can name: the final real token, or every position."""
        return frozenset({cls.LAST_TOKEN, cls.ALL_POSITIONS})


class InterventionMode(str, Enum):
    """How an intervention combines its tensor with the activation it targets.

    A ``str`` enum for the same reason as :class:`PositionScope`. The mode is the second axis of the
    intervention contract (scope is the first): a backend can implement ``fwd_w_intervention`` and still
    be unable to express most modes. Every mode but ``add`` reads the CURRENT activation, and the
    additional thing they need is that the READ HAPPENS DURING THE FORWARD PASS: ``patch`` and ``reject``
    compute coordinates from the activation itself, so no parameter fixed when the spec was built can
    stand in for them.

    The requirement is therefore sharper than "can observe the activation": **a steering surface whose
    parameters are all static scalars satisfies that and still expresses none of these modes.** What a
    backend must support is a parameter COMPUTED AT FORWARD TIME. Stated as a capability rather than by
    naming a backend, because core must not know which adapters exist, and because a rule about what an
    implementation needs stays true when the next one appears.

    Such a backend declares ``InterventionSupport(ops={ADD}, ...)``; the declaration, not the
    implementation, is what the dispatcher consults, so the refusal is by name rather than by trial.

    A mode a backend has not declared is refused by
    :func:`~interpretune.analysis.backends.interventions.require_intervention_mode` rather than applied
    as a different mode, since every mode returns plausible logits and the substitution is undetectable
    from the result.

    A payload naming several specs at one hook point applies them SEQUENTIALLY IN DECLARATION ORDER,
    each seeing the previous spec's output: ``[add(v), reject(v)]`` is not ``[reject(v), add(v)]``,
    while two ``add`` specs commute. Backends implement this by folding the spec tuple left to right,
    and adapters must preserve the declaration order rather than grouping by mode.
    A backend that cannot apply that order refuses the payload by name rather than reordering it.
    """

    REPLACE = "replace"
    ADD = "add"
    PATCH = "patch"
    PROJECT = "project"
    REJECT = "reject"
    CLAMP = "clamp"
    SCALE = "scale"


@dataclass(frozen=True)
class FeatureTargetSupport:
    """The configurations of the ``features`` target a feature-tuple provider honours (circuit-tracer's replacement
    model).

    The detail of the ``features`` target in that provider's :class:`InterventionSupport`, because what it adds over
    the other targets (where a value comes from, whether layers can be constrained, whether the intervened activations
    are returned, whether norm denominators can be frozen) is configuration of the same surface rather than a surface
    of its own. The settings ``resolve_feature_intervention_settings`` reads are the configuration space; the record is
    what lets a caller's settings be refused by name at the seam instead of at the first tensor op that disagrees.
    """

    value_sources: frozenset[str]
    """Accepted ``value_source`` spellings (``"constant"`` requires an explicit value)."""
    constrainable_layers: bool = True
    """Whether ``constrained_layers`` can restrict the intervention to a layer subset."""
    returns_activations: bool = False
    """Whether the intervened activations can be returned beside the logits."""
    freezable_norms: bool = False
    """Whether ``freeze_norms`` can hold every residual-stream norm's denominator at its clean value."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "value_sources", frozenset(self.value_sources))
        if not self.value_sources:
            raise ValueError("a features-target support record must accept at least one value source")

    def refusal(self, settings: Mapping[str, Any]) -> str | None:
        """Why ``settings`` cannot be honoured here, or ``None`` when every configuration they name is declared."""
        source = settings.get("value_source")
        if source not in self.value_sources:
            return f"value_source {source!r} is not honoured here; declared: {sorted(self.value_sources)}"
        if source == "constant" and settings.get("value") is None:
            return "value_source 'constant' needs an explicit intervention value, and none was given"
        if settings.get("constrained_layers") is not None and not self.constrainable_layers:
            return (
                "constrained_layers was given, and this backend cannot restrict a feature intervention to a"
                " layer subset"
            )
        if settings.get("return_activations") and not self.returns_activations:
            return "return_activations was requested, and this backend cannot return the intervened activations"
        if settings.get("freeze_norms") and not self.freezable_norms:
            return "freeze_norms was requested, and this backend cannot hold norm denominators at their clean values"
        return None

    def describe(self) -> str:
        """One line for a card or a report."""
        return (
            f"value sources {sorted(self.value_sources)}, constrainable layers {self.constrainable_layers}, "
            f"returns activations {self.returns_activations}, freezable norms {self.freezable_norms}"
        )


#: The detail record each target may carry; a target without configuration of its own carries ``None``.
_TARGET_DETAIL_TYPES: dict[str, type] = {"features": FeatureTargetSupport}


@dataclass(frozen=True)
class InterventionSupport:
    """Which configurations of ``INTERVENTION`` a provider can honour, on the axes venngram's ``Support`` names.

    ``ops`` are the ops it executes, ``targets`` what they may act on (``activation``, ``units``, ``features``, each
    with its detail record or ``None``), ``position_scopes`` which positions a payload may edit, ``values`` the value
    sources a recipe may draw from, ``bases`` the operand bases it honours (``None``: any) and ``of`` whether ops act on
    the forward value, the gradient, or both. :func:`~interpretune.analysis.venngram.venngram_support` derives
    venngram's record from this one directly.

    Declaring the capability without one of these is a contract violation, not a legacy default: the absence of a
    declaration used to mean "assume last-token", which is exactly the silent narrowing the scope field was introduced
    to remove.
    """

    ops: frozenset[InterventionMode]
    position_scopes: frozenset[PositionScope]
    targets: Mapping[str, Any] = field(default_factory=lambda: dict.fromkeys(INTERVENTION_TARGETS))
    """Every target with no detail by default, as venngram's ``Support`` defaults to every target."""
    values: frozenset[str] = frozenset(VALUE_SOURCES)
    bases: frozenset[str] | None = None
    of: frozenset[str] = frozenset({"value"})

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_scopes", frozenset(PositionScope(s) for s in self.position_scopes))
        object.__setattr__(self, "ops", frozenset(InterventionMode(m) for m in self.ops))
        object.__setattr__(self, "targets", dict(self.targets))
        object.__setattr__(self, "values", frozenset(self.values))
        object.__setattr__(self, "of", frozenset(self.of))
        if self.bases is not None:
            object.__setattr__(self, "bases", frozenset(self.bases))
        if not self.position_scopes:
            raise ValueError("InterventionSupport must declare at least one position scope")
        if not self.ops:
            raise ValueError("InterventionSupport must declare at least one op")
        if not self.targets:
            raise ValueError("InterventionSupport must declare at least one target")
        for axis, declared, known in (
            ("target", set(self.targets), INTERVENTION_TARGETS),
            ("value source", set(self.values), VALUE_SOURCES),
            ("flow (`of`)", set(self.of), FLOWS),
        ):
            if unknown := sorted(declared - set(known)):
                raise ValueError(f"InterventionSupport names unknown {axis}s {unknown}; known: {list(known)}")
        if not self.of:
            raise ValueError("InterventionSupport must declare at least one flow (`of`)")
        for target, detail in self.targets.items():
            expected = _TARGET_DETAIL_TYPES.get(target)
            if detail is not None and (expected is None or type(detail).__name__ != expected.__name__):
                raise ValueError(
                    f"the {target!r} target carries a {type(detail).__name__} detail; "
                    + (f"its detail is a {expected.__name__} or None" if expected else "it takes no detail record")
                )

    def target_detail(self, target: str) -> Any:
        """The detail record of ``target``, or ``None`` when it carries none; a target not declared is refused."""
        if target not in self.targets:
            raise ValueError(f"the {target!r} target is not declared here; declared: {sorted(self.targets)}")
        return self.targets[target]

    @classmethod
    def every(cls, *, targets: Mapping[str, Any] | None = None) -> InterventionSupport:
        """Every op and every hook scope: for a backend whose hook can compute its edit from the activation it sees,
        during the pass.

        Seeing the activation is not the criterion, which is worth stating because it reads like one: a
        steering surface can receive the activation and still express only ``add``, if every parameter it
        takes was fixed before the pass began. See :class:`InterventionMode`. ``targets`` defaults to every target with
        no detail: a hook reaches a whole activation, any of its units, and an attached latent model's features at its
        sub-hook.
        """
        return cls(
            ops=frozenset(InterventionMode),
            position_scopes=PositionScope.hook_scopes(),
            targets=dict.fromkeys(INTERVENTION_TARGETS) if targets is None else targets,
        )


@dataclass(frozen=True)
class LatentModelSupport:
    """How ``LATENT_MODELS`` runs on this backend.

    ``batched_hooks`` says whether ``fwd_w_hooks_batched`` fuses its hook configs into one execution
    (nnsight's multi-invoke) or loops. It lives here rather than in :class:`Capability` because it
    is a property of HOW a method in that group runs, not a surface of its own: every latent-models
    backend implements the method, and a sequential loop is a valid implementation.
    """

    batched_hooks: bool = False


@dataclass(frozen=True)
class CaptureSupport:
    """Which vocabulary points a backend can capture on the model it wraps, as a declaration a case can check.

    The record of :attr:`Capability.CAPTURE`. The member answers "is the surface implemented at all", and a backend
    that captures 181 of 298 points implements it. What varies is WHICH points, so the shape is a support record
    beside :class:`InterventionSupport`, keyed by the
    vocabulary's layer-free base spellings (``ln2.hook_out``, ``hook_resid_pre``, ``unembed.hook_in``) so one
    declaration covers every layer. ``uncapturable`` carries the reason per base, because a point a backend cannot
    capture must be refused by name with that reason rather than returned as a cache that is silently short.

    **Valid for one model instance as it stands when asked.** The record is derived from the backend AND the model
    it wraps, and a wrapper's hooks change with what is attached to it (a HookedSAETransformer with a latent model
    attached exposes points the bare HookedTransformer does not), so a backend provides ``capture_support(model)``
    as a method rather than a property, ``get_module_capabilities`` recomputes it on every call, and a consumer that
    caches one must re-query after attaching or removing a latent model or swapping the wrapper. A cached record is
    the declaration-versus-delivery gap one level out.
    """

    capturable: frozenset[str]
    uncapturable: dict[str, str]
    n_layers: int
    architecture: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "capturable", frozenset(self.capturable))
        object.__setattr__(self, "uncapturable", dict(self.uncapturable))
        overlap = self.capturable & set(self.uncapturable)
        if overlap:
            raise ValueError(f"a point cannot be both capturable and uncapturable: {sorted(overlap)}")
        if not self.capturable:
            raise ValueError("CaptureSupport must declare at least one capturable point")
        # A declaration decides EVERY point of the architecture's inventory, or it is refused by name. Otherwise the
        # fraction it reports is self-referential (a record that omits points shrinks its own denominator, so
        # omission reads as completeness), and the understating direction is the one no capture case catches: an
        # omitted point is neither captured nor refused, only unknown.
        from interpretune.analysis.points import component_map_for
        from interpretune.analysis.points.inventory import inventory

        expected = set(inventory(component_map_for(self.architecture)))
        declared = self.capturable | set(self.uncapturable)
        undecided = sorted(expected - declared)
        if undecided:
            raise ValueError(
                f"the capture declaration for {self.architecture} decides {len(declared)} of {len(expected)} inventory "
                f"points; undecided (declare each capturable or uncapturable with a reason): {undecided}"
            )
        foreign = sorted(declared - expected)
        if foreign:
            raise ValueError(
                f"the capture declaration for {self.architecture} names points outside the architecture's inventory: "
                f"{foreign}"
            )

    @property
    def inventory_size(self) -> int:
        """How many base points the architecture's inventory has: the denominator, equal to the declaration's size
        by the construction invariant above."""
        return len(self.capturable) + len(self.uncapturable)

    def refusal(self, name: str) -> str | None:
        """Why ``name`` cannot be captured here, or ``None`` when it can.

        Parsed through the vocabulary: a layer beyond the model is refused as such, an SAE sub-hook is judged by the
        point it hangs off, a spelling outside the vocabulary is refused as unknown, and a vocabulary point the
        architecture does not define is refused as the architecture's fact rather than the backend's, since the
        declaration decides every point of the inventory and so cannot be silent about one.
        """
        from interpretune.analysis.points.vocabulary import UnknownPointError, parse

        try:
            point = parse(name)
        except UnknownPointError as exc:
            return str(exc)
        if point.layer is not None and point.layer >= self.n_layers:
            return f"{name!r} names layer {point.layer}, and this model has {self.n_layers} blocks"
        from interpretune.analysis.points.vocabulary import declaration_key

        base = declaration_key(name)
        if base in self.capturable:
            return None
        reason = self.uncapturable.get(base)
        if reason is not None:
            return f"{name!r} cannot be captured here: {reason}"
        # A declaration decides every point of the architecture's inventory, so a base that is in neither set is not
        # a gap in the declaration: the architecture defines no such point. That is the resolver's refusal, an
        # architecture-scoped fact, and the record says so rather than presenting a backend gap it does not have.
        return (
            f"{name!r} is not defined on {self.architecture}: the vocabulary resolves {base!r} to no tensor position"
            " in this architecture, so no backend could capture it; this is the resolver's refusal, not a gap in the"
            " backend's capture declaration"
        )

    def can_capture(self, name: str) -> bool:
        """Whether ``name`` is capturable here."""
        return self.refusal(name) is None

    def describe(self) -> str:
        """One line for a report: the fraction and the gaps by name."""
        gaps = ", ".join(sorted(self.uncapturable)) or "-"
        return (
            f"captures {len(self.capturable)} of {self.inventory_size} base points on {self.architecture} "
            f"({self.n_layers} blocks); cannot capture: {gaps}"
        )


def _unwrap_execution_handle(model: Any) -> Any:
    """The HF model an execution wrapper carries, or ``model`` itself when it is one.

    Walks the private handle first (``_model``, which nnsight's wrapper and circuit-tracer's replacement model
    keep as the raw module), then the public ``model`` and ``hf_model``; an nnsight envoy met on the way is unwrapped
    to its module, since its type lives in nnsight and would make a provenance check look at the wrong modeling
    module and pass vacuously. A truth-test is never applied to a candidate: an envoy delegates ``len`` to the
    wrapped model.
    """
    inner = model
    for _ in range(4):
        module_attr = getattr(inner, "_module", None)
        if module_attr is not None and type(inner).__module__.startswith("nnsight"):
            inner = module_attr
            continue
        candidate = None
        for attr in ("_model", "model", "hf_model"):
            found = getattr(inner, attr, None)
            if found is not None and hasattr(found, "config") and found is not inner:
                candidate = found
                break
        if candidate is None:
            break
        inner = candidate
    return inner


def _circuit_tracer_own_attention_implementations() -> tuple[str, ...]:
    """Attention implementation names circuit-tracer registered with transformers itself.

    A circuit-tracer backend may point the model it builds graphs on at a variant of eager attention it registers under
    its own name (a frozen eager, so attention patterns can be replayed) and resolve attention through its own taps
    rather than through the modeling module's function. Such an implementation is what construction on that backend
    NEEDS, not a foreign substitution, so it is accepted by PROVENANCE: any registered implementation whose function
    lives in circuit-tracer's own package. No name is spelled here, so a rename there cannot leave a stale literal
    accepting nothing, and nothing below names any particular backend. Absent circuit-tracer or its registration,
    nothing is added and the check is unchanged.
    """
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return ()
    accepted: list[str] = []
    for name in ALL_ATTENTION_FUNCTIONS.valid_keys():
        fn = ALL_ATTENTION_FUNCTIONS.get(name)
        if str(getattr(fn, "__module__", "")).startswith("circuit_tracer."):
            accepted.append(str(name))
    return tuple(accepted)


@dataclass(frozen=True)
class AttributionGraphSupport:
    """What attribution-graph construction requires of the model it runs on, checked before construction.

    ``requires_own_eager_attention`` is the measured requirement: circuit-tracer resolves gemma's attention-pattern
    location through nnsight's source tracing of the function bound at the attention call site, so the modeling
    module's own ``eager_attention_forward`` must be what is bound there. A TransformerLens bridge of any llama,
    qwen or gemma model replaces that function in both gemma modeling modules for the rest of the process, with the
    config still reading ``eager``; the failure that produced is an ``AttributeError`` deep in circuit-tracer, so the
    check happens here, by provenance, where the name of the foreign function is still available.
    """

    requires_own_eager_attention: bool = True

    def refusal(self, model: Any) -> str | None:
        """Why a graph cannot be built on ``model`` here, or ``None`` when its attention is what construction
        needs."""
        if not self.requires_own_eager_attention:
            return None
        import inspect

        inner = _unwrap_execution_handle(model)
        modeling = inspect.getmodule(type(inner))
        fn = getattr(modeling, "eager_attention_forward", None) if modeling is not None else None
        if fn is None or modeling is None:
            return None  # an architecture without a module-level eager attention resolves its locations another way
        modeling_name = modeling.__name__
        impl = getattr(getattr(inner, "config", None), "_attn_implementation", None)
        if impl not in (None, "eager", *_circuit_tracer_own_attention_implementations()):
            return f"attribution graphs need the eager attention implementation and the model is configured as {impl!r}"
        if getattr(fn, "__module__", None) != modeling_name:
            return (
                f"{modeling_name}.eager_attention_forward is {getattr(fn, '__module__', '?')}."
                f"{getattr(fn, '__qualname__', '?')}, not the modeling module's own: another library replaced it for"
                " this process, and the attention-pattern location is resolved from that function's source. Restore"
                " the original before building a graph (a TransformerLens bridge of a llama, qwen or gemma model is"
                " the known cause)"
            )
        return None

    def describe(self) -> str:
        """One line for a card or a report."""
        return (
            "requires the modeling module's own eager attention"
            if self.requires_own_eager_attention
            else "no model requirement"
        )


@dataclass(frozen=True)
class ModuleCapabilities:
    """The capabilities a module's components provide, each with its providers and its support record.

    ``providers`` maps each declared capability to the components providing it, named by slot (``model_backend``,
    ``analysis_backend``). ``intervention`` keeps one record per provider, because two components providing one target
    is legitimate (a TransformerLens backend reaches an SAE's features at its sub-hook, and circuit-tracer's replacement
    model reaches its transcoders' features): :meth:`intervention_for` resolves a target to its provider and refuses an
    unqualified request with two. Every other surface has one provider, and its record is present iff it is declared:
    ``latent_models`` with ``LATENT_MODELS``, ``attribution_graph`` with ``ATTRIBUTION_GRAPH``. The constructor enforces
    that, so a consumer rendering this (the adapter card, ``adapter_info``, a conformance report) can rely on the record
    being there when the surface is. ``capture`` is the one record that may be absent while declared: it is derived from
    the model at hand, and a module without a model yet has none.
    """

    providers: Mapping[Capability, tuple[str, ...]]
    intervention: Mapping[str, InterventionSupport] = field(default_factory=dict)
    """Each intervention provider's record, keyed by its slot."""
    latent_models: LatentModelSupport | None = None
    capture: CaptureSupport | None = None
    """What the capture provider declares it can capture on this module's model; ``None`` when no provider is
    attached, the module has no model yet, or the provider predates the declaration (a conformance case fails the
    last by name)."""
    attribution_graph: AttributionGraphSupport | None = None
    requires: Mapping[str, frozenset[Capability]] = field(default_factory=dict)
    """What each component needs another component to provide, as declared (already checked to be met)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "providers", {cap: tuple(names) for cap, names in self.providers.items()})
        object.__setattr__(self, "intervention", dict(self.intervention))
        for capability, names in self.providers.items():
            if len(names) > 1 and capability is not Capability.INTERVENTION:
                raise ValueError(
                    f"{capability.name} is provided by {list(names)}; a module takes every surface but INTERVENTION"
                    " from one component, since its record would otherwise be ambiguous"
                )
        if set(self.intervention) != set(self.providers.get(Capability.INTERVENTION, ())):
            raise ValueError(
                f"INTERVENTION is provided by {list(self.providers.get(Capability.INTERVENTION, ()))} and support"
                f" records accompany {sorted(self.intervention)}; a component claiming the surface must say which"
                " configurations of it are supported"
            )
        for capability, record, name in (
            (Capability.LATENT_MODELS, self.latent_models, "latent_models"),
            (Capability.ATTRIBUTION_GRAPH, self.attribution_graph, "attribution_graph"),
            (Capability.CAPTURE, self.capture, "capture"),
        ):
            declared = capability in self.providers
            if declared and record is None and capability is not Capability.CAPTURE:
                raise ValueError(
                    f"{capability.name} is declared but no {name} support record accompanies it; a component "
                    "claiming the surface must say which configurations of it are supported"
                )
            if record is not None and not declared:
                raise ValueError(f"a {name} support record is present but {capability.name} is not declared")

    @property
    def declared(self) -> frozenset[Capability]:
        """Every capability some component provides."""
        return frozenset(self.providers)

    @property
    def values(self) -> frozenset[str]:
        """The capability names as plain strings, for logging and serialization."""
        return frozenset(cap.value for cap in self.declared)

    def supports(self, capability: Capability) -> bool:
        """Whether some component provides ``capability``, compared by value (see
        :func:`get_module_capabilities`)."""
        return normalize_backend_capability(capability).value in self.values

    def target_providers(self, target: str) -> tuple[str, ...]:
        """The components whose intervention record declares ``target``, in slot order."""
        return tuple(name for name, record in self.intervention.items() if target in record.targets)

    def intervention_for(self, target: str, *, provider: str | None = None) -> InterventionSupport:
        """The intervention record of the component providing ``target``, refused by name when it is ambiguous.

        An op that needs one provider of a target two components reach names the path it uses (``provider``); an
        unqualified request is refused rather than resolved by precedence, since either provider returns plausible
        results and a silent choice cannot be detected from them.
        """
        if target not in INTERVENTION_TARGETS:
            raise ValueError(f"{target!r} is not an intervention target; known: {list(INTERVENTION_TARGETS)}")
        found = self.target_providers(target)
        if provider is not None:
            if provider not in found:
                raise ValueError(
                    f"the {target!r} intervention target is not provided by {provider!r}; providers: {list(found)}"
                )
            return self.intervention[provider]
        if not found:
            raise ValueError(
                f"no component of this module provides the {target!r} intervention target"
                f" (intervention providers: {sorted(self.intervention) or 'none'})"
            )
        if len(found) > 1:
            raise ValueError(
                f"the {target!r} intervention target is provided by {list(found)}; name the provider the request uses"
                " rather than relying on a precedence between them"
            )
        return self.intervention[next(iter(found))]

    def target_detail(self, target: str, *, provider: str | None = None) -> Any:
        """The detail record of ``target`` at its provider (see :meth:`intervention_for`)."""
        return self.intervention_for(target, provider=provider).target_detail(target)


#: Spellings that once named a capability and no longer do, each with the spelling that replaced it. Refused by
#: name rather than translated: a caller carrying the old spelling has a manifest or a config to update, and a
#: silent translation would leave it carrying a name nothing else in the vocabulary recognizes.
_RETIRED_CAPABILITY_SPELLINGS: dict[str, str] = {
    "activation_intervention": Capability.INTERVENTION.value,
    "feature_intervention": Capability.INTERVENTION.value,
    "attribution": Capability.ATTRIBUTION_GRAPH.value,
}


def normalize_backend_capability(capability: Any) -> Capability:
    """Normalize a capability-like value to a :class:`Capability` member.

    Accepts a member, a member's value, or a dotted spelling whose last segment is a member name
    (``"Capability.GRADIENTS"``). Compared by value, so a member of a second load of this module normalizes to this
    one's. A retired spelling is refused by name with its replacement; an unknown one is refused with the vocabulary
    listed.
    """
    if isinstance(capability, Capability):
        return capability

    raw_value = getattr(capability, "value", capability)
    spelling = str(raw_value)
    candidate = spelling.split(".")[-1].lower() if "." in spelling else spelling
    if candidate in _RETIRED_CAPABILITY_SPELLINGS:
        raise ValueError(
            f"{spelling!r} is a retired capability spelling; the surface it named is now"
            f" {_RETIRED_CAPABILITY_SPELLINGS[candidate]!r}. Update the declaration rather than relying on a"
            " translation."
        )
    try:
        return Capability(candidate)
    except ValueError:
        raise ValueError(f"{spelling!r} is not a capability: spellings are {[m.value for m in Capability]}") from None


def get_model_backend(module: Any) -> ModelBackend | None:
    """Return the module's model backend while avoiding mock-created private attrs.

    Reads the property without a ``hasattr`` guard: the property may assert on a module that is not
    fully set up, and ``hasattr`` re-raises anything but ``AttributeError`` -- so the guard meant to
    protect the read is what would raise out of this capability probe. "Not set up yet" answers None.
    """
    module_dict = getattr(module, "__dict__", None)
    backend = module_dict.get("_model_backend") if isinstance(module_dict, dict) else None
    if backend is None:
        try:
            backend = module.model_backend
        except (AssertionError, AttributeError):
            backend = None
    return backend


def get_analysis_backend(module: Any) -> AnalysisBackend | None:
    """Return the module's analysis backend, or None when it has none.

    Reads ``__dict__`` directly before touching the ``analysis_backend`` property, because the property
    may assert on a module that is not fully set up -- and "not set up yet" must answer None here rather
    than raising out of a capability probe. The property itself is read without a ``hasattr`` guard for
    the same reason: ``hasattr`` re-raises an assertion instead of answering False.
    """
    module_dict = getattr(module, "__dict__", None)
    backend = module_dict.get("_analysis_backend") if isinstance(module_dict, dict) else None
    if backend is None:
        try:
            backend = module.analysis_backend
        except (AssertionError, AttributeError):
            backend = None
    return backend


# Named analysis backends (hydration seam): names are the PORTABLE reference artifacts use — an
# it_artifact.json envelope can only carry a backend NAME (instances are not wire-format). Backends
# register at import; resolve_analysis_backend lazily imports interpretune.analysis.backends.<name>
# on a miss before failing, keeping resolution extensible without eager imports.
ANALYSIS_BACKEND_REGISTRY: dict[str, "AnalysisBackend"] = {}


def register_analysis_backend(name: str, backend: "AnalysisBackend") -> None:
    """Register a named analysis backend (idempotent for the same object)."""
    ANALYSIS_BACKEND_REGISTRY[name] = backend


# Bundled backend NAME -> the module whose import registers it. An explicit table rather than a path
# built from the name: the ImportError below is swallowed, so a convention-derived path that goes stale
# degrades into "no backend registered" instead of failing where the cause is visible. A table breaks
# loudly at review time when a module moves, and `test_backend_name_resolution.py` resolves every entry
# from a cold registry as a positive control -- the failure mode here is an ABSENCE, which passes
# silently when nothing checks it.
_BUNDLED_BACKEND_MODULES: dict[str, str] = {
    "circuit_tracer": "interpretune.adapters.circuit_tracer.backends",
}


def resolve_analysis_backend(name: str) -> "AnalysisBackend":
    """Resolve a backend NAME to its registered instance, lazily importing the bundled module.

    Bundled backends register themselves at import time, so a name that is not in the registry yet may simply not have
    been imported. Hub-delivered backends register from their component entrypoint and are already present by the time
    anything resolves them, so they never reach the table.
    """
    if name not in ANALYSIS_BACKEND_REGISTRY:
        module_path = _BUNDLED_BACKEND_MODULES.get(name)
        if module_path is not None:
            import importlib

            try:
                importlib.import_module(module_path)
            except ImportError:
                pass
    if name not in ANALYSIS_BACKEND_REGISTRY:
        raise KeyError(
            f"No analysis backend registered as {name!r} (known: {sorted(ANALYSIS_BACKEND_REGISTRY)}). "
            "Hydrating this artifact requires the backend's package/extra to be installed."
        )
    return ANALYSIS_BACKEND_REGISTRY[name]


def require_analysis_backend(module: Any) -> AnalysisBackend:
    """Return the module's analysis backend or raise if it is unavailable."""

    backend = get_analysis_backend(module)
    if backend is None:
        raise ValueError("Target module must expose an analysis_backend for this operation")
    return backend


def _components(module: Any) -> dict[str, Any]:
    """The module's attached components by slot (:data:`COMPONENTS`), omitting an empty slot."""
    found = {"model_backend": get_model_backend(module), "analysis_backend": get_analysis_backend(module)}
    return {name: component for name, component in found.items() if component is not None}


def _declared(component: Any, attr: str) -> frozenset[Capability]:
    """A component's ``capabilities`` or ``requires`` set, normalized (a component that declares no ``requires``
    requires nothing)."""
    raw = getattr(component, attr, None) if attr == "capabilities" else getattr(component, attr, frozenset())
    return frozenset(normalize_backend_capability(c) for c in (raw or ()))


def get_module_capabilities(module: Any) -> ModuleCapabilities:
    """Aggregate the capabilities a module's components provide, and refuse a requirement nothing meets.

    Each component (:data:`COMPONENTS`) declares ``capabilities``, a support record per capability, and ``requires``:
    the capabilities it needs ANOTHER component of the composition to provide (circuit-tracer's analysis backend builds
    graphs on what the model backend computes). A requirement no other component provides is refused here, naming the
    component and what is missing, because the alternative is a surface that is declared and cannot run.
    """
    components = _components(module)
    declared = {name: _declared(component, "capabilities") for name, component in components.items()}
    requires = {name: _declared(component, "requires") for name, component in components.items()}
    for name, needed in requires.items():
        elsewhere = {cap for other, caps in declared.items() if other != name for cap in caps}
        if missing := sorted(cap.name for cap in needed - elsewhere):
            raise ValueError(
                f"the {name} ({type(components[name]).__name__}) of {type(module).__name__} requires {missing} from"
                f" another component, and none provides it (provided elsewhere: {sorted(c.name for c in elsewhere)});"
                " compose it with a component that does"
            )
    providers: dict[Capability, tuple[str, ...]] = {}
    for name, caps in declared.items():
        for cap in sorted(caps, key=lambda c: c.value):
            providers[cap] = (*providers.get(cap, ()), name)

    def _record(capability: Capability, attr: str) -> Any:
        names = providers.get(capability, ())
        return _support_record(components[names[0]], attr) if names else None

    capture_names = providers.get(Capability.CAPTURE, ())
    return ModuleCapabilities(
        providers=providers,
        intervention={
            name: _support_record(components[name], "intervention_support")
            for name in providers.get(Capability.INTERVENTION, ())
            if _support_record(components[name], "intervention_support") is not None
        },
        latent_models=_record(Capability.LATENT_MODELS, "latent_model_support"),
        capture=_capture_record(components[capture_names[0]], module) if capture_names else None,
        attribution_graph=_record(Capability.ATTRIBUTION_GRAPH, "attribution_graph_support"),
        requires={name: needed for name, needed in requires.items() if needed},
    )


def _capture_record(backend: Any, module: Any) -> CaptureSupport | None:
    """The capture provider's declaration for ``module.model``, or ``None`` when it has none to give.

    Capturability is a property of the backend AND the model it wraps (a TransformerLens backend captures different
    points on a bridge than on a HookedTransformer), so the declaration is a method taking the model rather than a
    property. A provider without the method is not refused here, because this aggregation feeds the adapter card and
    ``adapter_info`` on a bare install; the conformance suite is where its absence fails by name.
    """
    declare = getattr(backend, "capture_support", None)
    model = getattr(module, "model", None)
    if declare is None or model is None:
        return None
    return declare(model)


def _support_record(component: Any, attr: str) -> Any:
    """The support record a component attaches under ``attr``, or ``None`` when it attaches none.

    Read with ``getattr`` rather than through the protocol so a component that declares the surface and
    forgot the record fails in :class:`ModuleCapabilities`' invariant with a message naming the record,
    instead of as an ``AttributeError`` here.
    """
    return getattr(component, attr, None)
