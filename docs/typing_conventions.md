# Static typing conventions

Interpretune has exactly one static type checker: pyright, at `standard` strictness over `src/`,
run by the type-check CI workflow. mypy is not configured, not installed in CI, and not a
development dependency, so nothing in the tree should be written for it. In particular,
`# type: ignore[...]` comments are retired: `enableTypeIgnoreComments = false` in
`pyproject.toml` makes pyright disregard them, and every remaining suppression is a scoped
`# pyright: ignore[<rule>]` directive naming the one rule it silences.
`reportUnnecessaryTypeIgnoreComment = "error"` makes a stale suppression fail CI, so the count
cannot creep back in either form.

CI is the arbiter. Development environments install several dependencies editable or from source
and can disagree with the pinned dependency set on individual lines, in both directions. The
local loop that works is: remove a suppression, run pyright on the file, and let the hosted
type-check judge the result under the pins.

Most suppressions this codebase once carried were not third-party typing gaps. They were the
checker being unable to see across a composition seam: a mixin reading an attribute its host
class provides, a batch field the dispatcher had already validated, a protocol typed narrower
than the path that consumes it. For each shape there is one sanctioned remedy, preferred in this
order.

## 1. Declare what the host provides

A mixin that reads attributes or methods supplied by another class in the composition declares
them in a `if TYPE_CHECKING:` block at the top of the mixin, typed as the host provides them,
with a one-line comment saying who provides them. Plain annotations carry no runtime surface: no
attribute is created, nothing shadows the host's value, and every subsequent use in the mixin is
checked for real, including assignment and deletion. Host methods are declared as stubs;
attributes the host exposes as properties are declared in property shape, so the declaration
composes without an incompatible-override error where the classes meet. A mutable attribute
cannot be declared with incompatible types across sibling bases, so where siblings need different
model types, the broader mixins declare the attribute `Any` and say why in one line, and the one
mixin that needs precision declares the narrow type.

Examples: `AnalysisStepMixin` and the other base mixins in
`src/interpretune/base/components/mixins.py`, `LightningCLIAdapter` in
`src/interpretune/base/components/cli.py`, the circuit-tracer mixins in
`src/interpretune/adapters/circuit_tracer/adapter.py`. The same rule applies when a whole class
owns a value narrower than its base: `AnalysisRunner.run_cfg` is declared once as
`AnalysisRunnerCfg` rather than re-narrowed at every read, and `AnalysisStore.dataset` is typed
as the single-split `Dataset` the store actually holds, with the constructor's broader accepted
inputs resolved at the boundary.

## 2. Assert at the boundary the dispatcher already validated

Where a value is optional in its declared type but an earlier phase guarantees it is set (a
dispatcher-validated batch field, a value a `setup()` resolved), capture it in a local and assert
once, with a message naming the guarantee. The assertion is checked state, not ceremony: the
variable that would otherwise need a guard at every use carries its narrowed type onward, and a
genuinely missing value fails loudly at the boundary instead of as an attribute error deep in an
op. Examples: `label_ids` in `get_loss_preds_diffs`
(`src/interpretune/analysis/optools.py`), `core_log_dir` in the circuit-tracer adapter's setup.

## 3. Declare what a path uses on the protocol it is typed with

When code typed against a protocol reads fields the protocol does not declare, the fix is in the
protocol, not at the call sites. Fields a shared path attaches to every family belong on the base
protocol: `BaseAnalysisBatchProtocol` declares `prompts`, `tokens`, and `keys()` because the
persistence path uses them on batches of every protocol family. A protocol member must be
declared in the shape the concrete class has it: `_processed_analysis_cfgs` is a property, not a
mutable field, and declaring it as one made the concrete config unassignable to its own protocol.
Where a consumer needs only a small surface, type the parameter with the protocol that declares
exactly that surface (`TestLoadable` for a datamodule only read through its test dataloader)
rather than with a concrete class that does not provide it at all.

One caution follows from how `typing` checks protocols at runtime: `isinstance` against a
runtime-checkable protocol resolves members with `getattr_static`, which cannot see attributes
provided dynamically, such as a submodule registered on an `nn.Module`. A protocol that declares
such a member rejects every real instance at runtime while satisfying the static checker, so an
`isinstance` assert against it is worse than none. `DebugGeneration._check_phandle` documents the
case: the cast states the static contract, and there is deliberately no runtime check.

## 4. Scope what is left to the exact rule

What survives the three remedies above gets `# pyright: ignore[<rule>]` on the line pyright
reports, with a short reason after it. The recurring legitimate cases:

- Cooperative `super()` calls inside mixins, which resolve only in the composed MRO and are
  invisible to the checker from the mixin alone (`reportAttributeAccessIssue`).
- Composition class definitions whose bases declare a variable incompatibly by design, where the
  composition is the point (`reportIncompatibleVariableOverride` and friends, on the class line).
- Third-party APIs whose stubs disagree with the runtime contract in a way one call site cannot
  fix (a `datasets` or Lightning signature the pinned stub models wrongly).
- Deliberate runtime-only shapes: fallback `object` assignments when an optional framework is
  absent, enum and protocol metaprogramming in `src/interpretune/protocol.py`.

A scoped directive must name every rule the line actually trips, and only those: an over-broad
directive is the whole-line blanket returning under a new spelling, and
`reportUnnecessaryTypeIgnoreComment` already catches the wholly stale kind.

`cast` has two legitimate shapes and no others: specifying which overload a third-party API
means, and narrowing at a boundary where the guarantee comes from something the checker cannot
see (for example, that an HF model's base model is itself a `PreTrainedModel` where the stub
types it as `nn.Module`). A cast is never a substitute for a declaration: if the same cast would
be needed at a second site, declare instead. Dynamic attribute attachment (a tracking attribute
set on a class, function, or third-party object that does not declare it) is spelled `setattr`,
which is both honest and unchecked, rather than an assignment carrying a suppression.
