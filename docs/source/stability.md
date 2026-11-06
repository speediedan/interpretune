# Stability Policy

Interpretune is in its alpha line. This page states what that means for anyone building on it: what may
change, what gets notice first, and what is explicitly not covered. It is the public source of truth for
the project's backwards-compatibility posture.

## The promise during alpha is notice, not stability

Through the alpha line, backwards compatibility is kept minimal on purpose. There are no deprecation
shims, no fallback code paths, and no silent coercions preserving an older calling convention. When a
contract changes, every in-tree caller and test is updated to the new contract directly.

What users get instead is predictability: every breaking change lands with a `Breaking` entry in the
[CHANGELOG](https://github.com/speediedan/interpretune/blob/main/CHANGELOG.md) naming the old and new
form. That entry is the whole obligation, and it is not optional.

## A small stable surface gets one release of notice

Everything a quickstart or a published example touches is treated more carefully. Breaking one of these
is allowed during alpha, but it is announced in the release before it happens:

- the top-level verb surface: the `interpretune.hub` load and pull functions, and the top-level op
  wrappers (`interpretune.concept_direction`, `interpretune.compute_attribution_graph`, and their peers)
- `ITSession` construction and the documented config entry points
- the documented CLI entry points

## Everything else is explicitly unstable

The following may change in any release, with no notice beyond the CHANGELOG entry that ships with the
change:

- op input and output schemas
- `BaseAnalysisBatchProtocol` and the protocols derived from it
- dispatcher internals and registry key shapes
- adapter MRO composition order
- generated type stubs
- anything under `tests/`

If you build on one of these, pin an exact interpretune version and read the CHANGELOG before upgrading.

## Persisted artifacts are different

None of the above covers persisted artifacts: Hub artifacts and published component manifests. Code can
move fast because its reader and writer are the same checkout. A persisted artifact outlives the code
that wrote it, so artifacts get real compatibility handling: a versioned schema with a stated readable
floor, so an older artifact either loads or is refused by name with the reason. That handling is tracked
separately from this policy.

## Exit criterion

This policy is re-evaluated at the first non-alpha release (0.1.0). The default successor is:

- the stable surface above gains one-minor-release deprecation notice with a runtime warning, and
- the unstable list shrinks to whatever is still genuinely internal at that point.
