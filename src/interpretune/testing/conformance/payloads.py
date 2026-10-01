"""YAML-expressible references to intervention payloads in conformance samples.

A ``conformance.run_inputs`` sample is declared in YAML, so every value in it must be YAML-expressible.
Plain inputs (prompts, scales, scopes) are. Intervention payloads are not: an ``intervention_tensor``
is a ``torch.Tensor` built for one model, and writing one into YAML would freeze model-specific numbers
into a definition file while teaching nothing. The 09-06 residual on interpretune#450 named the two
forms such a reference could take -- a captured activation by point, or a fixture name the suite
resolves. This module is the second form; ``capture:`` stays a named future, stated below rather than
left for someone to reinvent.

A reference is a mapping with exactly one key, ``fixture``:

.. code-block:: yaml

    conformance:
      run_inputs:
        intervention_tensor: {fixture: pooled_direction}

Resolution replaces the mapping with whatever the named factory returns for the running session, so the
Resolution replaces the mapping with whatever the named factory builds for the running session, so the
same declaration measures the same thing on every model: the NAME is shared, the tensor is built for
the target. Anything that is not exactly one ``fixture`` key passes through untouched -- a plain
mapping stays a plain mapping, which is what keeps a second reference form addable without
reinterpretation. Unknown names, non-string names and almost-refs (a ``fixture`` key beside other keys,
which reads as two forms in one mapping) are refused by name: a sample that silently ran on the wrong
tensor would pass green while asserting nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

#: A payload factory builds the tensor for the session under test. It receives the conformance session
#: (so shapes come from the target's module, never from YAML) and returns the payload object.
PayloadFactory = Callable[[Any], Any]

#: The single key whose mapping is a payload reference. Anything else passes through.
FIXTURE_REF_KEY = "fixture"


def fixture_ref_name(value: Any) -> str | None:
    """The fixture name when ``value`` is a payload reference, else ``None``.

    A reference is exactly ``{FIXTURE_REF_KEY: <name>}``. Near-misses are refused rather than passed
    through, because a mapping carrying ``fixture`` plus siblings reads as a reference the resolver
    half-understood, and running it as a plain dict would execute something the author did not declare.
    """
    if not isinstance(value, dict) or FIXTURE_REF_KEY not in value:
        return None
    if len(value) != 1:
        raise ValueError(
            f"a payload reference is exactly {{'{FIXTURE_REF_KEY}': <name>}}; got keys "
            f"{sorted(map(str, value))}. If a second reference form is needed, add it as its own "
            "single-key mapping rather than beside 'fixture'."
        )
    name = value[FIXTURE_REF_KEY]
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"a payload reference names its fixture as a non-empty string; got {name!r}. "
            "Name the fixture the suite should build for this session."
        )
    return name


def resolve_payload_refs(
    run_inputs: dict[str, Any],
    fixtures: dict[str, PayloadFactory],
    session: Any,
    *,
    source: str = "<sample>",
) -> dict[str, Any]:
    """Replace every payload reference in ``run_inputs`` with the fixture built for this session.

    Only top-level values are examined: a reference nested inside another mapping is a shape this
    resolver does not claim to understand, and flattening it by guess would be the silent
    reinterpretation the refusal above exists to prevent.
    """
    resolved: dict[str, Any] = {}
    for key, value in run_inputs.items():
        name = fixture_ref_name(value)
        if name is None:
            resolved[key] = value
            continue
        if name not in fixtures:
            known = sorted(map(str, fixtures)) or ["<none registered>"]
            raise ValueError(
                f"{source}: sample references unknown fixture {name!r} (known: {known}). Register it "
                "on the target's `ConformanceInputs.payload_fixtures` so the tensor is built for the "
                "session under test rather than frozen into YAML."
            )
        resolved[key] = fixtures[name](session)
    return resolved
