"""The circuit-tracer backend registries must be reachable without circuit-tracer installed.

They are the seam a hub-delivered adapter enters to register its backend. Registering a backend NAME
needs nothing from circuit-tracer, so requiring the package to reach the registry made an OPTIONAL
dependency mandatory for every consumer of such a component -- and made
``CircuitTracerConfig`` unable to VALIDATE without it.

Registering a backend name is not the same as registering a COMPOSITION: a composition including the
circuit-tracer adapter needs that adapter's class and therefore genuinely requires the package. The
negative control below pins that distinction, so a later change cannot quietly erase it.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REGISTRY_MODULE = (
    Path(__file__).parent.parent.parent / "src" / "interpretune" / "adapters" / "circuit_tracer" / "registry.py"
)


def test_registry_module_has_no_module_level_circuit_tracer_import():
    """Static guard, so the regression is caught at the source rather than only under a blocker."""
    tree = ast.parse(REGISTRY_MODULE.read_text(encoding="utf-8"), filename=str(REGISTRY_MODULE))
    offenders = []
    for node in tree.body:  # module level ONLY; function-local imports are the documented pattern here
        if isinstance(node, ast.Import):
            offenders += [a.name for a in node.names if a.name.split(".")[0] == "circuit_tracer"]
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "circuit_tracer":
                offenders.append(node.module)
    assert not offenders, (
        f"{REGISTRY_MODULE.name} imports circuit-tracer at module level ({offenders}); that reintroduces the "
        "hard dependency this module exists to remove. Anything needing circuit-tracer belongs in adapter.py."
    )


# The probe imports interpretune BEFORE installing the blocker, so the availability check in
# `utils.import_utils` sees the truth; the blocker then simulates a consumer whose environment lacks
# circuit-tracer at the moment a hub entrypoint runs.
_PROBE = textwrap.dedent("""
    import sys, importlib.abc, warnings
    warnings.filterwarnings("ignore")
    import interpretune  # noqa: F401

    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "circuit_tracer" or name.startswith("circuit_tracer."):
                raise ModuleNotFoundError(f"No module named {name!r}")
            return None

    sys.meta_path.insert(0, Blocker())
    for mod in [m for m in sys.modules if m.startswith("circuit_tracer")]:
        del sys.modules[mod]
    for mod in [m for m in sys.modules if "adapters.circuit_tracer" in m]:
        del sys.modules[mod]

    # POSITIVE CONTROL. A blocker that silently fails to block reports "imports fine" for a module it
    # never guarded, which is indistinguishable from the result we want. Prove it blocks first.
    try:
        import circuit_tracer  # noqa: F401
        print("CONTROL_FAILED")
        raise SystemExit(0)
    except ModuleNotFoundError:
        print("CONTROL_OK")

    from interpretune.adapters.circuit_tracer.registry import CT_BACKEND_REGISTRY, CT_MODEL_BACKEND_FACTORIES
    print("REGISTRY_OK", sorted(CT_BACKEND_REGISTRY), sorted(CT_MODEL_BACKEND_FACTORIES))

    from interpretune.config import CircuitTracerConfig
    print("VALIDATE_OK", CircuitTracerConfig(backend="transformerlens").backend)

    # NEGATIVE CONTROL: adapter.py genuinely needs circuit-tracer, so it must still fail. If this ever
    # succeeds, the probe is not exercising absence and the two results above mean nothing.
    try:
        from interpretune.adapters.circuit_tracer.adapter import CircuitTracerModuleMixin  # noqa: F401
        print("NEGATIVE_CONTROL_FAILED")
    except ModuleNotFoundError:
        print("NEGATIVE_CONTROL_OK")
""")


@pytest.mark.usefixtures("cleanup_memory")
def test_registry_and_config_work_without_circuit_tracer_installed():
    """Runtime proof, in a subprocess so the blocker cannot leak into the rest of the suite."""
    proc = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, timeout=300)
    out = proc.stdout
    assert "CONTROL_OK" in out, f"blocker did not block circuit_tracer; probe proves nothing.\n{out}\n{proc.stderr}"
    assert "REGISTRY_OK" in out, f"registry unreachable without circuit-tracer.\n{out}\n{proc.stderr}"
    assert "VALIDATE_OK" in out, f"CircuitTracerConfig could not validate without circuit-tracer.\n{out}\n{proc.stderr}"
    assert "NEGATIVE_CONTROL_OK" in out, (
        f"adapter.py imported without circuit-tracer, so the probe was not exercising absence.\n{out}\n{proc.stderr}"
    )
