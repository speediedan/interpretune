"""Whether this environment satisfies a declared requirement, for hub components and bundled adapters alike.

**One predicate, two sources.** A hub component declares its requirements in ``it_component.yaml``; a bundled
adapter declares them as ``__it_requires__`` in its package ``__init__``. Both are answered here, in one
vocabulary, so "is this composition available here" cannot come to mean two different things depending on
where the adapter came from.

This module lives outside ``interpretune.hub`` deliberately: **a predicate shared by both paths cannot sit
inside one of them.** The bundled registration pass (:mod:`interpretune.adapters._light_register`) imports
only the standard library, and making it reach into the hub subsystem to answer "is this available here"
would invert the layering -- the bundled path would depend on the hub path to describe itself.

The argument is layering, NOT import cost. ``huggingface_hub`` and ``yaml`` are already resident after
``import interpretune`` (pulled by ``transformers`` and ``jsonargparse`` respectively, measured), so
importing ``hub.components`` here would not newly pull them in. An earlier revision of this docstring
claimed it would; that was wrong, and the layering reason is the one that survives checking.
"""

from __future__ import annotations

from dataclasses import dataclass


def installed_version(dist_name: str) -> str | None:
    """Installed distribution version, or ``None`` when metadata is unavailable (e.g. raw checkouts)."""
    from importlib.metadata import version

    try:
        return version(dist_name)
    except Exception:
        return None


@dataclass(frozen=True)
class UnmetRequirement:
    """One unsatisfied requirement, as DATA rather than as prose or as an exception.

    Not an exception instance: constructing exceptions nobody raises invites raising one from the soft path
    at the wrong layer, and traceback semantics mean nothing for a skip. Not a bare string either -- a caller
    asking "can my environment support this composition, and if not why" should branch on ``kind`` rather
    than parse ``message``. ``message`` is verbatim what the hard path raises, so
    :func:`~interpretune.hub.components.enforce_component_requires` stays character-identical to its
    pre-split behaviour.
    """

    kind: str  # "interpretune" | "adapters" | "pip"
    detail: str  # the specifier, adapter name, or pip requirement at fault
    message: str


def install_command(entry: str) -> str:
    """The command a user should run to satisfy one unmet ``requires.pip`` entry.

    ALWAYS the `uv` form, never bare `pip`, and that is a project rule rather than a preference. This
    project's environments depend on `[tool.uv] override-dependencies`, which pip ignores; `AGENTS.md`
    states that `pip install` into a dev venv is never acceptable, "not even for a one-off repair", and
    records a case where a pip repair layered two dist-infos for one package. A diagnostic that printed a
    pip command would be telling users to do the one thing the project documents most carefully against.

    Quoted because specifiers contain characters a shell treats as its own: `interp-engine~=1.5.1` is
    fine unquoted, `interp-engine>=1.5,<2` is not, and the difference is invisible until someone pastes
    the second one.

    THE ENTRY IS EMITTED VERBATIM, which is what makes a non-PyPI dependency work. A PEP 508 direct
    reference carries its own source, so `circuit-tracer @ git+https://.../repo.git@<sha>` round-trips
    into a command that installs the right thing from the right place. That is the form an author whose
    dependency is not on the configured index should declare.

    A BARE NAME IS A GUESS, and the caller should know it. `circuit-tracer>=0.5.3` produces
    `uv pip install 'circuit-tracer>=0.5.3'`, which is only correct if that distribution resolves from
    the configured index. If it does not, the command fails loudly, which is survivable -- but if the
    name happens to belong to an unrelated project on the index, it installs that instead. Nothing here
    can tell the two apart without the network, so the remedy is the direct-reference form above rather
    than a cleverer message.
    """
    return f"uv pip install {entry!r}"


def requirement_status(requires: dict, source: str = "<component>") -> list[UnmetRequirement]:
    """Evaluate a ``requires`` block against this environment, returning EVERY unmet requirement.

    This is the single predicate behind both dispositions: the hard path raises on the first unmet
    requirement, while the soft path skips one composition and registers the rest. They differ only in what
    they do with this list.

    Evaluating a declared requirement is also strictly better than importing and catching: it yields a
    REASON ("circuit-tracer is not installed"), where a caught ``ImportError`` yields a symptom that may have
    nothing to do with a missing optional dependency -- a genuine bug inside an adapter is otherwise
    indistinguishable from an absent package.

    THE EVALUATION ORDER (interpretune, then adapters, then modules, then pip) IS LOAD-BEARING, not stylistic: the hard
    path raises ``unmet[0]``, so this order is what keeps its message identical to the pre-split one when
    several requirements are unmet at once. Looping over ``requires`` in dict order instead would silently
    change which error a user sees.
    """
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet

    unmet: list[UnmetRequirement] = []
    req = requires or {}
    it_spec = req.get("interpretune")
    if it_spec:
        it_version = installed_version("interpretune")
        if it_version is None:  # raw-checkout fallback; skip when genuinely undeterminable
            import interpretune

            it_version = getattr(interpretune, "__version__", None)
        if it_version is not None and not SpecifierSet(str(it_spec)).contains(it_version, prereleases=True):
            unmet.append(
                UnmetRequirement(
                    "interpretune",
                    str(it_spec),
                    f"{source}: requires interpretune {it_spec!r} but {it_version!r} is installed. "
                    "(If this version looks wrong for your checkout, stale packaging metadata — e.g. an old "
                    "src/*.egg-info directory — can shadow the real installation when src/ is on sys.path.)",
                )
            )
    from interpretune.protocol import Adapter

    for name in req.get("adapters") or []:
        if name not in Adapter.__members__:
            unmet.append(
                UnmetRequirement(
                    "adapters",
                    str(name),
                    f"{source}: requires adapter {name!r}, which this interpretune does not provide "
                    f"(known: {sorted(Adapter.__members__)}) — a newer interpretune release may be required.",
                )
            )
    # `modules`: the importable modules a composition actually uses, evaluated with find_spec (no import, no side
    # effects). A distribution name is the wrong grain when the dependency is one module of a package that moves
    # faster than its releases, or a git-pinned package no specifier can express: a checkout two commits behind its
    # pin had the distribution installed and the module absent, and the composition registered and then broke on
    # first use. Skipped-and-reported is what the requires design promises; this axis is what lets it deliver.
    import importlib.util

    for name in req.get("modules") or []:
        try:
            present = importlib.util.find_spec(str(name)) is not None
        except (ModuleNotFoundError, ValueError):
            present = False  # a missing parent package raises rather than returning None
        if not present:
            unmet.append(
                UnmetRequirement(
                    "modules",
                    str(name),
                    f"{source}: requires importable module {name!r}, which is not present. The distribution may be "
                    "installed at a version or checkout that lacks it.",
                )
            )
    for entry in req.get("pip") or []:
        r = Requirement(str(entry))
        installed = installed_version(r.name)
        if installed is None:
            unmet.append(
                UnmetRequirement(
                    "pip",
                    str(entry),
                    f"{source}: requires pip package {entry!r}, which is not installed. "
                    f"Install it with: {install_command(str(entry))}",
                )
            )
        elif r.specifier and not r.specifier.contains(installed, prereleases=True):
            unmet.append(
                UnmetRequirement(
                    "pip",
                    str(entry),
                    f"{source}: requires {entry!r} but {r.name} {installed!r} is installed. "
                    f"Change it with: {install_command(str(entry))}",
                )
            )
    return unmet
