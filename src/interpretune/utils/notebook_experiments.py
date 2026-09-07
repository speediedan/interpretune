"""Rails for driving a notebook experiment, from inside this repository or from anywhere else.

The experiment infrastructure this replaces could only run from inside this tree: it inferred the
repository root from the notebook's depth, resolved shared base configs by relative path, imported one
specific experiment's helpers into the shared harness, and shipped a default pointing at one machine's
home directory. Each of those is fine until an experiment lives somewhere else, at which point the
shared infrastructure stops being shareable, which is the thing it exists to be.

Parameterized from the consuming repository's ``[tool.interpretune.experiments]``, exactly as the
notebook publisher is parameterized from ``[tool.interpretune.notebooks]``, so an out-of-tree experiment
consumes the rails rather than a copy of them::

    [tool.interpretune.experiments]
    config_dir    = "configs"       # relative to the pyproject directory
    output_root   = "..."           # default: <notebook dir>/generated_experiments
    harness_paths = ["."]           # extra sys.path entries, relative to the root

Every key has a default, so a repository that writes no table gets working behaviour.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

from interpretune.utils.notebook_publishing import _load_toml, find_root

TABLE = ("tool", "interpretune", "experiments")
#: An ``EXTENDS`` value of the form ``package.module:resource`` names a file inside an INSTALLED
#: package. It is the spelling entry points already use, and it works for any installed package rather
#: than privileging this one, which is the property the adapter rails established and this reuses.
PACKAGE_RESOURCE_SEPARATOR = ":"


@dataclass(frozen=True)
class ExperimentsConfig:
    """Where an experiment's configs come from, where its artifacts go, and what else is importable."""

    root: Path
    config_dir: Path | None = None
    output_root: Path | None = None
    harness_paths: tuple[Path, ...] = ()

    @classmethod
    def from_pyproject(cls, root: Path | None = None) -> ExperimentsConfig:
        """Read ``[tool.interpretune.experiments]`` from ``root/pyproject.toml``; keys fall back to defaults.

        ``root`` defaults to the nearest ancestor of the working directory holding a ``pyproject.toml``,
        rather than to a fixed number of levels up. The depth of a notebook below its repository root is
        not a property anyone declared, so inferring it is a guess that happens to hold in one layout.
        """
        root = (root or find_root()).resolve()
        table: dict[str, Any] = {}
        pyproject = root / "pyproject.toml"
        if pyproject.is_file():
            data: Any = _load_toml(pyproject)
            for key in TABLE:
                data = data.get(key) or {}
            table = dict(data)
        raw_output = table.get("output_root")
        return cls(
            root=root,
            config_dir=(root / raw_config_dir).resolve() if (raw_config_dir := table.get("config_dir")) else None,
            output_root=(root / raw_output).resolve() if raw_output else None,
            harness_paths=tuple((root / p).resolve() for p in table.get("harness_paths", ())),
        )


def bootstrap_experiment_imports(
    cwd: Path | None = None, *, extra_paths: Iterable[str | Path] | None = None
) -> ExperimentsConfig:
    """Put the experiment's repository root and declared harness paths on ``sys.path``.

    Deliberately adds only what a repository declares plus its own root. An earlier version appended
    ``tests/`` and ``tests/nb_experiments/`` unconditionally; the second of those no longer exists even
    here, which is the argument against encoding a layout rather than reading one.
    """
    working_dir = (cwd or Path.cwd()).resolve()
    config = ExperimentsConfig.from_pyproject(find_root(working_dir))
    for path in (
        config.root,
        *config.harness_paths,
        working_dir,
        *(Path(p).expanduser().resolve() for p in extra_paths or ()),
    ):
        if (path_str := str(path)) not in sys.path:
            sys.path.insert(0, path_str)
    return config


def resolve_extends_path(config_path: Path, raw_value: str) -> Path:
    """Resolve one ``EXTENDS`` entry, either as a package resource or as a path.

    ``package.module:resource`` names a file inside an installed package, which is how an out-of-tree
    experiment reaches base configs that ship in a wheel rather than sitting at a relative path.
    Anything else is a filesystem path, absolute or relative to the config that named it, unchanged from
    before so nothing in-tree moves.

    A package-resource base is **read-only**: resources may live inside a zip or a wheel and have no
    stable location on disk. A config that expects to write next to its base must name it by path.
    """
    if raw_value.startswith("<"):
        raise ValueError(
            f"{config_path}: EXTENDS value {raw_value!r} looks like a placeholder. There is no reserved "
            "prefix for the shared configs, deliberately: naming one package as special is the assumption "
            "the component rails removed. Use `package.module:resource`, for example "
            "`it_examples.experiments.notebook:configs/base.yaml`, or a path relative to this config."
        )
    if PACKAGE_RESOURCE_SEPARATOR in raw_value and not Path(raw_value).exists():
        package, _, resource = raw_value.partition(PACKAGE_RESOURCE_SEPARATOR)
        return _package_resource_path(config_path, package, resource)
    candidate = Path(raw_value).expanduser()
    return candidate if candidate.is_absolute() else (config_path.parent / candidate).resolve()


@lru_cache(maxsize=None)
def _materialize_resource(package: str, resource: str) -> Path:
    """A real filesystem path for a package resource, copying it out only when it is not already one.

    An installed package is usually unpacked, and then the traversable IS a ``Path`` and is returned
    unchanged. Inside a zipped wheel it is not: ``str()`` of it yields something path-shaped that no
    loader can open, so the failure would surface later as an unrelated read error rather than here.
    ``as_file`` is the supported way to get a real path, and its context manager may delete the file on
    exit, so the copy is taken while it is open and cached for the process.
    """
    from importlib.resources import as_file, files

    traversable = files(package).joinpath(resource)
    if isinstance(traversable, Path):
        return traversable
    cache_dir = (Path(tempfile.gettempdir()) / "interpretune_extends" / package).resolve()
    # Keyed by the FULL resource path, not its basename. A config tree routinely holds several files of
    # the same name at different depths (`configs/base.yaml` and `configs/gemma/base.yaml`), and a
    # basename key sends both to one destination: the second materialization overwrites what the first
    # call's cached path points at, so the first EXTENDS silently reads the second resource's content.
    destination = (cache_dir / resource).resolve()
    if cache_dir not in destination.parents:
        raise ValueError(f"resource {resource!r} in {package!r} escapes the materialization cache")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with as_file(traversable) as real_path:
        # The cache directory is shared between processes, and `copyfile` is not atomic: a concurrent
        # reader can observe a partially written file. Write a sibling and rename, which is.
        staged = destination.with_name(f"{destination.name}.{os.getpid()}.tmp")
        shutil.copyfile(real_path, staged)
        os.replace(staged, destination)
    return destination


def _package_resource_path(config_path: Path, package: str, resource: str) -> Path:
    """Resolve ``package.module:resource``, distinguishing a missing package from a missing resource."""
    from importlib.resources import files

    try:
        traversable = files(package).joinpath(resource)
    except (ModuleNotFoundError, TypeError) as exc:
        raise FileNotFoundError(
            f"{config_path}: EXTENDS names package {package!r}, which is not installed. A "
            "`package.module:resource` EXTENDS reaches configs shipped inside a package; use a "
            "relative path for a config in this tree."
        ) from exc
    if not traversable.is_file():
        raise FileNotFoundError(f"{config_path}: EXTENDS names {resource!r} in {package!r}, which does not exist")
    return _materialize_resource(package, resource)


def default_config_dir(notebook_path: Path, config: ExperimentsConfig | None = None) -> Path:
    """Where an experiment's configs live: the configured directory, else beside the notebook.

    Unset means "beside the notebook", which is what in-tree experiments already do, so a repository that writes no
    table keeps its current behaviour.
    """
    if config is not None and config.config_dir is not None:
        return config.config_dir
    return notebook_path.parent / "configs"


def default_output_dir(notebook_path: Path, config: ExperimentsConfig | None = None) -> Path:
    """Where an experiment's artifacts go: the configured root, else beside the notebook.

    The previous rule pattern-matched one experiment's directory names out of the notebook's path and
    sent those runs to a hard-coded ``/tmp`` root. That is invisible to anyone whose experiment is
    somewhere else, and it made the shared launcher carry one experiment's layout.
    """
    if config is not None and config.output_root is not None:
        return config.output_root
    return notebook_path.parent / "generated_experiments"


@dataclass(frozen=True)
class ExperimentHooks:
    """The experiment-specific callables the shared harness needs, supplied by the experiment.

    The harness previously imported these from one experiment at module level, so importing the SHARED rails required
    that particular experiment to be installed. Passing them in inverts that: the rails depend on a shape rather than on
    a package.

    A second experiment needing a hook the first did not adds a field with a default here. That is a code change in this
    package and not a schema change for anyone's config, which is the trade this shape is chosen for over a registry or
    an entry-point group.
    """

    build_classification_prompt_text: Callable[..., Any] | None = None
    resolve_artifact_output_dir: Callable[..., Any] | None = None
    save_preserved_intervention_artifacts: Callable[..., Any] | None = None
    tensor_fingerprint: Callable[..., Any] | None = None

    def require(self, name: str) -> Callable[..., Any]:
        """The named hook, or an error saying which experiment surface has to supply it."""
        hook = getattr(self, name, None)
        if hook is None:
            raise ValueError(
                f"the shared notebook harness needs the {name!r} hook for this operation and the experiment "
                f"supplied none. Pass ExperimentHooks({name}=...) from the experiment that owns it."
            )
        return hook


__all__ = [
    "bootstrap_experiment_imports",
    "default_config_dir",
    "default_output_dir",
    "ExperimentHooks",
    "ExperimentsConfig",
    "resolve_extends_path",
]
