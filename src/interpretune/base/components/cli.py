import warnings
import os
import sys
import numpy as np
import random
import logging
import weakref
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Type
from typing_extensions import override
from functools import reduce

import torch
from transformers import logging as transformers_logging
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace

from interpretune.base.config.shared import ITSharedConf
from interpretune.base.datamodules import ITDataModule
from interpretune.adapters.core import ITModule
from interpretune.base.contract.protocol import InterpretunableType
from interpretune.base.contract.session import ITSession, ITSessionConfig
from interpretune.utils.basic_trainer import BasicTrainer, BasicTrainerCfg
from interpretune.utils.logging import rank_zero_info, rank_zero_warn
from interpretune.utils.import_utils import _DOTENV_AVAILABLE, _LIGHTNING_AVAILABLE
from interpretune.utils.types import ArgsType


max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min

IT_BASE = os.environ.get("IT_BASE", Path(__file__).parent.parent.parent.parent / "it_examples")
IT_CONFIG_BASE = os.environ.get("IT_CONFIG_BASE", IT_BASE / "config")
IT_CONFIG_GLOBAL = os.environ.get("IT_CONFIG_GLOBAL", Path(IT_CONFIG_BASE) / "global")

log = logging.getLogger(__name__)


def _select_seed_randomly(min_seed_value: int = min_seed_value, max_seed_value: int = max_seed_value) -> int:
    return random.randint(min_seed_value, max_seed_value)


class ITSessionMixin:

    def add_base_args(self, parser: ArgumentParser) -> None:
        """Add and link args to the parser."""
        # NOTE [Interpretune Dataclass-Oriented Configuration]
        # For base Interpretune classes, we use configuration dataclasses (e.g. `ITConfig`, `ITDataModuleConfig`) rather
        # than passing numerous arguments to the relevant constructors. Aggregate feedback from other ML framework
        # usage arguably suggests this approach makes instantiation both more flexible and intuitive. (e.g. nested
        # configuration, configuration inheritance, modular `post_init` methods etc.)
        # Also note that making these dataclasses subclass arguments maximizes flexibility of this experimental
        # framework at the expense of modest marginal configuration verbosity (i.e. `init_args` nesting).

        # link our datamodule and module shared configuration
        skey = "session_cfg"
        for attr in ITSharedConf.__dataclass_fields__:
            # parser.link_arguments(f"it_session.session_cfg.init_args.datamodule_cfg.init_args.{attr}",
            #                     f"it_session.session_cfg.init_args.module_cfg.init_args.{attr}")
            parser.link_arguments(f"{skey}.datamodule_cfg.init_args.{attr}", f"{skey}.module_cfg.init_args.{attr}")
        parser.link_arguments(skey, f"it_session.{skey}", apply_on="instantiate")

    def add_arguments_to_parser(self, parser: ArgumentParser) -> None:
        parser.add_class_arguments(ITSession, "it_session", instantiate=True, sub_configs=True)
        parser.add_class_arguments(ITSessionConfig, "session_cfg", instantiate=True, sub_configs=True)
        self.add_base_args(parser)

    def _get(self, config: Namespace, key: str, default: Optional[Any] = None) -> Any:
        """Utility to get a config value which might be inside a subcommand."""
        return config.get(str(getattr(self, 'subcommand', None)), config).get(key, default)


class ITCLI(ITSessionMixin):
    """To maximize compability, the core ITCLI was originally adapted from https://bit.ly/lightning_cli."""
    def __init__(
        self,
        module_class: ITModule = None,
        datamodule_class: ITDataModule = None,
        parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
        args: ArgsType = None,
        seed_everything_default: Union[bool, int] = True,
        run_command: Optional[str] = "test",
        trainer_class: Union[Type[BasicTrainer], Callable[..., BasicTrainer]] = BasicTrainer,
        trainer_cfg: Union[Type[BasicTrainerCfg], Dict[str, Any]] = BasicTrainerCfg,
    ) -> None:
        """fill in
            seed_everything_default: Number for the :func:`~interpretune.base.clieverything`
                seed value. Set to True to automatically choose a seed value.
        Args:
            model_class: model class

        """
        self.seed_everything_default = seed_everything_default
        self.parser_kwargs = parser_kwargs or {}  # type: ignore[var-annotated]  # github.com/python/mypy/issues/6463
        self.module_class = module_class
        self.datamodule_class = datamodule_class
        self.trainer_class = trainer_class
        self._supported_run_commands = getattr(self.trainer_class, "supported_commands", None) or (None, "train",
                                                                                                   "test")
        self.trainer_cfg = trainer_cfg
        self.setup_parser(parser_kwargs)
        self.parse_arguments(self.parser, args)

        self.run_command = run_command
        assert self.run_command in self._supported_run_commands, \
              f"`{self.trainer_class}` only supports the following commands: {self._supported_run_commands}"

        self._set_seed()

        self.before_instantiate_classes()
        self.instantiate_classes()

        if self.run_command:
            getattr(self.trainer, self.run_command)()

    def setup_parser(
        self, main_kwargs: Dict[str, Any]) -> None:
        """Initialize and setup the parser, subcommands, and arguments."""
        self.parser = self.init_parser(**main_kwargs)
        self._add_arguments(self.parser)

    def init_parser(self, **kwargs: Any) -> ArgumentParser:
        """Method that instantiates the argument parser."""
        parser = ArgumentParser(**kwargs)
        parser.add_argument(
            "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
        )
        return parser

    def sanitize_seed(self, seed_in: int | str | float) -> int:
        try:
            seed = int(seed_in)
        except ValueError:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
            rank_zero_info(f"Invalid seed found: {repr(seed_in)}, seed set to {seed}")
        return seed

    def seed_everything(self, seed: Optional[int] = None, workers: bool = False) -> int:
        r""""""
        if seed is None:
            env_seed = os.environ.get("IT_GLOBAL_SEED")
            if env_seed is None:
                seed = _select_seed_randomly(min_seed_value, max_seed_value)
                rank_zero_info(f"No seed found, seed set to {seed}")
            else:
                seed = self.sanitize_seed(env_seed)
        elif not isinstance(seed, int):
            seed = self.sanitize_seed(seed)
        if not (min_seed_value <= seed <= max_seed_value):
            rank_zero_info(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
            seed = _select_seed_randomly(min_seed_value, max_seed_value)

        log.info(f"Seed set to {seed}")
        os.environ["IT_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _add_arguments(self, parser: ArgumentParser) -> None:
        self.add_default_arguments_to_parser(parser)
        self.add_arguments_to_parser(parser)

    def add_default_arguments_to_parser(self, parser: ArgumentParser) -> None:
            """Adds default arguments to the parser."""
            parser.add_argument(
                "--seed_everything",
                type=Union[bool, int, str, float],
                default=self.seed_everything_default,
                help=(
                    "Set to an int to run seed_everything with this value before classes instantiation."
                    "Set to True to use a random seed."
                ),
            )

    def add_base_args(self, parser: ArgumentParser) -> None:
        """Adds core arguments to the parser."""
        super().add_base_args(parser)
        parser.add_class_arguments(self.trainer_class, "trainer", instantiate=True, sub_configs=True,)
        parser.add_dataclass_arguments(self.trainer_cfg, "trainer_cfg", instantiate=True, sub_configs=True)
        parser.link_arguments("it_session", "trainer_cfg.it_session", apply_on="instantiate")
        parser.link_arguments("trainer_cfg", "trainer.trainer_cfg", apply_on="instantiate")

    def parse_arguments(self, parser: ArgumentParser, args: ArgsType) -> None:
        """Parses command line arguments and stores it in ``self.config``."""
        if args is not None and len(sys.argv) > 1:
            rank_zero_info(
                "The args parameter is intended to run from within Python as if it were the command-line. To prevent"
                " mistakes it is not recommended to provide both args and command line arguments, got: "
                f"sys.argv[1:]={sys.argv[1:]}, args={args}."
            )

        # TODO: consider supporting parse_object path in the future and document its (in)availability either way
        # e.g. self.config = parser.parse_object(args)
        self.config = parser.parse_args(args)

    def _set_seed(self) -> None:
        """Sets the seed."""
        config_seed = self.config.get("seed_everything")
        if config_seed is False:
            return
        if config_seed is True:
            # user requested seeding, choose randomly
            config_seed = self.seed_everything(workers=True)
        else:
            config_seed = self.seed_everything(config_seed, workers=True)
        self.config["seed_everything"] = config_seed

    def before_instantiate_classes(self) -> None:
        """Implement to run some code before instantiating the classes."""

    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""
        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self._get(self.config_init.it_session, "datamodule")
        self.module = self._get(self.config_init.it_session, "module")
        self.trainer =  self._get(self.config_init, "trainer")


def env_setup() -> None:
    if _DOTENV_AVAILABLE:
        from dotenv import load_dotenv
        # set WandB API Key if desired, load HF_GATED_PUBLIC_REPO_AUTH_KEY if it exists
        load_dotenv()
    transformers_logging.set_verbosity_error()
    # ignore warnings related tokenizers_parallelism/DataLoader parallelism tradeoff and
    #  expected logging behavior (e.g. we don't depend on jsonargparse config serialization)
    for warnf in [".*does not have many workers*", ".*The number of training samples.*",
                  r"\n.*Unable to serialize.*\n"]:
        warnings.filterwarnings("ignore", warnf)

def enumerate_config_files(folder: Union[Path, str]) -> List:
    if not isinstance(folder, Path):
        folder = Path(folder)
    files = [fp for fp in folder.glob("*.yaml") if fp.is_file()]
    non_yaml_files = [fp for fp in folder.glob("*") if fp.is_file() and not fp.suffix == ".yaml"]
    if non_yaml_files:
        raise ValueError(f"Non-YAML files found in directory: {non_yaml_files}")
    return files

def compose_config(config_files: Iterable[str]) -> List:
    # TODO: consider deprecating `compose_config` for simplicity and subsequently removing this path if not widely used
    args = []
    config_file_paths = []


    def raise_fnf(p: Path):
        raise FileNotFoundError(f"Could not find configuration file path: {p}. Please provide file paths relative to"
                                f" the interpretune config base directory {IT_CONFIG_BASE} or provide a valid"
                                " absolute path.")

    for p in config_files:
        p = Path(p)
        if p.is_absolute():
            if p.exists():
                config_file_paths.append(p)
            else:
                raise_fnf(p)
        else:
            if (p_cfg_base_found := IT_CONFIG_BASE / p).exists():  # try explicit path in the config base
                config_file_paths.append(p_cfg_base_found)
            elif (p_base_found := sorted(IT_BASE.rglob(p.name))) and p_base_found[0].exists():  # more expansive search
                if p_base_found[0].exists():
                    rank_zero_warn(f"Could not find explicit path for config file: `{IT_CONFIG_BASE / p}`. Glob"
                                   f" search within `{IT_BASE}` found `{p_base_found[0]}` which will be used instead.")
                    config_file_paths.append(p_base_found[0])
            else:
                raise_fnf(p)
    for config in config_file_paths:
        args.extend(["--config", str(config)])
    return args

def configure_cli(shared_config_dir: Union[Path, str]) -> Tuple[bool, List]:
    env_setup()
    shared_config_files = enumerate_config_files(shared_config_dir)
    return shared_config_files

def core_cli_main(run_mode: Optional[str | bool] = None , args: ArgsType = None) -> Optional[ITCLI]:
    # note deferred resolution
    default_config_dir = os.environ.get("IT_CONFIG_DEFAULTS", IT_CONFIG_GLOBAL / "defaults" )
    default_config_files = configure_cli(default_config_dir)
    parser_kwargs = {"default_config_files": default_config_files}
    default_run_command = "test"
    run_command = default_run_command if run_mode is True else None if run_mode is False else run_mode
    cli = ITCLI(
        parser_kwargs=parser_kwargs,
        run_command=run_command,
        args=args,
    )
    if not run_command:
        return cli


##########################################################################
# CLI Adapters
##########################################################################

if _LIGHTNING_AVAILABLE:
    from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, ArgsType

    class LightningCLIAdapter:
        core_to_lightning_cli_map = {"data": "it_session.datamodule", "model": "it_session.module"}

        def instantiate_classes(self) -> None:
            super().instantiate_classes()
            # create a convenient alias for the lightning model attribute that uses a standard `module` reference
            self.module = weakref.proxy(self.model)

        def _it_session_cfg(self, config, key) -> Optional[InterpretunableType]:
            try:
                attr_val = reduce(getattr, key.split("."), config)
            except AttributeError:
                attr_val = None
            return attr_val

        def _get(self, config: Namespace, key: str, default: Optional[Any] = None) -> Any:
            """Utility to get a config value which might be inside a subcommand."""
            if target_key := self.core_to_lightning_cli_map.get(key, None):
                return self._it_session_cfg(config.get(str(self.subcommand), config), target_key)
            return config.get(str(self.subcommand), config).get(key, default)


    class LightningITCLI(LightningCLIAdapter, ITSessionMixin, LightningCLI):
        """Customize the :class:`~lightning.pytorch.cli.LightningCLI` to ensure the
        :class:`~pytorch_lighting.core.LightningDataModule` and
        :class:`~lightning.pytorch.core.module.LightningModule` use the same Hugging Face model, SuperGLUE task and
        custom logging tag."""

        @override
        def add_core_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
            """Adds arguments from the Lightning's Trainer to the parser."""
            # We override LightningCLI's `add_core_arguments_to_parser` because model/data are handled by `it_session`
            parser.add_lightning_class_args(self.trainer_class, "trainer")
            trainer_defaults = {"trainer." + k: v for k, v in self.trainer_defaults.items() if k != "callbacks"}
            parser.set_defaults(trainer_defaults)


    def l_cli_main(run_mode: bool = True, args: ArgsType = None) -> Optional[LightningITCLI]:
        # note deferred resolution
        default_config_dir = os.environ.get("IT_CONFIG_DEFAULTS", IT_CONFIG_GLOBAL / "defaults" )
        default_config_files = configure_cli(default_config_dir)
        # currently, share config files for each subcommand but leave separate for future customization
        parser_kwargs = {"default_config_files": default_config_files} if not run_mode else \
            {"fit": {"default_config_files": default_config_files},
            "test": {"default_config_files": default_config_files},
            "predict": {"default_config_files": default_config_files},}
        cli = LightningITCLI(
            datamodule_class=ITDataModule,
            # N.B. we can provide a regular PyTorch module as we're wrapping it as necessary
            model_class=torch.nn.Module,
            subclass_mode_model=True,
            subclass_mode_data=True,
            save_config_kwargs={"overwrite": True},
            parser_kwargs=parser_kwargs,
            args=args,
            run=run_mode,
        )
        if not run_mode:
            return cli

else:
    l_cli_main = object

def _parse_run_option(lightning_cli: bool = False) -> Optional[bool | str]:
    run_mode = None
    if lightning_cli:
        sys.argv.pop(sys.argv.index("--lightning_cli"))
        # LightningCLI offers a boolean `run` option that is by default `True`, we offer the `--no_run` flag to
        # control setting it to `False` which returns the CLI with parsed/instantiated config.
        no_run = False
        if no_run := "--no_run" in sys.argv[1:]:
            sys.argv.pop(sys.argv.index("--no_run"))
        return not no_run
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--run_command"):
            run_mode = sys.argv[i + 1] if "=" not in arg else arg.split("=")[1]
            sys.argv.pop(i)
            if "=" not in arg:
                sys.argv.pop(i)
    # core CLI's string `run_mode` controls both the command to run and if not provided, invokes parse/instantiate only
    return run_mode

def bootstrap_cli() -> Callable:
    # TODO: consider adding an env var option to control CLI selection
    # dispatch the relevant CLI, right now only `--lightning_cli` is supported beyond the default core CLI.
    # TODO: note in the run_experiment.py documentation that we provide the --no_run flag to allow configuring the
    #       Lightning CLI to not run subcommands and instead return the cli with parsed/instantiated config.
    # TODO: for the core CLI only, we provide the --run_command flag option to to control which command to run,
    #       LightningCLI uses the normal LightningCLI format (passing the command as a separate arg without a flag,
    #       e.g. `python run_experiment.py fit --config some_path/to/some_config.yaml`).
    lightning_cli = "--lightning_cli" in sys.argv[1:]
    run_mode = None
    if lightning_cli:
        cli_main = l_cli_main
        run_mode = _parse_run_option(lightning_cli=True)
    else:
        cli_main = core_cli_main
        run_mode = _parse_run_option()
    return cli_main(run_mode=run_mode)
