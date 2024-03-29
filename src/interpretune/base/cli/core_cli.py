# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
import os
import sys
import numpy as np
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterable, Tuple, Callable, Type

import torch
from transformers import logging as transformers_logging

from interpretune.base.config.shared import ITSharedConfig
from interpretune.base.datamodules import ITDataModule
from interpretune.base.modules import ITModule
from interpretune.base.contract.session import ITSession, ITSessionConfig
from interpretune.utils.basic_trainer import BasicTrainer, BasicTrainerCfg
from interpretune.utils.logging import rank_zero_info, rank_zero_warn
from interpretune.utils.import_utils import _DOTENV_AVAILABLE
from interpretune.utils.types import ArgsType

from jsonargparse import (
    ActionConfigFile,
    ArgumentParser,
    Namespace,
)

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min

IT_BASE = os.environ.get("IT_BASE", Path(__file__).parent.parent.parent.parent / "it_examples")
IT_CONFIG_BASE = os.environ.get("IT_BASE", IT_BASE / "config")
IT_CONFIG_GLOBAL = os.environ.get("IT_CONFIG_GLOBAL", IT_CONFIG_BASE / "global")

log = logging.getLogger(__name__)

def _select_seed_randomly(min_seed_value: int = min_seed_value, max_seed_value: int = max_seed_value) -> int:
    return random.randint(min_seed_value, max_seed_value)


def bootstrap_cli() -> Callable:
    # TODO: consider adding an env var option to control CLI selection
    if "--lightning_cli" in sys.argv[1:]:
        lightning_cli = True
        sys.argv.remove("--lightning_cli")
    else:
        lightning_cli = False
    if lightning_cli:
        from interpretune.base.cli.lightning_cli import l_cli_main as cli_main
    else:
        from interpretune.base.cli.core_cli import core_cli_main as cli_main  # type: ignore[no-redef]
    return cli_main()


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
        for attr in ITSharedConfig.__dataclass_fields__:
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
    """To maximize Lightning compability, the core ITCLI was originally adapted from
    https://bit.ly/lightning_cli."""
    def __init__(
        self,
        module_class: ITModule = None,
        datamodule_class: ITDataModule = None,
        parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
        args: ArgsType = None,
        seed_everything_default: Union[bool, int] = True,
        run_command: str = "test",
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

    def seed_everything(self, seed: Optional[int] = None, workers: bool = False) -> int:
        r"""
        """
        if seed is None:
            env_seed = os.environ.get("IT_GLOBAL_SEED")
            if env_seed is None:
                seed = _select_seed_randomly(min_seed_value, max_seed_value)
                rank_zero_info(f"No seed found, seed set to {seed}")
            else:
                try:
                    seed = int(env_seed)
                except ValueError:
                    seed = _select_seed_randomly(min_seed_value, max_seed_value)
                    rank_zero_info(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
        elif not isinstance(seed, int):
            seed = int(seed)

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
                type=Union[bool, int],
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
                "LightningCLI's args parameter is intended to run from within Python like if it were from the command "
                "line. To prevent mistakes it is not recommended to provide both args and command line arguments, got: "
                f"sys.argv[1:]={sys.argv[1:]}, args={args}."
            )
        if isinstance(args, (dict, Namespace)):
            self.config = parser.parse_object(args)
        else:
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
        # set WandB API Key if desired, load LLAMA2_AUTH_KEY if it exists
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
                    rank_zero_warn(f"Could not find explicit path for config file: `{IT_CONFIG_BASE / p}`. Glob "
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

def core_cli_main(args: ArgsType = None, run_command: Optional[str] = None) -> Optional[ITCLI]:
    shared_config_dir = os.environ.get("IT_CORE_SHARED", IT_CONFIG_GLOBAL / "core" )  # deferred resolution
    shared_config_files = configure_cli(shared_config_dir)
    parser_kwargs = {"default_config_files": shared_config_files}
    cli = ITCLI(
        parser_kwargs=parser_kwargs,
        run_command=run_command,
        args=args,
    )
    if not run_command:
        return cli
