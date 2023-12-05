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
from typing import Any, Dict, List, Optional, Union, Iterable, Tuple, Callable

import torch
from transformers import logging as transformers_logging

from interpretune.utils.types import ArgsType
from interpretune.base.config_classes import ITConfig, ITDataModuleConfig, ITSharedConfig
from interpretune.base.it_datamodule import ITDataModule
from interpretune.base.it_module import ITModule, BaseITModule
from interpretune.utils.logging import rank_zero_info, rank_zero_warn
from interpretune.utils.import_utils import _DOTENV_AVAILABLE
from interpretune.base.call import _run


from jsonargparse import (
    ActionConfigFile,
    ArgumentParser,
    Namespace,
)

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min

IT_BASE = os.environ.get("IT_BASE", Path(__file__).parent.parent.parent / "it_examples")
IT_CONFIG_BASE = os.environ.get("IT_BASE", IT_BASE / "config")
IT_CONFIG_GLOBAL = os.environ.get("IT_CONFIG_GLOBAL", IT_CONFIG_BASE / "global")
IT_CORE_SHARED = os.environ.get("IT_CORE_SHARED", IT_CONFIG_GLOBAL / "core" )
IT_LIGHTING_SHARED = os.environ.get("IT_LIGHTING_SHARED", IT_CONFIG_GLOBAL / "lightning" )

log = logging.getLogger(__name__)

def _select_seed_randomly(min_seed_value: int = min_seed_value, max_seed_value: int = max_seed_value) -> int:
    return random.randint(min_seed_value, max_seed_value)

def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    r"""Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition,
    sets the following environment variables:

    - ``PL_GLOBAL_SEED``: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - ``PL_SEED_WORKERS``: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
            If ``None``, will read seed from ``PL_GLOBAL_SEED`` env variable
            or select it randomly.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~lightning.fabric.utilities.seed.pl_worker_init_function`.

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

def add_base_args(parser: ArgumentParser) -> None:
    """Add and link args to the parser."""
    # NOTE [Interpretune Dataclass-Oriented Configuration]
    # For base Interpretune classes, we use configuration dataclasses (e.g. `ITConfig`, `ITDataModuleConfig`) rather
    # than passing numerous arguments to the relevant constructors. Aggregate feedback from other ML framework
    # usage arguably suggests this approach makes instantiation both more flexible and intuitive. (e.g. nested
    # configuration, configuration inheritance, modular `post_init` methods etc.)
    # Also note that making these dataclasses subclass arguments maximizes flexibility of this experimental
    # framework at the expense of modest marginal configuration verbosity (i.e. `init_args` nesting).
    parser.add_subclass_arguments(ITDataModuleConfig, "itdm_cfg", fail_untyped=False, required=True)
    parser.add_subclass_arguments(ITConfig, "it_cfg", fail_untyped=False, required=True)
    parser.link_arguments("itdm_cfg", "data.init_args.itdm_cfg")
    parser.link_arguments("it_cfg", "model.init_args.it_cfg")
    # link our datamodule and module shared configuration
    for attr in ITSharedConfig.__dataclass_fields__:
        parser.link_arguments(f"itdm_cfg.init_args.{attr}", f"it_cfg.init_args.{attr}")

def bootstrap_cli() -> Callable:
    # TODO: consider adding an env var option to control CLI selection
    if "--lightning_cli" in sys.argv[1:]:
        lightning_cli = True
        sys.argv.remove("--lightning_cli")
    else:
        lightning_cli = False
    if lightning_cli:
        from interpretune.cli.lightning_cli import cli_main
    else:
        from interpretune.cli.core_cli import cli_main  # type: ignore[no-redef]
    return cli_main()


class ITCLI:
    """Customize the :class:`~lightning.pytorch.cli.LightningCLI` to ensure the
    :class:`~pytorch_lighting.core.LightningDataModule` and :class:`~lightning.pytorch.core.module.LightningModule`
    use the same Hugging Face model, SuperGLUE task and custom logging tag."""
    def __init__(
        self,
        model_class: ITModule = None,
        datamodule_class: ITDataModule = None,
        parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
        args: ArgsType = None,
        seed_everything_default: Union[bool, int] = True,
        instantiate_only: bool = False,
    ) -> None:
        """fill in
            seed_everything_default: Number for the :func:`~interpretune.clieverything`
                seed value. Set to True to automatically choose a seed value.
        Args:
            model_class: model class

        """
        self.seed_everything_default = seed_everything_default
        self.instantiate_only = instantiate_only
        self.parser_kwargs = parser_kwargs or {}  # type: ignore[var-annotated]  # github.com/python/mypy/issues/6463
        self.model_class = model_class
        self.datamodule_class = datamodule_class
        self.setup_parser(parser_kwargs)
        self.parse_arguments(self.parser, args)

        self._set_seed()

        self.before_instantiate_classes()
        self.instantiate_classes()
        if not self.instantiate_only:
            self._run_core_flow()


    def setup_parser(
        self, main_kwargs: Dict[str, Any]) -> None:
        """Initialize and setup the parser, subcommands, and arguments."""
        self.parser = self.init_parser(**main_kwargs)
        self._add_arguments(self.parser)


    def init_parser(self, **kwargs: Any) -> ArgumentParser:
        """Method that instantiates the argument parser."""
        #kwargs.setdefault("dump_header", [f"lightning.pytorch=={pl.__version__}"])
        parser = ArgumentParser(**kwargs)
        parser.add_argument(
            "-c", "--config", action=ActionConfigFile, help="Path to a configuration file in json or yaml format."
        )
        return parser

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

    def add_arguments_to_parser(self, parser: ArgumentParser) -> None:
        parser.add_subclass_arguments(ITDataModule, "data", fail_untyped=False, required=True)
        parser.add_subclass_arguments(BaseITModule, "model", fail_untyped=False, required=True)
        add_base_args(parser)

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
            config_seed = seed_everything(workers=True)
        else:
            config_seed = seed_everything(config_seed, workers=True)
        self.config["seed_everything"] = config_seed

    def before_instantiate_classes(self) -> None:
        """Implement to run some code before instantiating the classes."""

    def instantiate_classes(self) -> None:
        """Instantiates the classes and sets their attributes."""
        self.config_init = self.parser.instantiate_classes(self.config)
        self.datamodule = self.config_init.get("data", None)
        self.model = self.config_init.get("model", None)

    def _run_core_flow(self) -> None:
        _run(model=self.model, datamodule=self.datamodule)

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


def _check_cli_mode() -> bool:
    if "--instantiate_only" in sys.argv[1:]:
        insantiate_only = True
        sys.argv.remove("--instantiate_only")
    else:
        insantiate_only = False
    return insantiate_only

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

def configure_cli(instantiate_only: bool, shared_config_dir: Union[Path, str]) -> Tuple[bool, List]:
    env_setup()
    # configuring `cli_main` with `instantiate_only` short-circuits command-line `--instantiate_only` check
    instantiate_only |=_check_cli_mode()
    shared_config_files = enumerate_config_files(shared_config_dir)
    return instantiate_only, shared_config_files

def cli_main(args: ArgsType = None, instantiate_only: bool = False) -> Optional[ITCLI]:
    instantiate_only, shared_config_files = configure_cli(instantiate_only, IT_CORE_SHARED)
    parser_kwargs = {"default_config_files": shared_config_files}
    cli = ITCLI(
        parser_kwargs=parser_kwargs,
        instantiate_only=instantiate_only,
        args=args,
    )
    if instantiate_only:
        return cli
