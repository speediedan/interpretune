from typing import Optional, Any
from typing_extensions import override
from functools import reduce
import os
import weakref

from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, ArgsType
import torch
from jsonargparse import Namespace

from interpretune.base.cli.core_cli import configure_cli, IT_CONFIG_GLOBAL, ITSessionMixin
from interpretune.base.datamodules import ITDataModule
from interpretune.base.contract.protocol import InterpretunableType


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
    :class:`~pytorch_lighting.core.LightningDataModule` and :class:`~lightning.pytorch.core.module.LightningModule`
    use the same Hugging Face model, SuperGLUE task and custom logging tag."""

    @override
    def add_core_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Adds arguments from the Lightning's Trainer to the parser."""
        # We override LightningCLI's `add_core_arguments_to_parser` because model and data are handled by `it_session`
        parser.add_lightning_class_args(self.trainer_class, "trainer")
        trainer_defaults = {"trainer." + k: v for k, v in self.trainer_defaults.items() if k != "callbacks"}
        parser.set_defaults(trainer_defaults)


def l_cli_main(args: ArgsType = None, run_command: Optional[str] = None) -> Optional[LightningITCLI]:
    shared_config_dir = os.environ.get("IT_LIGHTNING_SHARED", IT_CONFIG_GLOBAL / "lightning" )  # deferred resolution
    shared_config_files = configure_cli(shared_config_dir)
    # currently, share config files for each subcommand but leave separate for future customization
    parser_kwargs = {"default_config_files": shared_config_files} if not run_command else \
        {"fit": {"default_config_files": shared_config_files},
         "test": {"default_config_files": shared_config_files},
         "predict": {"default_config_files": shared_config_files},}
    cli = LightningITCLI(
        datamodule_class=ITDataModule,
        # N.B. we can provide a regular PyTorch module as we're wrapping it as necessary
        model_class=torch.nn.Module,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        parser_kwargs=parser_kwargs,
        args=args,
        run=bool(run_command),
    )
    if not run_command:
        return cli
