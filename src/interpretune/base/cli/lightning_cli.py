from typing import Optional, Any
from typing_extensions import override
from functools import reduce

from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, ArgsType
import torch
from jsonargparse import Namespace

from interpretune.base.cli.core_cli import configure_cli, IT_LIGHTING_SHARED, ITSessionMixin
from interpretune.base.datamodules import ITDataModule
from interpretune.base.contract.protocol import InterpretunableType


class LightningCLIAdapter:
    core_to_lightning_cli_map = {"data": "it_session.datamodule", "model": "it_session.module"}

    def _it_session(self, config, key) -> Optional[InterpretunableType]:
        try:
            attr_val = reduce(getattr, key.split("."), config)
        except AttributeError:
            attr_val = None
        return attr_val

    def _get(self, config: Namespace, key: str, default: Optional[Any] = None) -> Any:
        """Utility to get a config value which might be inside a subcommand."""
        if target_key := self.core_to_lightning_cli_map.get(key, None):
            return self._it_session(config.get(str(self.subcommand), config), target_key)
        return config.get(str(self.subcommand), config).get(key, default)


class ITCLI(LightningCLIAdapter, ITSessionMixin, LightningCLI):
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


def cli_main(args: ArgsType = None, instantiate_only: bool = False) -> Optional[ITCLI]:
    instantiate_only, shared_config_files = configure_cli(instantiate_only, IT_LIGHTING_SHARED)
    # currently, share config files for each subcommand but leave separate for future customization
    parser_kwargs = {"default_config_files": shared_config_files} if instantiate_only else \
        {"fit": {"default_config_files": shared_config_files},
         "test": {"default_config_files": shared_config_files},
         "predict": {"default_config_files": shared_config_files},}
    cli = ITCLI(
        datamodule_class=ITDataModule,
        # N.B. we can provide a regular PyTorch module as we're wrapping it as necessary
        model_class=torch.nn.Module,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        parser_kwargs=parser_kwargs,
        args=args,
        run=not instantiate_only,
    )
    if instantiate_only:
        return cli
