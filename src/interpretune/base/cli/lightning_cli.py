from typing import Optional
from typing_extensions import override

from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, ArgsType
import torch

from interpretune.base.cli.core_cli import configure_cli, IT_LIGHTING_SHARED, ITSessionMixin
from interpretune.base.datamodules import ITDataModule


class ITCLI(ITSessionMixin, LightningCLI):
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
