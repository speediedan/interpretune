from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, ArgsType

from interpretune.cli.core_cli import configure_cli, IT_LIGHTING_SHARED, add_base_args

class ITCLI(LightningCLI):
    """Customize the :class:`~lightning.pytorch.cli.LightningCLI` to ensure the
    :class:`~pytorch_lighting.core.LightningDataModule` and :class:`~lightning.pytorch.core.module.LightningModule`
    use the same Hugging Face model, SuperGLUE task and custom logging tag."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        # See NOTE [Interpretune Dataclass-Oriented Configuration]
        add_base_args(parser)


def cli_main(args: ArgsType = None, instantiate_only: bool = False) -> Optional[ITCLI]:
    instantiate_only, shared_config_files = configure_cli(instantiate_only, IT_LIGHTING_SHARED)
    # currently, share config files for each subcommand but leave separate for future customization
    parser_kwargs = {"default_config_files": shared_config_files} if instantiate_only else \
        {"fit": {"default_config_files": shared_config_files},
         "test": {"default_config_files": shared_config_files},
         "predict": {"default_config_files": shared_config_files},}
    cli = ITCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        parser_kwargs=parser_kwargs,
        args=args,
        run=not instantiate_only,
    )
    if instantiate_only:
        return cli
