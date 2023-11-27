from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, ArgsType

from interpretune.base.config_classes import ITConfig, ITDataModuleConfig
from interpretune.utils.cli import enumerate_config_files, env_setup, IT_LIGHTING_SHARED

class ITCLI(LightningCLI):
    """Customize the :class:`~lightning.pytorch.cli.LightningCLI` to ensure the
    :class:`~pytorch_lighting.core.LightningDataModule` and :class:`~lightning.pytorch.core.module.LightningModule`
    use the same Hugging Face model, SuperGLUE task and custom logging tag."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_class_arguments(ITDataModuleConfig, "itdm_cfg")
        parser.add_class_arguments(ITConfig, "it_cfg")
        parser.link_arguments("itdm_cfg", "data.init_args.itdm_cfg")
        parser.link_arguments("it_cfg", "model.init_args.it_cfg")
        parser.link_arguments("trainer.logger.init_args.name", "it_cfg.experiment_tag")
        parser.link_arguments("itdm_cfg.model_name_or_path", "it_cfg.model_name_or_path")
        parser.link_arguments("itdm_cfg.tokenizer_id_overrides", "it_cfg.tokenizer_id_overrides")
        parser.link_arguments("itdm_cfg.os_env_model_auth_key", "it_cfg.os_env_model_auth_key")
        parser.link_arguments("itdm_cfg.task_name", "it_cfg.task_name")


def cli_main(args: ArgsType = None, instantiate_only: bool = False) -> Optional[ITCLI]:
    env_setup()
    shared_config_files = enumerate_config_files(IT_LIGHTING_SHARED)
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
