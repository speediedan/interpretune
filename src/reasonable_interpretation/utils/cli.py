from typing import Any, List, Union, Optional, Dict
import warnings
from pathlib import Path

import lightning.pytorch as pl
from lightning_utilities.core.imports import module_available
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser, ArgsType
from fts_examples import _HF_AVAILABLE, _SP_AVAILABLE

from reasonable_interpretation.base.config_classes import RIConfig, RIDataModuleConfig

if _HF_AVAILABLE:
    from transformers import logging as transformers_logging

_DOTENV_AVAILABLE = module_available("dotenv")

def env_setup() -> None:
    if _DOTENV_AVAILABLE:
        from dotenv import load_dotenv
        # set WandB API Key if desired, load LLAMA2_AUTH_KEY if it exists
        load_dotenv()
    transformers_logging.set_verbosity_error()
    # ignore warnings related tokenizers_parallelism/DataLoader parallelism tradeoff and
    #  expected logging behavior
    for warnf in [".*does not have many workers*", ".*The number of training samples.*"]:
        warnings.filterwarnings("ignore", warnf)

def _import_class(class_path: str) -> Any:
    class_module, class_name = class_path.rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    return getattr(module, class_name)

class RICLI(LightningCLI):
    """Customize the :class:`~lightning.pytorch.cli.LightningCLI` to ensure the
    :class:`~pytorch_lighting.core.LightningDataModule` and :class:`~lightning.pytorch.core.module.LightningModule`
    use the same Hugging Face model, SuperGLUE task and custom logging tag."""

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_class_arguments(RIDataModuleConfig, "ridm_cfg")
        parser.add_class_arguments(RIConfig, "ri_cfg")
        parser.link_arguments("ridm_cfg", "data.init_args.ridm_cfg")
        parser.link_arguments("ri_cfg", "model.init_args.ri_cfg")
        parser.link_arguments("trainer.logger.init_args.name", "ri_cfg.experiment_tag")
        parser.link_arguments("ridm_cfg.model_name_or_path", "ri_cfg.model_name_or_path")
        parser.link_arguments("ridm_cfg.tokenizer_id_overrides", "ri_cfg.tokenizer_id_overrides")
        parser.link_arguments("ridm_cfg.os_env_model_auth_key", "ri_cfg.os_env_model_auth_key")
        parser.link_arguments("ridm_cfg.task_name", "ri_cfg.task_name")


def enumerate_config_files(folder: Union[Path, str]) -> List:
    if not isinstance(folder, Path):
        folder = Path(folder)
    files = [fp for fp in folder.glob("*.yaml") if fp.is_file()]
    non_yaml_files = [fp for fp in folder.glob("*") if fp.is_file() and not fp.suffix == ".yaml"]
    if non_yaml_files:
        raise ValueError(f"Non-YAML files found in directory: {non_yaml_files}")
    return files

def compose_config(config_files: List[str]) -> Dict:
    args = []
    config_file_paths = []
    # TODO: make the ri_base config directory configurable
    ri_base = Path(__file__).parent.parent
    config_base_path = ri_base / "config"

    def raise_fnf(p: Path):
        raise FileNotFoundError(f"Could not find configuration file path: {p}. Please provide file paths relative to "
                                f"the ri config base directory {config_base_path} or provide a valid absolute path.")

    for p in config_files:
        p = Path(p)
        if p.is_absolute():
            if p.exists():
                config_file_paths.append(p)
            else:
                raise_fnf(p)
        else:
            if (p_cfg_base_found := config_base_path / p).exists():  # try explicit path in the config base
                config_file_paths.append(p_cfg_base_found)
            elif (p_base_found := sorted(ri_base.rglob(p.name))) and p_base_found[0].exists():  # more expansive search
                if p_base_found[0].exists():
                    config_file_paths.append(p_base_found[0])
            else:
                raise_fnf(p)
    for config in config_file_paths:
        args.extend(["--config", str(config)])
    instantiate_only = True
    return {"args": args, "instantiate_only": instantiate_only}

def cli_main(args: ArgsType = None, instantiate_only: bool = False) -> Optional[RICLI]:
    env_setup()
    if not _HF_AVAILABLE:  # pragma: no cover
        print("Running the fts_superglue example requires the `transformers` and `datasets` packages from Hugging Face")
    if not _SP_AVAILABLE:
        print("Note using the default model in this fts_superglue example requires the `sentencepiece` package.")
    if not all([_HF_AVAILABLE, _SP_AVAILABLE]):
        return
    # every configuration of this example depends upon a shared set of defaults.
    # default_config_filenames = ["core_defaults.yaml", "prompt_defaults.yaml", "optimizer_defaults.yaml",
    #                             "lr_scheduler_defaults.yaml", "quantization_defaults.yaml", "tokenizer_defaults.yaml"]
    shared_config_path = Path(__file__).parent.parent / "config" / "shared"
    shared_config_files = enumerate_config_files(shared_config_path)
    # currently, share config files for each subcommand but leave separate for future customization
    parser_kwargs = {"default_config_files": shared_config_files} if instantiate_only else \
        {"fit": {"default_config_files": shared_config_files},
         "test": {"default_config_files": shared_config_files},
         "predict": {"default_config_files": shared_config_files},}
    cli = RICLI(
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
