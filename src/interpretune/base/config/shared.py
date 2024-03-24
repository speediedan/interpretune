from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import logging
from enum import auto, Enum

import yaml
from transformers import PreTrainedTokenizerBase


log = logging.getLogger(__name__)

################################################################################
# Core Enums
################################################################################

class AutoStrEnum(Enum):
    def _generate_next_value_(name, start, count, last_values) -> str:  # type: ignore
        return name

class CorePhase(AutoStrEnum):
    train = auto()
    validation = auto()
    test = auto()
    predict = auto()

class CoreSteps(AutoStrEnum):
    training_step = auto()
    validation_step = auto()
    test_step = auto()
    predict_step = auto()


################################################################################
# Configuration Serialization
################################################################################

@dataclass(kw_only=True)
class ITSerializableCfg(yaml.YAMLObject):
    ...

def it_cfg_mapping_representer(dumper, data):
    return dumper.represent_mapping('!InterpretuneCfg', data.__dict__)

yaml.add_representer(ITSerializableCfg, it_cfg_mapping_representer)


################################################################################
# Core Shared Configuration for Datamodules and Modules
################################################################################

@dataclass(kw_only=True)
class ITSharedConfig(ITSerializableCfg):
    model_name_or_path: str = ''
    task_name: str = ''
    tokenizer_name: Optional[str] = None
    tokenizer: Optional[PreTrainedTokenizerBase] = None
    os_env_model_auth_key: Optional[str] = None
    tokenizer_id_overrides: Optional[Dict] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    defer_model_init: Optional[bool] = False
