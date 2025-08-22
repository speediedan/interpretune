"""Auto-columns system for analysis operations."""
from __future__ import annotations
from typing import Dict, Any, Union, Literal, Tuple
from dataclasses import dataclass, field

from interpretune.analysis.ops.base import ColCfg


@dataclass(frozen=True)
class FieldCondition:
    """Represents a condition for a field in a schema."""
    field_name: str
    conditions: Dict[str, Any] = field(default_factory=dict)

    def matches(self, field_config: Dict[str, Any] | ColCfg) -> bool:
        """Check if a field configuration matches this condition."""
        if not isinstance(field_config, (dict, ColCfg)):
            raise TypeError(f"Expected field_config to be dict or ColCfg, got {type(field_config)}")
        for attr_name, expected_value in self.conditions.items():
            if isinstance(field_config, dict):
                actual_value = field_config.get(attr_name)
            elif isinstance(field_config, ColCfg):
                actual_value = getattr(field_config, attr_name, None)

            if actual_value != expected_value:
                return False
        return True


@dataclass(frozen=True)
class AutoColumnCondition:
    """Represents a complete condition set for triggering auto-columns."""
    field_conditions: Tuple[FieldCondition, ...]
    auto_columns: Dict[str, Union[ColCfg, Dict[str, Any]]]
    condition_target: Literal["input_schema", "output_schema"] = "input_schema"

    def __post_init__(self):
        # Ensure field_conditions is a tuple for hashability
        if not isinstance(self.field_conditions, tuple):
            object.__setattr__(self, 'field_conditions', tuple(self.field_conditions))

    def matches_schema(self, input_schema: Dict[str, Any], output_schema: Dict[str, Any] = None) -> bool:  # type: ignore[assignment]
        """Check if schemas match all field conditions."""
        # Select the schema the condition should apply to based on condition_target
        condition_schema = input_schema if self.condition_target == "input_schema" else (output_schema or {})

        for field_condition in self.field_conditions:
            field_config = condition_schema.get(field_condition.field_name)
            if not field_config or not field_condition.matches(field_config):
                return False
        return True


# Auto-columns mapping: conditions that trigger automatic column addition
AUTO_COLUMNS = [
    AutoColumnCondition(
        field_conditions=(
            FieldCondition(
                field_name="input",
                conditions={"connected_obj": "datamodule"}
            ),
        ),
        condition_target="input_schema",
        auto_columns={
            "tokens": {
                "datasets_dtype": "int64",
                "required": False,
                "dyn_dim": 1,
                "array_shape": (None, "batch_size"),
                "sequence_type": False
            },
            "prompts": {
                "datasets_dtype": "string",
                "required": False,
                "non_tensor": True
            }
        }
    )
]


def apply_auto_columns(op_def: dict) -> None:
    """Apply optional auto-columns based on schema conditions."""
    input_schema = op_def.get("input_schema", {})
    output_schema = op_def.get("output_schema", {})

    # Check each condition in AUTO_COLUMNS
    for auto_column_condition in AUTO_COLUMNS:
        if auto_column_condition.matches_schema(input_schema, output_schema):
            # Add auto-columns that don't already exist.
            # We only support auto-adding columns to output_schema at this point.
            target_schema = op_def.setdefault('output_schema', {})

            for col_name, col_cfg in auto_column_condition.auto_columns.items():
                if col_name not in target_schema:
                    # Convert ColCfg to dict if needed
                    if isinstance(col_cfg, ColCfg):
                        target_schema[col_name] = col_cfg.to_dict()
                    else:
                        target_schema[col_name] = col_cfg.copy()
