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
"""Parameter naming transformation strategies (ModelView) for fine-tuning schedules.

A ModelView encapsulates a specific way of viewing/naming model parameters, enabling transformation between the view's
naming convention and canonical PyTorch parameter names. This abstraction allows FTS schedules to use various naming
conventions (e.g., TransformerLens-style, SAELens-style, custom architectures) while maintaining clean separation of
concerns.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from interpretune.utils import rank_zero_debug

if TYPE_CHECKING:
    from finetuning_scheduler.strategy_adapters.base import StrategyAdapter

__all__ = ["ModelView", "CanonicalModelView"]


class ModelView(ABC):
    """Abstract base for parameter naming transformation strategies.

    A ModelView encapsulates a specific way of viewing/naming model parameters,
    enabling transformation between the view's naming convention and canonical
    PyTorch parameter names.

    Examples:
        - TLNamesModelView: TransformerLens-style names (blocks.0.attn.W_Q)
        - CanonicalView: Standard PyTorch names (model.transformer.h.0.attn.q.weight)
        - Future: SAELensView, CustomArchitectureView, etc.

    Args:
        adapter: Parent StrategyAdapter instance for context access
    """

    def __init__(self, adapter: "StrategyAdapter"):
        self.adapter = adapter
        self.pl_module = adapter.pl_module
        self.trainer = adapter.trainer

    @abstractmethod
    def build_param_mapping(self) -> None:
        """Build bidirectional parameter name mappings.

        Called once during FTS initialization to establish mappings between view-specific names and canonical parameter
        names.
        """
        pass

    @abstractmethod
    def transform_to_canonical(self, param_names: list[str], inspect_only: bool = False) -> list[str]:
        """Transform view-specific parameter names to canonical names.

        Used by FTS to convert schedule params to optimizer params.

        Args:
            param_names: Parameters in view-specific naming
            inspect_only: If True, only validate mapping without transforming

        Returns:
            Canonical parameter names for optimizer
        """
        pass

    @abstractmethod
    def transform_from_canonical(self, param_names: list[str]) -> list[str]:
        """Transform canonical parameter names to view-specific names.

        Used for logging/reporting optimizer params in view naming.

        Args:
            param_names: Canonical parameter names from optimizer

        Returns:
            View-specific parameter names
        """
        pass

    @abstractmethod
    def get_named_params(self) -> dict[str, torch.Tensor]:
        """Get model parameters using view-specific naming.

        Returns:
            Dict mapping view-specific names to parameter tensors
        """
        pass

    @abstractmethod
    def gen_schedule(self, dump_loc: str | os.PathLike) -> os.PathLike | None:
        """Generate implicit schedule using view-specific naming.

        Args:
            dump_loc: Directory to write schedule file

        Returns:
            Path to generated schedule YAML
        """
        pass

    @abstractmethod
    def validate_schedule(self) -> tuple[int, int]:
        """Validate schedule with optional view-specific diagnostics.

        Delegates to base StrategyAdapter implementation.
        Subclasses can override to add view-specific logging/checks.

        Returns:
            Tuple of (max_depth, max_epoch_watermark)
        """
        pass


class CanonicalModelView(ModelView):
    """Default canonical parameter naming (no transformation).

    This view represents the identity transformation - parameters are
    already in canonical PyTorch naming and no transformation is needed.
    Used as the default when no specific view is requested.
    """

    def __init__(self, adapter: "StrategyAdapter", **kwargs):
        """Initialize canonical model view.

        Args:
            adapter: The strategy adapter instance
            **kwargs: Future configuration options (currently unused)
        """
        super().__init__(adapter)
        if kwargs:
            rank_zero_debug(f"CanonicalModelView received unused config: {kwargs}")

    def build_param_mapping(self) -> None:
        """No mapping needed for canonical naming."""
        pass

    def transform_to_canonical(self, param_names: list[str], inspect_only: bool = False) -> list[str]:
        """Identity transformation - params already canonical."""
        return param_names

    def transform_from_canonical(self, param_names: list[str]) -> list[str]:
        """Identity transformation - params already canonical."""
        return param_names

    def get_named_params(self) -> dict[str, torch.Tensor]:
        """Get canonical parameter names."""
        return dict(self.pl_module.named_parameters())

    def gen_schedule(self, dump_loc: str | os.PathLike) -> os.PathLike | None:
        """Generate schedule with canonical naming.

        Delegates to base StrategyAdapter implementation via the adapter reference.
        """
        # Call the base StrategyAdapter.gen_ft_schedule() method through the adapter
        from finetuning_scheduler.strategy_adapters.base import StrategyAdapter

        return StrategyAdapter.gen_ft_schedule(self.adapter, dump_loc)

    def validate_schedule(self) -> tuple[int, int]:
        """Validate schedule with canonical naming.

        Delegates to base StrategyAdapter implementation via the adapter reference.
        """
        # Call the base StrategyAdapter.validate_ft_sched() method through the adapter
        from finetuning_scheduler.strategy_adapters.base import StrategyAdapter

        return StrategyAdapter.validate_ft_sched(self.adapter)
