"""Datasets library extensions for Interpretune."""
from datasets.formatting import _register_formatter

from .formatters import ITAnalysisFormatter

# Register the custom formatter
_register_formatter(ITAnalysisFormatter, "interpretune", aliases=["it", "itanalysis"])
