"""Orchestrator for analysis injection framework.

This module provides a high-level interface to:
1. Load configuration from YAML
2. Patch circuit_tracer files with hooks
3. Register analysis functions
4. Enable/disable hook execution
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

import torch
import yaml
from tabulate import tabulate

from .analysis_hook_patcher import (
    HOOK_REGISTRY,
    install_patched_modules_with_references,
    patch_target_package_files,
    verify_patching,
)
from .config_parser import get_enabled_points, load_config, merge_config_dict, parse_config_dict

try:
    from IPython.display import display, HTML
except ImportError:
    display = None
    HTML = None


@dataclass
class VarAnnotate:
    """Specification for a variable to annotate in an analysis injection context.

    - `var_ref` the variable reference key in the `data` dictionary (e.g., "x", "y")
    - `var_value` the raw inspected value
    - `annotation` stores optional user annotation
    - `output` property returns a formatted representation
    """

    var_ref: str
    var_value: Any = ""
    annotation: str = ""
    # Optional per-instance kwargs forwarded to `format_tensor` when the
    # inspected value is a tensor. Example: {'float_precision': 3, 'max_rows': 8}
    format_tensor_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.var_ref:
            raise ValueError("var_ref cannot be empty")

    @property
    def output(self) -> str:
        if isinstance(self.var_value, torch.Tensor):
            return format_tensor(self.var_value, **(self.format_tensor_kwargs or {}))
        if isinstance(self.var_value, dict) or (
            isinstance(self.var_value, (tuple, list)) and all(isinstance(d, dict) for d in self.var_value)
        ):
            return format_dict_sequence(self.var_value)
        return str(self.var_value)

    @output.setter
    def output(self, value: Any) -> None:
        self.var_value = value

    # Note: formatting at display time will combine per-instance
    # `format_tensor_kwargs` with any caller-provided overrides.  The
    # orchestrator display functions (e.g. `get_output`) perform that merge
    # and call `_format_value_for_display` with the merged kwargs. We keep
    # only the simple `output` property for backward compatibility.


# Global analysis logger holder
_ANALYSIS_LOGGER = None
ANALYSIS_FUNCTIONS: Dict[str, Callable] = {}

# Global data collection dictionary
ANALYSIS_DATA: OrderedDict[str, dict | None] = OrderedDict()


def get_analysis_data() -> OrderedDict[str, dict | None]:
    """Get the collected analysis data.

    Returns:
        OrderedDict mapping analysis point IDs to collected data dictionaries (or None if not executed).
    """
    return ANALYSIS_DATA.copy()


def clear_analysis_data():
    """Clear all collected analysis data."""
    global ANALYSIS_DATA
    ANALYSIS_DATA.clear()


def _load_analysis_functions_from_module(module_path: Path) -> Dict[str, Any]:
    """Load analysis functions from a Python module.

    The module must define an ``AP_FUNCTIONS`` mapping. The returned dictionary
    is a shallow copy so callers can mutate it without affecting the source.
    """

    if not module_path.exists():
        raise FileNotFoundError(f"Analysis points module not found at {module_path}")

    module_name = f"analysis_points_{module_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load analysis points module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "AP_FUNCTIONS"):
        raise AttributeError(f"Module {module_path} does not define AP_FUNCTIONS; unable to auto-load analysis points")

    functions = getattr(module, "AP_FUNCTIONS")
    if not isinstance(functions, Mapping):
        raise TypeError(f"AP_FUNCTIONS in {module_path} must be a mapping; got {type(functions)!r}")

    return {key: value for key, value in dict(functions).items()}


def init_analysis_logger(
    log_dir: str | None = None,
    log_to_console: bool = False,
    log_to_file: bool = True,
    log_prefix: str = "attribution_flow_analysis",
) -> logging.Logger:
    """Initialize or reconfigure the global analysis logger.

    This replaces the earlier logger implementation which lived in
    analysis_functions.py so that notebook-local analysis point modules can
    import a single, stable `get_analysis_logger()` from the orchestrator.

    Note: By default, console logging is disabled to avoid duplicate messages
    in notebook output (the original module's logger already logs to console).
    Analysis output is written to a file instead.

    Args:
        log_dir: Directory for log files. If None, uses system temp directory.
        log_to_console: Whether to log to console.
        log_to_file: Whether to log to file.
        log_prefix: Prefix for log file name.
    """
    global _ANALYSIS_LOGGER

    # Use system temp directory if not specified (platform-agnostic)
    if log_dir is None:
        log_dir = tempfile.gettempdir()

    logger = logging.getLogger("analysis_injection")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False  # Prevent double logging if root logger has handlers

    # Store the log file path for later reference
    log_file_path = None

    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)

    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = Path(log_dir) / f"{log_prefix}_{timestamp}.log"
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)

        # Store the log file path as an attribute on the logger
        setattr(logger, "log_file_path", log_file_path)

    _ANALYSIS_LOGGER = logger
    return _ANALYSIS_LOGGER


def get_analysis_logger() -> logging.Logger:
    """Return the global analysis logger, initializing with defaults if needed."""
    global _ANALYSIS_LOGGER
    if _ANALYSIS_LOGGER is None:
        _ANALYSIS_LOGGER = init_analysis_logger()
    return _ANALYSIS_LOGGER


def _format_data_for_logging(data: dict) -> str:
    """Format a data dict for logging, with basic tensor handling."""
    parts = []
    for k, v in data.items():
        try:
            if isinstance(v, torch.Tensor):
                shape = tuple(v.shape)
                if v.numel() <= 10:
                    vals = v.detach().cpu().tolist()
                    parts.append(f"  {k} shape: {shape}, values: {vals}")
                else:
                    parts.append(f"  {k} shape: {shape}")
                continue
        except Exception as e:
            _get_pkg_logger().warning(f"Failed to format data for key '{k}': {e}")
            pass

        if isinstance(v, dict):
            parts.append(f"  {k}:")
            for kk, vv in v.items():
                parts.append(f"    {kk}: {vv}")
        else:
            parts.append(f"  {k}: {v}")

    return "\n".join(parts)


def format_tensor(
    tensor,
    edgeitems: int = 5,
    max_rows: int = 10,
    max_cols: int = 10,
    float_precision: int = 6,
) -> str:
    """
    Format a tensor for display:
    - Move to CPU, preserve dtype for integers
    - For 1D: show first/last edgeitems, ellipsis if longer
    - For 2D: show up to max_rows, each row on separate line
    - For >2D: report unsupported
    """
    if not isinstance(tensor, torch.Tensor):
        return str(tensor)

    t = tensor.detach().cpu()
    shape = t.shape

    # Check if tensor is integer type
    is_integer = t.dtype in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    )

    if len(shape) == 0:
        if is_integer:
            return str(int(t.item()))
        else:
            return str(round(float(t.item()), float_precision))

    if len(shape) == 1:
        n = shape[0]
        if n > max_cols:
            if is_integer:
                vals = [int(x) for x in t[:edgeitems]] + ["..."] + [int(x) for x in t[-edgeitems:]]
            else:
                vals = (
                    [round(float(x), float_precision) for x in t[:edgeitems]]
                    + ["..."]
                    + [round(float(x), float_precision) for x in t[-edgeitems:]]
                )
        else:
            if is_integer:
                vals = [int(x) for x in t]
            else:
                vals = [round(float(x), float_precision) for x in t]
        return "[" + ", ".join(map(str, vals)) + "]"

    if len(shape) == 2:
        rows = []
        n_rows = shape[0]
        n_cols = shape[1]
        row_indices = list(range(min(n_rows, max_rows)))
        for i in row_indices:
            row = t[i]
            if n_cols > max_cols:
                if is_integer:
                    vals = [int(x) for x in row[:edgeitems]] + ["..."] + [int(x) for x in row[-edgeitems:]]
                else:
                    vals = (
                        [round(float(x), float_precision) for x in row[:edgeitems]]
                        + ["..."]
                        + [round(float(x), float_precision) for x in row[-edgeitems:]]
                    )
            else:
                if is_integer:
                    vals = [int(x) for x in row]
                else:
                    vals = [round(float(x), float_precision) for x in row]
            rows.append("[" + ", ".join(map(str, vals)) + "]")
        if n_rows > max_rows:
            rows.append("...")

        return "\n".join(rows)

    return f"<unsupported tensor > 2D: shape={shape}>"


def format_per_token(
    tensor: torch.Tensor,
    tokens: list,
    edgeitems: int = 5,
    max_rows: int = 10,
    max_cols: int = 10,
    float_precision: int = 4,
) -> dict:
    """Format a tensor that contains per-token data for display.

    For 3D tensors (shape: n_tokens x ...): unbinds along first dimension to get
    separate 2D tensors for each token, then formats each as a string.

    For 2D tensors (shape: n_tokens x ...): unbinds along first dimension to get
    separate 1D tensors for each token, then formats each as a string.

    Returns a dict mapping each token to its formatted tensor string.
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(tensor)}")

    if len(tokens) != tensor.shape[0]:
        raise ValueError(f"Number of tokens ({len(tokens)}) must match first tensor dimension ({tensor.shape[0]})")

    # Unbind along first dimension to get per-token tensors
    unbound_tensors = tensor.cpu().unbind(dim=0)

    # Create dict mapping tokens to formatted tensors
    torch.set_printoptions(linewidth=80)
    formatted_dict = {
        tok: format_tensor(tensor, edgeitems, max_rows, max_cols, float_precision)
        for tok, tensor in zip(tokens, unbound_tensors)
    }

    return formatted_dict


def format_dict_sequence(
    obj,
    max_items: int = 10,
    edgeitems: int = 5,
) -> str:
    """
    Format a dict or sequence of dicts for display:
    - For tuple/list of dicts: show up to max_items, first/last edgeitems, ellipsis if longer
    - Each dict on a new line
    """
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            lines.append(f"{k!r}: {v}")
        return "\n".join(lines)
    if isinstance(obj, (tuple, list)) and all(isinstance(d, dict) for d in obj):
        n = len(obj)
        dicts = []
        if n > max_items:
            items = list(obj[:edgeitems]) + ["..."] + list(obj[-edgeitems:])
        else:
            items = obj
        for d in items:
            if d == "...":
                dicts.append("...")
            else:
                sorted_items = sorted(d.items())
                dict_str = "{" + ", ".join(f"{k!r}: {v!r}" for k, v in sorted_items) + "}"
                dicts.append(dict_str)
        return "\n".join(dicts)
    return str(obj)


def html_postprocess(text: str) -> str:
    """Replace line breaks with <br> for HTML tablefmt."""
    return text.replace("\n", "<br>")


def build_html_table(headers, rows) -> str:
    """Build a simple HTML table from headers and rows.

    Cells are inserted without escaping so that preformatted <br> tags inserted by formatting functions render
    correctly.
    """

    # Ensure headers and rows are strings and replace newlines with <br>
    def cell_str(x):
        if x is None:
            return ""
        s = str(x)
        # cell content may already include <br>; ensure newlines are replaced
        s = s.replace("\n", "<br>")
        return s

    parts = []
    parts.append("<table>")
    # headers
    parts.append("<thead><tr>")
    for h in headers:
        parts.append(f"<th>{cell_str(h)}</th>")
    parts.append("</tr></thead>")
    parts.append("<tbody>")
    for row in rows:
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<td>{cell_str(cell)}</td>")
        parts.append("</tr>")
    parts.append("</tbody>")
    parts.append("</table>")
    # Join without leading newline/space so display() places table immediately
    return "".join(parts)


def _format_value_for_display(
    obj: Any,
    tablefmt: str = "html",
    format_tensor_kwargs: dict | None = None,
) -> str:
    """Format a value for display, with special handling for HTML output.

    Args:
        obj: value to format (may be a tensor, dict/sequence of dicts, or other)
        tablefmt: table format (if 'html' the html postprocessing is applied)
        format_tensor_kwargs: optional kwargs forwarded to `format_tensor` when
            the object is a torch.Tensor (e.g. float_precision, edgeitems)
    """
    try:
        if isinstance(obj, torch.Tensor):
            effective_kwargs = format_tensor_kwargs if format_tensor_kwargs is not None else {}
            val = format_tensor(obj, **effective_kwargs)
        elif isinstance(obj, dict) or (isinstance(obj, (tuple, list)) and all(isinstance(d, dict) for d in obj)):
            val = format_dict_sequence(obj)
        else:
            val = str(obj)
    except Exception:
        val = str(obj)
    if tablefmt == "html":
        val = html_postprocess(val)
    return val


def format_tensor_sample(
    tensor: torch.Tensor,
    iter_indices: Sequence[int],
    headers: Sequence[str],
    k: int = 10,
    sample_op: Callable[..., Any] = torch.topk,
    tablefmt: str = "github",
    format_tensor_kwargs: dict | None = None,
) -> str:
    """Sample rows (dim 0) of `tensor` using `sample_op` (defaults to torch.topk), format the results and return
    either an HTML table (if tablefmt == 'html') or a tabulated string for other tablefmt values.

    Args:
        tensor: 2D tensor with shape (n_iters, width)
        iter_indices: list/seq of row indices along dim 0 to sample
        headers: sequence of header names for the table
        k: number of top elements to sample (passed to sample_op)
        sample_op: callable that given (row, k=...) returns (vals, inds)
        tablefmt: 'html' to return an HTML table, otherwise forwarded to tabulate

    Returns:
        Formatted table (HTML string if tablefmt == 'html', else tabulate string)
    """
    # Basic validation
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if tensor.ndim < 1:
        raise ValueError("tensor must have at least 1 dimension")
    n_rows = tensor.shape[0]
    width = tensor.shape[1] if tensor.ndim > 1 else 1

    max_k = min(k, width)
    # Normalize format kwargs to an empty dict to avoid truthy checks later
    effective_format_tensor_kwargs = format_tensor_kwargs if format_tensor_kwargs is not None else {}

    rows = []
    for it in iter_indices:
        if 0 <= it < n_rows:
            try:
                vals, inds = sample_op(tensor[it, :], k=max_k)
            except TypeError:
                # sample_op may accept different args; fallback to calling with single arg
                vals_inds = sample_op(tensor[it, :])
                if isinstance(vals_inds, tuple) and len(vals_inds) >= 2:
                    vals, inds = vals_inds[0], vals_inds[1]
                else:
                    raise
            # Use the centralized display formatter so HTML postprocessing
            # (e.g. replacing newlines with <br>) is applied when requested.
            vals_str = _format_value_for_display(
                vals, tablefmt=tablefmt, format_tensor_kwargs=effective_format_tensor_kwargs
            )
            inds_str = _format_value_for_display(
                inds, tablefmt=tablefmt, format_tensor_kwargs=effective_format_tensor_kwargs
            )
        else:
            vals_str = "n/a"
            inds_str = "n/a"
        rows.append([str(it), vals_str, inds_str])

    if tablefmt == "html":
        return build_html_table(headers, rows)

    try:
        return tabulate(rows, headers=headers, tablefmt=tablefmt)
    except Exception:
        # Last-resort plain text
        lines = [", ".join(headers)]
        for r in rows:
            lines.append(", ".join(map(str, r)))
        return "\n".join(lines)


def display_or_print(output: str, tablefmt: str = "html") -> None:
    """Display HTML output in IPython when available, otherwise print.

    This small helper centralizes the common pattern used when showing formatted tables or other outputs so it can be
    reused by callers.
    """
    if tablefmt == "html" and display is not None and HTML is not None:
        display(HTML(output))
    else:
        print(output)


def sample_tensor_output(
    tensor: torch.Tensor,
    iter_indices: Sequence[int],
    headers: Sequence[str],
    k: int = 10,
    sample_op: Callable[..., Any] = torch.topk,
    tablefmt: str = "github",
    format_tensor_kwargs: dict | None = None,
) -> None:
    """Sample rows from `tensor`, format them and show the result.

    This convenience function calls `format_tensor_sample(...)` to build the
    formatted table (HTML or plain-text) and then uses
    `display_or_print` to render it appropriately.
    """
    effective_format_tensor_kwargs = format_tensor_kwargs if format_tensor_kwargs is not None else {}

    output = format_tensor_sample(
        tensor=tensor,
        iter_indices=iter_indices,
        headers=headers,
        k=k,
        sample_op=sample_op,
        tablefmt=tablefmt,
        format_tensor_kwargs=effective_format_tensor_kwargs,
    )
    display_or_print(output, tablefmt=tablefmt)


def analysis_log(message: str, data: dict | None = None) -> None:
    """Log a simple message with optional structured data."""
    logger = get_analysis_logger()
    if data:
        logger.info(f"{message}\n{_format_data_for_logging(data)}")
    else:
        logger.info(message)


def get_caller_context(candidate_ctxs: list[str], target_ctx: str, skip_frames: int = 4) -> str:
    """Determine the calling context by inspecting the call stack.

    Args:
        candidate_ctxs: List of possible function names that could be calling contexts
        target_ctx: The specific context name we want to match (must be in candidate_ctxs)
        skip_frames: Number of frames to skip to reach the actual caller (default 4)

    Returns:
        The matched context name, or 'unknown' if not found

    This function walks up the call stack to find which of the candidate functions
    is calling the current analysis point. The skip_frames parameter accounts for:
    1. The current frame (inspect.currentframe() in this function)
    2. The analysis point function frame
    3. The hook execution frame
    4. The hooked function frame
    """
    import inspect

    if target_ctx not in candidate_ctxs:
        raise ValueError(f"target_ctx '{target_ctx}' must be in candidate_ctxs {candidate_ctxs}")

    context = "unknown"
    frame = inspect.currentframe()
    try:
        # Walk up the stack to find the calling context
        current_frame = frame
        for _ in range(skip_frames):
            if current_frame is None:
                break
            current_frame = current_frame.f_back

        while current_frame:
            func_name = current_frame.f_code.co_name
            if func_name in candidate_ctxs:
                context = func_name
                break
            current_frame = current_frame.f_back
    finally:
        del frame

    return context


def analysis_log_point(description: str, data: dict) -> None:
    """Log a standardized analysis point entry.

    Automatically determines the analysis point ID from the calling function name. Also collects data in the global
    ANALYSIS_DATA dictionary.
    """
    import inspect

    # Get the calling function name (the analysis function that called this)
    frame = inspect.currentframe()
    try:
        # frame.f_back is the caller of analysis_log_point (the analysis function)
        if frame and frame.f_back:
            caller_frame = frame.f_back
            point_id = caller_frame.f_code.co_name
        else:
            point_id = "unknown"
    finally:
        del frame

    # Collect data
    ANALYSIS_DATA[point_id] = data.copy()

    logger = get_analysis_logger()
    header = f"ANALYSIS_POINT_{point_id}: {description}"
    logger.info(f"{header}\n{_format_data_for_logging(data)}")


def _get_pkg_logger() -> logging.Logger:
    """Helper to quickly get a logger for orchestrator/patcher debug messages."""
    return logging.getLogger("analysis_injection")


class AnalysisInjectionOrchestrator:
    """Manages the complete lifecycle of the analysis injection framework."""

    def __init__(
        self,
        config_path: Path | str,
        target_package_path: Path | str,
        target_package_name: str = "circuit_tracer",
    ):
        """Initialize the orchestrator.

        Args:
            config_path: Path to YAML configuration file
            target_package_path: Path to target package root (e.g., circuit_tracer package)
            target_package_name: Name of the target package (e.g., 'circuit_tracer')
        """
        self.config_path = Path(config_path)
        self.target_package_path = Path(target_package_path)
        self.target_package_name = target_package_name

        self.config = None
        self.patched_modules: Dict[str, Path] = {}
        self.verification_info: Dict[str, Dict[str, Any]] = {}

        self._logger_initialized = False
        self._modules_patched = False
        self._hooks_registered = False
        self.logger = None  # Reference to the analysis logger
        self._config_override_data: Optional[Dict[str, Any]] = None
        self._config_override_source: Optional[Path] = None
        self._version_manager: Optional[Any] = None  # PackageVersionManager instance for cleanup

    def set_config_override(self, config_data: Dict[str, Any], source_path: Optional[Path] = None) -> None:
        """Provide an in-memory configuration that supersedes ``config_path``.

        Args:
            config_data: Fully merged configuration dictionary.
            source_path: Path to the base configuration file (used for resolving
                relative paths such as ``analysis_points_module_path``).
        """

        self._config_override_data = config_data
        self._config_override_source = source_path

    def load_config(self) -> None:
        """Load configuration from YAML file or provided override."""
        if self._config_override_data is not None:
            source_path = self._config_override_source if self._config_override_source is not None else self.config_path
            self.config = parse_config_dict(self._config_override_data, source_path=Path(source_path))
        else:
            self.config = load_config(self.config_path)

    def setup_logging(self) -> None:
        """Initialize the analysis logger."""
        if not self._logger_initialized and self.config:
            self.logger = init_analysis_logger(
                log_dir=self.config.log_dir,
                log_to_console=self.config.log_to_console,
                log_to_file=self.config.log_to_file,
                log_prefix=self.config.analysis_log_prefix,
            )
            self._logger_initialized = True

    def patch_files(self) -> None:
        """Patch target package files with hooks."""
        if not self._modules_patched and self.config:
            _get_pkg_logger().info(f"Starting patching of {self.target_package_name} files...")
            # Patch files
            self.patched_modules = patch_target_package_files(
                self.config.file_hooks, str(self.target_package_path), self.target_package_name
            )

            if not self.patched_modules:
                _get_pkg_logger().warning(
                    f"No patched modules produced by patch_target_package_files() for {self.target_package_name}"
                )
            else:
                _get_pkg_logger().info(f"Patched modules: {list(self.patched_modules.keys())}")

            # Install patched modules (replaces modules in sys.modules)
            # Note: install_patched_modules_with_references handles module removal, fresh loading,
            # and updating cross-references to patched functions
            install_patched_modules_with_references(self.patched_modules)

            # Verify patching
            self.verification_info = verify_patching(self.patched_modules)
            _get_pkg_logger().info("Patching verification complete")
            for module_name, info in self.verification_info.items():
                if info.get("hook_call_count", 0) > 0:
                    _get_pkg_logger().info(
                        f"  ✓ {module_name}: {info['hook_call_count']} hook calls, "
                        f"paths_match={info.get('paths_match', False)}"
                    )
                else:
                    _get_pkg_logger().warning(f"  ✗ {module_name}: No hook calls found!")

            self._modules_patched = True

    def register_hooks(self, tokenizer=None) -> None:
        """Register analysis hooks and set up shared context.

        Args:
            tokenizer: Optional tokenizer for converting token lists in shared_context to IDs
        """
        if not self._hooks_registered and self.config:
            # Get enabled points from config
            enabled_points = get_enabled_points(self.config)
            _get_pkg_logger().info(f"Enabled points resolved: {enabled_points}")

            # Initialize ANALYSIS_DATA for all enabled points (even if they haven't executed yet)
            for point_id in enabled_points:
                if point_id not in ANALYSIS_DATA:
                    ANALYSIS_DATA[point_id] = None

            # Register enabled analysis functions
            for point_id, func in ANALYSIS_FUNCTIONS.items():
                if point_id in enabled_points:
                    HOOK_REGISTRY.register(point_id, func)
                    _get_pkg_logger().info(f"Registered in-repo analysis function for point: {point_id}")

            # Set shared context from config
            if self.config.shared_context is not None:
                for key, value in self.config.shared_context.items():
                    # Set the value directly
                    HOOK_REGISTRY.set_context(**{key: value})

            # Log current hook registry contents
            try:
                _get_pkg_logger().info(f"HOOK_REGISTRY contains points: {list(HOOK_REGISTRY._hooks.keys())}")
            except Exception:
                _get_pkg_logger().warning("Unable to introspect HOOK_REGISTRY._hooks")

            self._hooks_registered = True

    def enable_hooks(self) -> None:
        """Enable hook execution."""
        if self.config and self.config.enabled:
            HOOK_REGISTRY.enable()

    def disable_hooks(self) -> None:
        """Disable hook execution."""
        HOOK_REGISTRY.disable()

    def setup(self, tokenizer=None) -> None:
        """Complete setup: load config, patch, register, enable.

        Args:
            tokenizer: Optional tokenizer for converting token lists in shared_context to IDs
        """
        self.load_config()
        self.setup_logging()
        self.patch_files()
        self.register_hooks(tokenizer=tokenizer)
        self.enable_hooks()

    def teardown(self) -> None:
        """Disable hooks and clean up."""
        self.disable_hooks()

        # Cleanup version manager if present
        if self._version_manager is not None:
            self._version_manager.cleanup()

    @property
    def analysis_log(self) -> Optional[str]:
        """Get the path to the analysis log file if one exists.

        Returns:
            Path to the log file, or None if no file handler is found.
        """
        if self.logger and hasattr(self.logger, "handlers"):
            for handler in self.logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    return handler.baseFilename
        return None

    def get_verification_report(self) -> str:
        """Get a formatted report of the patching verification.

        Returns:
            Formatted string with verification details
        """
        if not self.verification_info:
            return "No verification info available. Run setup() first."

        lines = [f"Patching Verification Report for {self.target_package_name}:"]
        lines.append("=" * 60)

        for module_name, info in self.verification_info.items():
            lines.append(f"\nModule: {module_name}")
            lines.append(f"  Loaded: {info['loaded']}")
            lines.append(f"  File Path: {info.get('file_path', 'N/A')}")
            lines.append(f"  Expected Path: {info.get('expected_patched_path', 'N/A')}")
            lines.append(f"  Paths Match: {info.get('paths_match', False)}")
            lines.append(f"  Has HOOK_REGISTRY: {info.get('has_hook_registry', False)}")
            lines.append(f"  Hook Call Count: {info.get('hook_call_count', 0)}")
            lines.append(f"  Has Analysis Import: {info.get('has_analysis_import', False)}")

            functions_with_hooks = info.get("functions_with_hooks", [])
            if functions_with_hooks:
                lines.append(f"  Functions with hooks: {', '.join(functions_with_hooks)}")

            if info.get("read_error"):
                lines.append(f"  Read Error: {info['read_error']}")
            if info.get("inspect_error"):
                lines.append(f"  Inspect Error: {info['inspect_error']}")

        return "\n".join(lines)

    def __getitem__(self, key: str) -> dict | None:
        """Get analysis data for a specific point ID.

        Args:
            key: Analysis point ID

        Returns:
            Data dictionary for the analysis point, or None if not executed, or raises KeyError if point not registered
        """
        return ANALYSIS_DATA[key]

    def __iter__(self):
        """Iterate over analysis point IDs."""
        return iter(ANALYSIS_DATA)

    def items(self):
        """Return an iterator over (point_id, data) pairs."""
        return ANALYSIS_DATA.items()

    def keys(self):
        """Return an iterator over analysis point IDs."""
        return ANALYSIS_DATA.keys()

    def values(self):
        """Return an iterator over analysis data values."""
        return ANALYSIS_DATA.values()

    def get_output(
        self,
        key: str,
        tablefmt: str = "html",
        skip: Union[str, Sequence[str], None] = None,
        format_tensor_kwargs: dict | None = None,
    ) -> None:
        """Get formatted output for a specific analysis point and display/print it.

        Args:
            key: Analysis point ID
            tablefmt: Table format for tabulate. If 'html', displays HTML table.
                     Otherwise, prints the table with the specified format.
            skip: Single key or sequence of keys to skip when displaying data.
                 Can be a string, list, tuple, or None.
        """
        try:
            data = self[key]
            if data is None:
                output = f"No analysis data for analysis point {key}"
            else:
                # Normalize skip parameter to a set of keys to skip
                if skip is None:
                    keys_to_skip = set()
                elif isinstance(skip, str):
                    keys_to_skip = {skip}
                else:
                    # Handle sequence types (list, tuple, etc.)
                    try:
                        keys_to_skip = set(skip)
                    except TypeError:
                        keys_to_skip = {skip}

                # Filter data to exclude skipped keys
                filtered_data = {k: v for k, v in data.items() if k not in keys_to_skip}

                # Normalize format kwargs to an empty dict to avoid truthy checks
                effective_format_tensor_kwargs = format_tensor_kwargs if format_tensor_kwargs is not None else {}

                # Check if any values are VarAnnotate objects
                has_var_inspects = any(isinstance(v, VarAnnotate) for v in filtered_data.values())

                if has_var_inspects:
                    # Create table with VarAnnotate-aware columns
                    table_data = []
                    for k, v in filtered_data.items():
                        if isinstance(v, VarAnnotate):
                            # Merge per-instance kwargs with caller overrides.
                            merged_kwargs = dict(v.format_tensor_kwargs or {})
                            merged_kwargs.update(effective_format_tensor_kwargs or {})
                            # Format the underlying inspected value using the merged kwargs.
                            formatted = _format_value_for_display(
                                v.var_value, tablefmt, format_tensor_kwargs=merged_kwargs
                            )
                            table_data.append([k, formatted, v.annotation])
                        else:
                            # Non-VarAnnotate values go in the output column
                            table_data.append(
                                [
                                    k,
                                    _format_value_for_display(
                                        v, tablefmt, format_tensor_kwargs=effective_format_tensor_kwargs
                                    ),
                                    "",
                                ]
                            )

                    # For HTML we will build our own table for more control especially w.r.t. tensor formatting
                    if tablefmt == "html":
                        headers = ["Key", "Inspected Output", "Annotation"]
                        output = build_html_table(headers, table_data)
                    else:
                        output = tabulate(
                            table_data,
                            headers=["Key", "Inspected Output", "Annotation"],
                            tablefmt=tablefmt,
                        )
                else:
                    # Standard table for non-VarAnnotate data
                    table_data = [
                        [k, _format_value_for_display(v, tablefmt, format_tensor_kwargs=format_tensor_kwargs)]
                        for k, v in filtered_data.items()
                    ]
                    if tablefmt == "html":
                        output = build_html_table(["Key", "Value"], table_data)
                    else:
                        output = tabulate(table_data, headers=["Key", "Value"], tablefmt=tablefmt)

            display_or_print(output, tablefmt=tablefmt)

        except KeyError:
            error_msg = f"analysis point key '{key}' not found"
            if tablefmt == "html" and display is not None and HTML is not None:
                display(HTML(f"<pre>{error_msg}</pre>"))
            else:
                print(error_msg)


# Convenience function for notebook usage
def setup_analysis_injection(
    config_path: Optional[Path | str] = None,
    target_package: Optional[str] = None,
    target_package_path: Optional[Path | str] = None,
    analysis_functions: Optional[Dict] = None,
    tokenizer=None,
    *,
    config_overrides: Optional[str] = None,
) -> AnalysisInjectionOrchestrator:
    """Set up analysis injection with default paths.

    Args:
        config_path: Path to config file (required)
        target_package: Name of package to patch (e.g., 'circuit_tracer'). Auto-detected if None.
        target_package_path: Path to target package (auto-detected if None)
        analysis_functions: Dict mapping point IDs to analysis functions that should augment the
            functions loaded from the configured module (e.g., {'0_10': ap_0_10, ...})
        tokenizer: Optional tokenizer for converting token lists in shared_context to IDs
        config_overrides: Optional YAML string or path to YAML file to merge into the base configuration
            prior to parsing. Supports nested dictionaries and declarative hook updates.

    Returns:
        Configured AnalysisInjectionOrchestrator instance

    Example:
        >>> def ap_0_10(ctx):
        ...     return None
        >>> def ap_0_11(ctx):
        ...     return None
        >>> AP_FUNCTIONS = {'0_10': ap_0_10, '0_11': ap_0_11}
        >>> orchestrator = setup_analysis_injection(  # doctest: +SKIP
        ...     config_path='config.yaml',
        ...     target_package='circuit_tracer',
        ...     analysis_functions=AP_FUNCTIONS
        ... )
        >>> # Run code with hooks active
        >>> orchestrator.teardown()  # doctest: +SKIP
    """
    # Require explicit config path to avoid ambiguity. Callers typically pass the
    # repository-provided analysis_injection_config.yaml and rely on
    # ``config_overrides`` to tweak settings.
    if config_path is None:
        raise ValueError("config_path must be provided. Supply the base analysis_injection_config.yaml path to load.")

    # Auto-detect target package if not provided (backward compatibility with circuit_tracer)
    if target_package is None:
        target_package = "circuit_tracer"

    # Auto-detect package path if not provided
    if target_package_path is None:
        try:
            module = __import__(target_package)
            target_package_path = Path(module.__file__).parent
        except ImportError:
            raise ValueError(
                f"Could not auto-detect path for package '{target_package}'. "
                f"Please provide target_package_path explicitly."
            )

    with open(config_path, "r", encoding="utf-8") as handle:
        base_config_dict = yaml.safe_load(handle)

    # Parse config_overrides if provided
    override_config_dict = None
    if config_overrides:
        # Check if it's a file path
        override_path = Path(config_overrides)
        if override_path.exists() and override_path.is_file():
            with open(override_path, "r", encoding="utf-8") as handle:
                override_config_dict = yaml.safe_load(handle)
        else:
            # Treat as YAML string
            override_config_dict = yaml.safe_load(config_overrides)

    merged_config_dict = (
        merge_config_dict(base_config_dict, override_config_dict) if override_config_dict else base_config_dict
    )

    # Handle version management before patching
    version_manager = None
    if merged_config_dict.get("settings", {}).get("target_package_version"):
        from .version_manager import PackageVersionManager

        required_version = merged_config_dict["settings"]["target_package_version"]
        version_manager = PackageVersionManager(target_package, required_version)

        # Install temp version if needed and get package path
        if version_manager.needs_temp_install():
            # Remove old module from sys.modules if it was imported with wrong version
            package_module = target_package.replace("-", "_")
            if package_module in sys.modules:
                _get_pkg_logger().info(f"Removing {package_module} from sys.modules to reload correct version")
                # Remove all submodules too
                to_remove = [k for k in sys.modules if k.startswith(f"{package_module}.")]
                for k in to_remove:
                    del sys.modules[k]
                del sys.modules[package_module]

            # Install and get path
            target_package_path = version_manager.install_temp_version()
        else:
            # Use existing installation path (already determined above)
            pass

    orchestrator = AnalysisInjectionOrchestrator(
        config_path=config_path,
        target_package_path=target_package_path,
        target_package_name=target_package,
    )
    orchestrator._version_manager = version_manager  # Store for cleanup
    orchestrator.set_config_override(merged_config_dict, source_path=Path(config_path))

    # Setup patches first
    orchestrator.load_config()
    config = orchestrator.config
    if config is None:
        raise RuntimeError("Failed to load analysis injection configuration")

    merged_config_yaml = yaml.safe_dump(merged_config_dict, sort_keys=False)
    _get_pkg_logger().info("Final merged analysis injection config:\n%s", merged_config_yaml)

    auto_functions: Dict[str, Any] = {}
    if config.analysis_points_module_path:
        auto_functions = _load_analysis_functions_from_module(config.analysis_points_module_path)
        _get_pkg_logger().info(
            "Loaded %s analysis functions from %s",
            len(auto_functions),
            config.analysis_points_module_path,
        )

    combined_functions: Dict[str, Any] = dict(auto_functions)
    if analysis_functions:
        combined_functions.update(analysis_functions)
        _get_pkg_logger().info("Merged %s caller-supplied analysis functions", len(analysis_functions))

    ANALYSIS_FUNCTIONS.clear()
    ANALYSIS_FUNCTIONS.update(combined_functions)

    orchestrator.setup_logging()
    orchestrator.patch_files()

    # Validate file hooks
    from .analysis_hook_patcher import validate_file_hooks

    if orchestrator.config:
        issues = validate_file_hooks(orchestrator.config.file_hooks, orchestrator.target_package_path)
        if issues["missing"] or issues["multiple"]:
            error_msg = "Hook validation failed:\n"
            for m in issues["missing"]:
                error_msg += f"  MISSING: {m}\n"
            for m in issues["multiple"]:
                error_msg += f"  MULTIPLE: {m}\n"
            raise RuntimeError(error_msg)

    # Complete setup
    orchestrator.register_hooks(tokenizer=tokenizer)
    orchestrator.enable_hooks()
    orchestrator._hooks_registered = True

    # Print log file location
    analysis_logger = get_analysis_logger()
    if analysis_logger and hasattr(analysis_logger, "handlers"):
        for handler in analysis_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                print(f"📝 Analysis output will be logged to: {handler.baseFilename}")
                break

    return orchestrator
