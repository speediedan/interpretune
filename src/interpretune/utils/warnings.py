import warnings
from typing import Optional, Union, Type
from pathlib import Path


_default_format_warning = warnings.formatwarning

def _is_path_in_interpretune(path: Path) -> bool:
    """Naive check whether the path looks like a path from the Interpretune package."""
    return "interpretune" in str(path.absolute())

def _is_path_in_interpretune(path: Path) -> bool:
    """Naive check whether the path looks like a path from the Interpretune package."""
    return "interpretune" in str(path.absolute())

# adapted from lightning.fabric.utilities.warnings
def _custom_format_warning(
    message: Union[Warning, str], category: Type[Warning], filename: str, lineno: int, line: Optional[str] = None
) -> str:
    """Custom formatting that avoids an extra line in case warnings are emitted from the `rank_zero`-functions."""
    if _is_path_in_interpretune(Path(filename)):
        # The warning originates from the Interpretune package
        return f"{filename}:{lineno}: {message}\n"
    return _default_format_warning(message, category, filename, lineno, line)

warnings.formatwarning = _custom_format_warning
