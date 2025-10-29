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
"""Version management for target packages in analysis injection.

This module provides utilities for managing package versions when running analysis notebooks. It allows installing a
temporary version of a package (e.g., circuit_tracer) to a temp directory without modifying the user's installed
packages.
"""

from __future__ import annotations

import importlib.metadata
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Mapping of (package_name, version) to git-based installation URLs
# Used as fallback when package is not available on PyPI
# Note: Supports both hyphenated and underscored package names
GIT_FALLBACK_URLS = {
    (
        "circuit-tracer",
        "0.1.0",
    ): "git+https://github.com/speediedan/circuit-tracer.git@b228bf190fadb3cb30f6a5ba6691dc4c86d76ba3",
    (
        "circuit_tracer",
        "0.1.0",
    ): "git+https://github.com/speediedan/circuit-tracer.git@b228bf190fadb3cb30f6a5ba6691dc4c86d76ba3",
}


class PackageVersionManager:
    """Manages temporary installation of specific package versions.

    This class handles version checking and temporary installation of packages
    to ensure compatibility with analysis notebooks without modifying the user's
    environment.

    Example:
        >>> mgr = PackageVersionManager("circuit-tracer", "0.1.0")
        >>> if mgr.needs_temp_install():
        ...     pkg_path = mgr.install_temp_version()
        >>> # Use the package...
        >>> mgr.cleanup()
    """

    def __init__(self, package_name: str, required_version: str):
        """Initialize the version manager.

        Args:
            package_name: Name of the package (e.g., 'circuit-tracer')
            required_version: Required version string (e.g., '0.1.0')
        """
        self.package_name = package_name
        self.required_version = required_version
        self.temp_dir: Optional[Path] = None
        self.temp_site_packages: Optional[Path] = None
        self._original_path_entry: Optional[str] = None

    def get_installed_version(self) -> Optional[str]:
        """Get currently installed version of the package.

        Returns:
            Version string if installed, None otherwise
        """
        try:
            return importlib.metadata.version(self.package_name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def needs_temp_install(self) -> bool:
        """Check if we need to install a temporary version.

        Returns:
            True if temp installation is needed, False otherwise
        """
        installed = self.get_installed_version()
        if installed is None:
            logger.warning(f"{self.package_name} not installed, will use temp version {self.required_version}")
            return True
        if installed != self.required_version:
            logger.info(
                f"{self.package_name} version mismatch: "
                f"installed={installed}, required={self.required_version}. "
                f"Installing temporary version {self.required_version}..."
            )
            return True
        logger.info(f"{self.package_name} version {installed} matches requirement")
        return False

    def install_temp_version(self) -> Path:
        """Install required version to temp directory and update sys.path.

        This method installs the package to a temporary directory and prepends
        it to sys.path so that imports resolve to the temp version. The user's
        installed version remains unchanged.

        For packages not available on PyPI, falls back to git-based installation
        if a URL is available in GIT_FALLBACK_URLS.

        Returns:
            Path to the package in the temp site-packages directory

        Raises:
            RuntimeError: If installation fails or package not found after install
        """
        if not self.needs_temp_install():
            # Return path to existing installation
            import importlib.util

            spec = importlib.util.find_spec(self.package_name.replace("-", "_"))
            if spec and spec.origin:
                return Path(spec.origin).parent
            raise RuntimeError(f"Could not locate installed {self.package_name}")

        # Create temp directory for installation
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"{self.package_name}_"))
        logger.info(f"Created temp directory: {self.temp_dir}")

        # Try PyPI installation first
        package_spec = f"{self.package_name}=={self.required_version}"
        success = self._try_install_from_pypi(package_spec)

        # If PyPI installation fails, try git-based fallback
        if not success:
            git_url = GIT_FALLBACK_URLS.get((self.package_name, self.required_version))
            if git_url:
                logger.warning(
                    f"{self.package_name}=={self.required_version} not available on PyPI. "
                    f"Attempting git-based installation from fallback URL."
                )
                success = self._try_install_from_git(git_url)
            else:
                raise RuntimeError(
                    f"Failed to install {package_spec}: not available on PyPI and no git fallback URL configured"
                )

        if not success:
            raise RuntimeError(f"All installation methods failed for {package_spec}")

        # Setup sys.path and verify installation
        self.temp_site_packages = self.temp_dir
        self._original_path_entry = str(self.temp_site_packages)
        sys.path.insert(0, self._original_path_entry)
        logger.info(f"Added {self.temp_site_packages} to sys.path")

        # Verify installation
        package_path = self.temp_site_packages / self.package_name.replace("-", "_")
        if not package_path.exists():
            raise RuntimeError(f"Package not found at expected location: {package_path}")

        logger.info(f"Successfully installed {self.package_name}=={self.required_version} to temp location")
        return package_path

    def _try_install_from_pypi(self, package_spec: str) -> bool:
        """Try to install package from PyPI.

        Args:
            package_spec: Package specification (e.g., 'circuit-tracer==0.1.0')

        Returns:
            True if installation succeeded, False otherwise
        """
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(self.temp_dir),
            "--no-deps",  # Don't install dependencies (assume they're already available)
            "--no-warn-script-location",  # Suppress warnings about scripts not in PATH
            package_spec,
        ]

        logger.info(f"Attempting PyPI installation: {package_spec}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"Successfully installed {package_spec} from PyPI")
            return True
        else:
            logger.debug(f"PyPI installation failed: {result.stderr}")
            return False

    def _try_install_from_git(self, git_url: str) -> bool:
        """Try to install package from git URL.

        Args:
            git_url: Git URL for installation (e.g., 'git+https://...')

        Returns:
            True if installation succeeded, False otherwise
        """
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(self.temp_dir),
            "--no-deps",  # Don't install dependencies (assume they're already available)
            "--no-warn-script-location",  # Suppress warnings about scripts not in PATH
            git_url,
        ]

        logger.info(f"Attempting git-based installation from: {git_url}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Successfully installed from git URL")
            return True
        else:
            logger.error(f"Git installation failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
            return False

    def cleanup(self):
        """Remove temp directory from sys.path, clean up sys.modules, and delete temp files.

        This method ensures complete cleanup of the temporary package installation:
        1. Removes the temp directory from sys.path
        2. Removes all package modules from sys.modules to prevent stale references
        3. Deletes the temporary directory to avoid leaving artifacts
        """
        # Remove from sys.path first
        if self._original_path_entry and self._original_path_entry in sys.path:
            sys.path.remove(self._original_path_entry)
            logger.info(f"Removed {self._original_path_entry} from sys.path")

        # Clean up sys.modules to remove all traces of the temp installation
        # This is critical to ensure the original (potentially editable) installation is used
        package_module_name = self.package_name.replace("-", "_")
        modules_to_remove = []

        for module_name in list(sys.modules.keys()):
            # Remove the package itself and all its submodules
            if module_name == package_module_name or module_name.startswith(f"{package_module_name}."):
                modules_to_remove.append(module_name)

        for module_name in modules_to_remove:
            del sys.modules[module_name]
            logger.debug(f"Removed {module_name} from sys.modules")

        if modules_to_remove:
            logger.info(f"Removed {len(modules_to_remove)} module(s) from sys.modules for {self.package_name}")

        # Delete temp directory to avoid leaving artifacts that could confuse pip or import system
        if self.temp_dir and self.temp_dir.exists():
            import shutil

            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Deleted temp directory: {self.temp_dir}")
            except OSError as e:
                # Log but don't fail on cleanup errors (e.g., permission issues on Windows)
                logger.warning(f"Failed to delete temp directory {self.temp_dir}: {e}")

        # Reset state
        self.temp_dir = None
        self.temp_site_packages = None
        self._original_path_entry = None
