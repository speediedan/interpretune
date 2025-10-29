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
"""Tests for the version manager in analysis injection framework."""

from __future__ import annotations

from it_examples.utils.analysis_injection.version_manager import GIT_FALLBACK_URLS, PackageVersionManager


def test_git_fallback_urls_configured():
    """Test that git fallback URLs are configured correctly."""
    assert len(GIT_FALLBACK_URLS) > 0
    # Check circuit-tracer 0.1.0 is configured
    assert ("circuit-tracer", "0.1.0") in GIT_FALLBACK_URLS
    url = GIT_FALLBACK_URLS[("circuit-tracer", "0.1.0")]
    assert url.startswith("git+https://")
    assert "circuit-tracer" in url


def test_version_manager_init():
    """Test that PackageVersionManager can be initialized."""
    mgr = PackageVersionManager("circuit-tracer", "0.1.0")
    assert mgr.package_name == "circuit-tracer"
    assert mgr.required_version == "0.1.0"
    assert mgr.temp_dir is None
    assert mgr.temp_site_packages is None
    assert mgr._original_path_entry is None


def test_get_installed_version():
    """Test getting installed version of a package."""
    mgr = PackageVersionManager("circuit-tracer", "0.1.0")
    version = mgr.get_installed_version()
    # circuit-tracer should be installed in the test environment
    assert version is not None
    assert isinstance(version, str)


def test_get_installed_version_nonexistent():
    """Test getting version of a non-existent package."""
    mgr = PackageVersionManager("fake-package-xyz-12345", "1.0.0")
    version = mgr.get_installed_version()
    assert version is None


def test_needs_temp_install_matching_version():
    """Test that temp install is not needed when versions match."""
    mgr = PackageVersionManager("circuit-tracer", "0.1.0")
    # Assuming circuit-tracer 0.1.0 is installed in test environment
    installed = mgr.get_installed_version()
    if installed == "0.1.0":
        assert not mgr.needs_temp_install()


def test_needs_temp_install_mismatched_version():
    """Test that temp install is needed when versions don't match."""
    mgr = PackageVersionManager("circuit-tracer", "99.99.99")
    assert mgr.needs_temp_install()


def test_needs_temp_install_nonexistent():
    """Test that temp install is needed for non-existent package."""
    mgr = PackageVersionManager("fake-package-xyz-12345", "1.0.0")
    assert mgr.needs_temp_install()


def test_install_temp_version_with_matching_version():
    """Test install_temp_version when version already matches (should return existing path)."""
    mgr = PackageVersionManager("circuit-tracer", "0.1.0")
    installed = mgr.get_installed_version()

    if installed == "0.1.0":
        # Should return existing installation path without creating temp dir
        path = mgr.install_temp_version()
        assert path.exists()
        assert mgr.temp_dir is None  # No temp directory created
        assert mgr.temp_site_packages is None


def test_version_manager_cleanup():
    """Test that cleanup removes sys.path entry."""
    import sys

    mgr = PackageVersionManager("circuit-tracer", "0.1.0")

    # If we already have the right version, cleanup should be safe to call
    if not mgr.needs_temp_install():
        mgr.cleanup()
        # Should not raise any errors

    # If temp install was done, verify cleanup removes from sys.path
    if mgr._original_path_entry:
        assert mgr._original_path_entry not in sys.path
