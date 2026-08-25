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

import pytest

from it_examples.utils.analysis_injection.version_manager import GIT_FALLBACK_URLS, PackageVersionManager


def test_git_fallback_urls_configured():
    """Test that git fallback URLs are configured correctly."""
    assert len(GIT_FALLBACK_URLS) > 0
    # Check circuit-tracer 0.1.0 is configured
    assert ("circuit-tracer", "0.1.0") in GIT_FALLBACK_URLS
    url = GIT_FALLBACK_URLS[("circuit-tracer", "0.1.0")]
    assert url.startswith("git+https://")
    assert "circuit-tracer" in url


def test_package_name_normalization():
    """Test that package name normalization works according to PEP 503."""
    # Test various forms normalize to the same canonical name
    assert PackageVersionManager._normalize_package_name("circuit_tracer") == "circuit-tracer"
    assert PackageVersionManager._normalize_package_name("circuit-tracer") == "circuit-tracer"
    assert PackageVersionManager._normalize_package_name("Circuit.Tracer") == "circuit-tracer"
    assert PackageVersionManager._normalize_package_name("circuit__tracer") == "circuit-tracer"
    assert PackageVersionManager._normalize_package_name("CIRCUIT_TRACER") == "circuit-tracer"


def test_normalized_git_fallback_lookup():
    """Test that git fallback lookup works with normalized package names."""
    # Both forms should find the same URL after normalization
    mgr_underscore = PackageVersionManager("circuit_tracer", "0.1.0")
    mgr_hyphen = PackageVersionManager("circuit-tracer", "0.1.0")

    normalized_underscore = mgr_underscore._normalize_package_name(mgr_underscore.package_name)
    normalized_hyphen = mgr_hyphen._normalize_package_name(mgr_hyphen.package_name)

    assert normalized_underscore == normalized_hyphen == "circuit-tracer"

    url_underscore = GIT_FALLBACK_URLS.get((normalized_underscore, "0.1.0"))
    url_hyphen = GIT_FALLBACK_URLS.get((normalized_hyphen, "0.1.0"))

    assert url_underscore == url_hyphen
    assert url_underscore is not None
    assert url_underscore.startswith("git+https://")


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
    """Test that temp install is needed when versions don't match.

    Both carve-outs are exercised here rather than mocked, so this asserts against however
    circuit-tracer actually got into the running environment: editable in a from-source dev env,
    a VCS pin in CI and in any env built from the pins.
    """
    mgr = PackageVersionManager("circuit-tracer", "99.99.99")
    if mgr.is_editable_install():
        # Editable installs preserve the developer's local checkout. When a required
        # version is requested but an editable install is present, we skip a temp
        # install and emit a UserWarning to make this behavior explicit to the user.
        with pytest.warns(match="Preserving editable install"):
            assert not mgr.needs_temp_install()
    elif mgr.get_vcs_commit() is not None:
        # A git-pinned install already fixes the code; the version string is derived from the
        # commit and cannot be matched against a hard-coded value across pin bumps.
        with pytest.warns(match="installed from a VCS pin"):
            assert not mgr.needs_temp_install()
    else:
        assert mgr.needs_temp_install()


def _vcs_direct_url(commit: str = "1e40cbf6e5edacf3f5bbb1bf11cb1ea375d37aea") -> dict:
    """Build a PEP 610 ``direct_url.json`` payload of the shape pip records for a git install."""
    return {
        "url": "https://github.com/speediedan/circuit-tracer.git",
        "vcs_info": {"vcs": "git", "commit_id": commit, "requested_revision": commit},
    }


def test_get_vcs_commit_returns_commit_for_vcs_install(monkeypatch):
    """A git-installed distribution reports the commit recorded under vcs_info."""
    mgr = PackageVersionManager("circuit-tracer", "0.1.0")
    monkeypatch.setattr(mgr, "_read_direct_url", lambda: _vcs_direct_url())
    assert mgr.get_vcs_commit() == "1e40cbf6e5edacf3f5bbb1bf11cb1ea375d37aea"


@pytest.mark.parametrize(
    "direct_url",
    [
        pytest.param(None, id="index_install"),
        pytest.param({"url": "file:///repo", "dir_info": {"editable": True}}, id="editable_install"),
        pytest.param({"url": "file:///wheel.whl"}, id="local_wheel"),
    ],
)
def test_get_vcs_commit_none_without_vcs_info(monkeypatch, direct_url):
    """Anything that is not a VCS install reports no commit."""
    mgr = PackageVersionManager("circuit-tracer", "0.1.0")
    monkeypatch.setattr(mgr, "_read_direct_url", lambda: direct_url)
    assert mgr.get_vcs_commit() is None


def test_needs_temp_install_skips_vcs_install_on_mismatch(monkeypatch):
    """A VCS-pinned install is honored over a stale configured version, and says so."""
    mgr = PackageVersionManager("circuit-tracer", "0.4.0")
    monkeypatch.setattr(mgr, "get_installed_version", lambda: "0.5.3.dev31+g1e40cbf6e")
    monkeypatch.setattr(mgr, "is_editable_install", lambda: False)
    monkeypatch.setattr(mgr, "_read_direct_url", lambda: _vcs_direct_url())

    with pytest.warns(UserWarning, match="installed from a VCS pin"):
        assert not mgr.needs_temp_install()


def test_needs_temp_install_vcs_matching_version_is_quiet(monkeypatch, recwarn):
    """A VCS install whose derived version happens to match warns about nothing."""
    mgr = PackageVersionManager("circuit-tracer", "0.5.3.dev31+g1e40cbf6e")
    monkeypatch.setattr(mgr, "get_installed_version", lambda: "0.5.3.dev31+g1e40cbf6e")
    monkeypatch.setattr(mgr, "is_editable_install", lambda: False)
    monkeypatch.setattr(mgr, "_read_direct_url", lambda: _vcs_direct_url())

    assert not mgr.needs_temp_install()
    assert not [w for w in recwarn if issubclass(w.category, UserWarning)]


def test_needs_temp_install_index_install_still_installs(monkeypatch):
    """The carve-out is scoped to VCS installs: an index install still triggers a temp install.

    This is the guard against the carve-out being too broad -- without it, a genuine version
    mismatch on a normally-installed package would be silently ignored.
    """
    mgr = PackageVersionManager("circuit-tracer", "0.4.0")
    monkeypatch.setattr(mgr, "get_installed_version", lambda: "0.3.0")
    monkeypatch.setattr(mgr, "is_editable_install", lambda: False)
    monkeypatch.setattr(mgr, "_read_direct_url", lambda: None)

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
