from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "requirements" / "utils"))

from collect_env_details import _direct_url_git_metadata, _format_git_metadata  # noqa: E402


def test_direct_url_git_metadata_for_pinned_git_install():
    direct_url = {
        "url": "https://github.com/speediedan/circuit-tracer.git",
        "vcs_info": {
            "vcs": "git",
            "commit_id": "14cc3e8443d6bca1fc58752aca28207ba2128a3c",
            "requested_revision": "14cc3e8443d6bca1fc58752aca28207ba2128a3c",
        },
    }

    assert _direct_url_git_metadata(direct_url) == {
        "fork": "speediedan/circuit-tracer",
        "sha": "14cc3e8",
    }


def test_direct_url_git_metadata_preserves_named_ref_when_available():
    direct_url = {
        "url": "https://github.com/ndif-team/nnsight.git",
        "vcs_info": {
            "vcs": "git",
            "commit_id": "1234567890abcdef1234567890abcdef12345678",
            "requested_revision": "v0.6.0",
        },
    }

    assert _format_git_metadata(_direct_url_git_metadata(direct_url)) == (
        "(fork:ndif-team/nnsight, branch:v0.6.0, sha:1234567)"
    )
