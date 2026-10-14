"""Re-record ``jlens_repo_listing.json`` from the published lens repository.

Run from the repository root with network access: ``python tests/core/fixtures/record_jlens_repo_listing.py``.
It records every ``.pt`` path at the repository's current revision, plus the two sidecar facts lens resolution
reads (``hf_model_name`` and ``fit.method``) for every artifact directory that publishes a ``config.yaml``.
"""

from __future__ import annotations

import json
import pathlib

import yaml
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "neuronpedia/jacobian-lens"
OUT = pathlib.Path(__file__).parent / "jlens_repo_listing.json"
COMMENT = (
    "Recorded file listing of neuronpedia/jacobian-lens, used to exercise lens resolution offline. Kept as a "
    "recording rather than a hand-built sample because the awkward cases are the point: a stem that is not the "
    "directory name, a directory with two artifacts, directories whose only artifact is suffixed, a fitting corpus "
    "that is not wikitext, a directory that publishes no config.yaml at all, and J++ lenses under `jpp/` beside "
    "J-lenses under `jlens/` (plus non-lens `oracle/` directories). `sidecars` maps each artifact directory to the "
    "sidecar facts resolution reads. Re-record with record_jlens_repo_listing.py beside this file."
)


def _sidecar_facts(revision: str, sidecar_path: str) -> dict:
    text = pathlib.Path(hf_hub_download(REPO_ID, sidecar_path, revision=revision)).read_text(encoding="utf-8")
    body = yaml.safe_load("\n".join(line for line in text.splitlines() if not line.startswith("#"))) or {}
    fit = body.get("fit") if isinstance(body.get("fit"), dict) else {}
    return {"hf_model_name": body.get("hf_model_name"), "fit_method": fit.get("method")}


def main() -> None:
    revision = HfApi().repo_info(REPO_ID).sha
    assert revision is not None
    files = sorted(HfApi().list_repo_files(REPO_ID, revision=revision))
    sidecars = {
        path.rsplit("/", 1)[0]: _sidecar_facts(revision, path)
        for path in files
        if path.endswith("/config.yaml") and len(path.split("/")) == 4
    }
    listing = {
        "_comment": COMMENT,
        "repo_id": REPO_ID,
        "revision": revision,
        "files": [f for f in files if f.endswith(".pt")],
        "sidecars": dict(sorted(sidecars.items())),
    }
    OUT.write_text(json.dumps(listing, indent=1) + "\n", encoding="utf-8")
    print(f"recorded {len(listing['files'])} .pt files and {len(sidecars)} sidecars at {revision}")


if __name__ == "__main__":
    main()
