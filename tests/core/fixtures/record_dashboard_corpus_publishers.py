"""Re-record ``dashboard_corpus_publishers.json`` from the default trusted publishers' buckets.

Run from the repository root with network access: ``python tests/core/fixtures/record_dashboard_corpus_publishers.py``.
For each default publisher it records every bucket, whether it carries a ``dashboards.json``, and that manifest when it
does, all anonymously, as a client without a token sees them.
"""

from __future__ import annotations

import json
import pathlib

from huggingface_hub import HfApi

from interpretune.hub.discovery import DEFAULT_TRUSTED_PUBLISHERS
from interpretune.utils.neuronpedia_hub_fetch import CORPUS_MANIFEST, read_remote_manifest

OUT = pathlib.Path(__file__).parent / "dashboard_corpus_publishers.json"
COMMENT = (
    "Recorded buckets of the default trusted publishers, and each corpus bucket's dashboards.json, used to exercise "
    "discovery offline. Two corpora share a model and differ in source set and prompt corpus, which is the case "
    "resolution must not collapse. Re-record with record_dashboard_corpus_publishers.py beside this file."
)


def main() -> None:
    api = HfApi(token=False)
    publishers: dict[str, list[dict]] = {}
    manifests = {}
    for publisher in DEFAULT_TRUSTED_PUBLISHERS:
        rows = []
        for bucket in api.list_buckets(namespace=publisher, token=False):
            has_manifest = bool(list(api.get_bucket_paths_info(bucket.id, [CORPUS_MANIFEST], token=False)))
            rows.append({"id": bucket.id, "has_manifest": has_manifest})
            if has_manifest:
                manifests[bucket.id] = read_remote_manifest(bucket.id, False)
        publishers[publisher] = sorted(rows, key=lambda r: r["id"])
    OUT.write_text(
        json.dumps({"_comment": COMMENT, "publishers": publishers, "manifests": manifests}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
