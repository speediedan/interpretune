"""Re-record ``dashboard_corpus_publishers.json`` from the default trusted publishers' buckets.

Run from the repository root with network access: ``python tests/core/fixtures/record_dashboard_corpus_publishers.py``.
For each default publisher it records every bucket, whether it carries a corpus ``it_artifact.json``, and that envelope
when it does, all anonymously, as a client without a token sees them.
"""

from __future__ import annotations

import json
import pathlib

from huggingface_hub import HfApi

from interpretune.hub.discovery import DEFAULT_TRUSTED_PUBLISHERS
from interpretune.utils.neuronpedia_hub_fetch import CORPUS_ENVELOPE, read_remote_corpus_envelope

OUT = pathlib.Path(__file__).parent / "dashboard_corpus_publishers.json"
COMMENT = (
    "Recorded buckets of the default trusted publishers, and each corpus bucket's it_artifact.json, used to exercise "
    "discovery offline. Two corpora share a model and differ in source set and prompt corpus, which is the case "
    "resolution must not collapse. Re-record with record_dashboard_corpus_publishers.py beside this file."
)


def main() -> None:
    api = HfApi(token=False)
    publishers: dict[str, list[dict]] = {}
    envelopes = {}
    for publisher in DEFAULT_TRUSTED_PUBLISHERS:
        rows = []
        for bucket in api.list_buckets(namespace=publisher, token=False):
            has_envelope = bool(list(api.get_bucket_paths_info(bucket.id, [CORPUS_ENVELOPE], token=False)))
            rows.append({"id": bucket.id, "has_envelope": has_envelope})
            if has_envelope:
                envelopes[bucket.id] = read_remote_corpus_envelope(bucket.id, False)
        publishers[publisher] = sorted(rows, key=lambda r: r["id"])
    OUT.write_text(
        json.dumps({"_comment": COMMENT, "publishers": publishers, "envelopes": envelopes}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
