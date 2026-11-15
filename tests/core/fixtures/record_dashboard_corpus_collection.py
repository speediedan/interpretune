"""Re-record ``dashboard_corpus_collection.json`` from the project's dashboard-corpus collection.

Run from the repository root with network access: ``python tests/core/fixtures/record_dashboard_corpus_collection.py``.
It records the collection's items and, for each bucket item, the ``dashboards.json`` discovery reads, both
anonymously, as a client without a token sees them.
"""

from __future__ import annotations

import json
import pathlib

from huggingface_hub import HfApi

from interpretune.utils.neuronpedia_hub_fetch import DEFAULT_CORPUS_COLLECTIONS, read_remote_manifest

OUT = pathlib.Path(__file__).parent / "dashboard_corpus_collection.json"
COMMENT = (
    "Recorded items of the project's dashboard-corpus collection and each bucket item's dashboards.json, used to "
    "exercise discovery offline. Two corpora share a model and differ in source set and prompt corpus, which is the "
    "case resolution must not collapse. Re-record with record_dashboard_corpus_collection.py beside this file."
)


def main() -> None:
    collections = {}
    manifests = {}
    for slug in DEFAULT_CORPUS_COLLECTIONS:
        items = HfApi(token=False).get_collection(slug, token=False).items
        collections[slug] = [{"item_type": i.item_type, "item_id": i.item_id} for i in items]
        for item in items:
            if item.item_type == "bucket":
                manifests[item.item_id] = read_remote_manifest(item.item_id, False)
    OUT.write_text(
        json.dumps({"_comment": COMMENT, "collections": collections, "manifests": manifests}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
