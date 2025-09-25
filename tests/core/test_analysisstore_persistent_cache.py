import os
import time
from pathlib import Path
import datasets


def simple_generator():
    for i in range(3):
        yield {"x": i}


def test_dataset_cache_created_under_explicit_analysis_cache(tmp_path):
    parent_cache = tmp_path / "hf_cache"
    analysis_cache = parent_cache / "analysis_cache"
    os.makedirs(parent_cache, exist_ok=True)

    features = datasets.Features({"x": datasets.Value("int64")})

    # Create dataset using explicit analysis cache dir nested under parent
    _ = datasets.Dataset.from_generator(
        generator=simple_generator,
        features=features,
        cache_dir=str(analysis_cache),
        split="train",
    )

    # Ensure cache dir was created and contains dataset files
    assert analysis_cache.exists()
    contents = list(analysis_cache.glob("**/*"))
    assert len(contents) > 0, "Expected dataset cache files under analysis cache dir"


def test_dataset_reuse_existing_cache(tmp_path):
    parent_cache = tmp_path / "hf_cache"
    analysis_cache = parent_cache / "analysis_cache"
    os.makedirs(parent_cache, exist_ok=True)

    features = datasets.Features({"x": datasets.Value("int64")})

    # First creation
    _ = datasets.Dataset.from_generator(
        generator=simple_generator,
        features=features,
        cache_dir=str(analysis_cache),
        split="train",
    )

    # Snapshot listing and mtimes for files
    files_before = {p: p.stat().st_mtime for p in analysis_cache.rglob("*")}
    assert files_before, "No files created in cache dir"

    # Sleep to ensure mtimes would change if files are rewritten
    time.sleep(0.1)

    # Recreate dataset (should reuse cache, not rewrite files)
    _ = datasets.Dataset.from_generator(
        generator=simple_generator,
        features=features,
        cache_dir=str(analysis_cache),
        split="train",
    )

    files_after = {p: p.stat().st_mtime for p in analysis_cache.rglob("*")}

    # Filter out transient lock files that HF datasets may update on reuse
    def is_transient(p: Path):
        name = p.name
        return name.endswith(".lock") or name.startswith("_tmp_") or "builder.lock" in name

    stable_before = {p: m for p, m in files_before.items() if not is_transient(p)}
    stable_after = {p: m for p, m in files_after.items() if not is_transient(p)}

    # Ensure the same set of stable files exist
    assert set(stable_before.keys()) == set(stable_after.keys())

    # And stable mtimes didn't increase (i.e., files were not rewritten)
    rewritten = [p for p in stable_before if stable_after[p] > stable_before[p] + 1e-6]
    assert not rewritten, f"Expected cache reuse, but some stable files were rewritten: {rewritten}"
