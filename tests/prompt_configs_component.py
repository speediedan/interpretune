"""The ``speediedan/prompt-configs`` component, resolved from the warmed components cache.

The Gemma and Llama3 prompt definitions are Hub-resident: CI warms the pinned revision from
``tests/hf_warm_manifest.yaml``, and a dev checkout fetches it once with
``it.hub.pull('speediedan/prompt-configs')``. Core tests read them from there rather than importing the in-tree
publish source, so the suite does not depend on ``it_examples``.
"""

from __future__ import annotations

from pathlib import Path

PROMPT_CONFIGS_REPO = "speediedan/prompt-configs"


def prompt_configs_dir() -> Path:
    """The warmed snapshot directory: the manifest and the ``prompt_configs.py`` entrypoint."""
    from interpretune.hub.components import resolve_component_manifest

    _, snapshot, _ = resolve_component_manifest(PROMPT_CONFIGS_REPO)
    return snapshot


def publish_prompt_configs(cache_dir: Path) -> str:
    """Local-publish the warmed snapshot into an isolated components cache; returns its pseudo-revision."""
    from interpretune.hub.components import local_publish

    return local_publish(prompt_configs_dir(), PROMPT_CONFIGS_REPO, cache_dir=cache_dir)


def prompt_config_class(name: str) -> type:
    """A prompt definition class (``GemmaPromptConfig``, ``Llama3PromptConfig``) from the default cache."""
    from interpretune.hub.promptconfigs import resolve_prompt_config_class

    return resolve_prompt_config_class(f"{PROMPT_CONFIGS_REPO}#{name}")
