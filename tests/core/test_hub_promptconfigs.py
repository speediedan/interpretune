"""4d promptconfigs kind: manifest validation, cache-only ref resolution, and composition parity."""

from __future__ import annotations

from pathlib import Path

import pytest

import it_examples

PROMPT_COMPONENT_DIR = Path(it_examples.__file__).parent / "examples" / "prompt_configs"


@pytest.fixture()
def seeded_cache(tmp_path):
    from it_examples.seeds import ensure_local_seeds

    cache = tmp_path / "components"
    ensure_local_seeds(cache_dir=cache)
    return cache


class TestPromptConfigsKind:
    def test_seed_manifest_validates(self):
        from interpretune.hub.manifest import load_component_manifest

        manifest = load_component_manifest(PROMPT_COMPONENT_DIR / "it_component.yaml")
        assert manifest["kinds"] == ["promptconfigs"]
        assert set(manifest["promptconfigs"]["definitions"]) == {"GemmaPromptConfig", "Llama3PromptConfig"}

    @pytest.mark.parametrize(
        ("mutation", "match"),
        [
            ({"it_schema_version": 1, "kinds": ["promptconfigs"]}, "requires a `promptconfigs.entrypoint`"),
            (
                {"it_schema_version": 1, "kinds": ["promptconfigs"], "promptconfigs": {"entrypoint": "x.py"}},
                "definitions",
            ),
            (
                {
                    "it_schema_version": 1,
                    "kinds": ["promptconfigs"],
                    "promptconfigs": {"entrypoint": "x.py", "definitions": {}},
                },
                "non-empty",
            ),
        ],
        ids=["no-section", "no-definitions", "empty-definitions"],
    )
    def test_kind_validation_failure_modes(self, mutation, match):
        from interpretune.hub.manifest import ComponentManifestError, validate_component_manifest

        with pytest.raises(ComponentManifestError, match=match):
            validate_component_manifest(mutation)

    def test_card_lists_definitions_and_ref_syntax(self):
        from interpretune.hub.cards import generate_component_card
        from interpretune.hub.manifest import load_component_manifest

        manifest = load_component_manifest(PROMPT_COMPONENT_DIR / "it_component.yaml")
        card = generate_component_card(manifest, "speediedan/prompt-configs")
        assert "interpretune-promptconfigs" in card.data.tags
        assert "`GemmaPromptConfig`" in str(card)
        assert "compose_ref" in str(card)


class TestCachedEntrypointResolution:
    def test_resolution_is_cache_only(self, seeded_cache, monkeypatch):
        import socket

        monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
        from interpretune.hub.promptconfigs import resolve_prompt_config_class

        cls = resolve_prompt_config_class("speediedan/prompt-configs#GemmaPromptConfig", cache_dir=seeded_cache)
        assert cls.__name__ == "GemmaPromptConfig"
        # published definition matches the (dissolving) in-repo spelling byte-for-byte
        from it_examples.example_prompt_configs import GemmaPromptConfig as InRepoGemma

        assert cls().model_chat_template_fn("Hi", "gemma-chat") == InRepoGemma().model_chat_template_fn(
            "Hi", "gemma-chat"
        )

    def test_unknown_definition_names_available(self, seeded_cache):
        from interpretune.hub.promptconfigs import resolve_prompt_config_class

        with pytest.raises(KeyError, match="GemmaPromptConfig"):
            resolve_prompt_config_class("speediedan/prompt-configs#Nope", cache_dir=seeded_cache)

    @pytest.mark.parametrize("ref", ["no-hash", "org/repo#", "norepo#Name"], ids=["no-hash", "no-name", "no-slash"])
    def test_malformed_ref_rejected(self, ref):
        from interpretune.hub.promptconfigs import resolve_prompt_config_class

        with pytest.raises(ValueError, match="Malformed prompt-config reference"):
            resolve_prompt_config_class(ref)

    def test_module_names_are_revision_scoped(self, seeded_cache, tmp_path):
        """Definitions from different cached revisions must never collide in sys.modules (umbrella spec)."""
        import shutil
        import sys

        from interpretune.hub.components import local_publish
        from interpretune.hub.promptconfigs import import_cached_entrypoint, resolve_prompt_config_class

        cls_v1 = resolve_prompt_config_class("speediedan/prompt-configs#GemmaPromptConfig", cache_dir=seeded_cache)
        # publish a MODIFIED copy of the component -> new content-derived revision in the same cache
        src_copy = tmp_path / "component"
        shutil.copytree(PROMPT_COMPONENT_DIR, src_copy)
        ep = src_copy / "prompt_configs.py"
        ep.write_text(ep.read_text(encoding="utf-8").replace('B_TEXT: str = "<bos>"', 'B_TEXT: str = "<BOS2>"'))
        local_publish(src_copy, "speediedan/prompt-configs", cache_dir=seeded_cache)
        cls_v2 = resolve_prompt_config_class("speediedan/prompt-configs#GemmaPromptConfig", cache_dir=seeded_cache)
        assert cls_v1 is not cls_v2 and cls_v2().B_TEXT == "<BOS2>" and cls_v1().B_TEXT == "<bos>"
        mods = [m for m in sys.modules if m.startswith("it_hub_components.speediedan__prompt_configs.")]
        assert len(mods) >= 2, mods
        # idempotent within one revision: repeated import returns the SAME module object
        module = import_cached_entrypoint("speediedan/prompt-configs", cache_dir=seeded_cache)
        assert import_cached_entrypoint("speediedan/prompt-configs", cache_dir=seeded_cache) is module


class TestComposeRefParity:
    """MRO-equivalence pins vs the dissolving hand-written compositions (umbrella spec, pre-deletion)."""

    @pytest.mark.parametrize(
        ("ref_name", "dissolving_cls_name", "pattern"),
        [
            ("GemmaPromptConfig", "RTEBoolqGemmaPromptConfig", "gemma-chat"),
            ("Llama3PromptConfig", "RTEBoolqLlama3PromptConfig", "llama3-chat"),
        ],
    )
    def test_composed_class_mirrors_dissolving_composition(self, seeded_cache, ref_name, dissolving_cls_name, pattern):
        import dataclasses

        import it_examples.example_prompt_configs as legacy
        from it_examples.experiments.rte_boolq import RTEBoolqPromptConfig
        from interpretune.hub.promptconfigs import compose_prompt_config_class, resolve_prompt_config_class

        ref_cls = resolve_prompt_config_class(f"speediedan/prompt-configs#{ref_name}", cache_dir=seeded_cache)
        composed = compose_prompt_config_class(ref_cls, RTEBoolqPromptConfig)
        dissolving = getattr(legacy, dissolving_cls_name)
        # (RefClass, TaskSchema) base order mirrors the hand-written MRO shape
        assert [b.__name__ for b in composed.__bases__] == [b.__name__ for b in dissolving.__bases__]
        # field surface identical
        assert {f.name for f in dataclasses.fields(composed)} == {f.name for f in dataclasses.fields(dissolving)}
        # produced prompt bytes identical
        assert composed().model_chat_template_fn("Does A imply B?", pattern) == dissolving().model_chat_template_fn(
            "Does A imply B?", pattern
        )

    def test_compose_ref_through_factory_merge_site(self, seeded_cache, monkeypatch):
        """The one-merge-site path instantiates a compose_ref prompt_cfg node end-to-end."""
        from interpretune.hub import promptconfigs as pc_mod
        from interpretune.registry import itdm_cfg_factory

        orig = pc_mod.instantiate_prompt_cfg_node
        monkeypatch.setattr(
            pc_mod, "instantiate_prompt_cfg_node", lambda node, cache_dir=None: orig(node, cache_dir=seeded_cache)
        )
        cfg = itdm_cfg_factory(
            {
                "prompt_cfg": {
                    "class_path": "it_examples.experiments.rte_boolq.RTEBoolqPromptConfig",
                    "compose_ref": "speediedan/prompt-configs#GemmaPromptConfig",
                },
                "signature_columns": ["input", "labels"],
            },
            {"model_name_or_path": "gpt2", "task_name": "rte"},
        )
        assert type(cfg.prompt_cfg).__name__ == "GemmaPromptConfig_RTEBoolqPromptConfig"
        assert cfg.prompt_cfg.model_chat_template_fn("Hi", "gemma-chat").startswith("<bos><start_of_turn>user")


class TestChatTemplatePromptConfig:
    """Chat-template-first default (design §11.5): delegate to the tokenizer; graceful fallback."""

    class _FakeTokenizer:
        chat_template = "{{ messages }}"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert not tokenize
            return f"<tmpl>{messages[0]['content']}<gen={add_generation_prompt}>"

    def test_delegates_when_tokenizer_bound(self):
        from interpretune.config.datamodule import ChatTemplatePromptConfig

        cfg = ChatTemplatePromptConfig()
        cfg.bind_tokenizer(self._FakeTokenizer())
        assert cfg.model_chat_template_fn("  Hi  ") == "<tmpl>Hi<gen=True>"

    def test_falls_back_without_tokenizer_or_template(self):
        from interpretune.config.datamodule import ChatTemplatePromptConfig

        cfg = ChatTemplatePromptConfig()
        assert cfg.model_chat_template_fn("  Hi  ") == "Hi"  # unbound
        tokenizer = self._FakeTokenizer()
        tokenizer.chat_template = None
        cfg.bind_tokenizer(tokenizer)
        assert cfg.model_chat_template_fn("Hi") == "Hi"  # template-less model
