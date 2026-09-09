"""The snapshot/restore pair around TransformerLens's process-wide gemma eager attention patch.

The pair exists because a bridge built earlier in a session breaks circuit-tracer's nnsight attention locations on every
gemma model for the rest of the process. These tests plant a stand-in wrapper so the mechanics are checked without
building a bridge; the real patch is measured by the circuit-tracer conformance target, which uses the fixture.
"""

from __future__ import annotations

import importlib

import pytest

from tests.utils import GEMMA_MODELING_MODULES, restore_gemma_eager_attention, snapshot_gemma_eager_attention


def _stand_in(*args, **kwargs):  # pragma: no cover - never called
    raise AssertionError("the planted wrapper must never run")


class TestGemmaEagerAttentionRestore:
    def test_restore_puts_the_originals_back_and_reinstates_the_wrapper_afterwards(
        self, gemma_eager_attention_originals, monkeypatch
    ):
        modules = {name: importlib.import_module(name) for name in GEMMA_MODELING_MODULES}
        for module in modules.values():
            monkeypatch.setattr(module, "eager_attention_forward", _stand_in)
        put_back = restore_gemma_eager_attention(gemma_eager_attention_originals)
        for name, module in modules.items():
            assert module.eager_attention_forward is gemma_eager_attention_originals[name]
            assert module.eager_attention_forward.__module__ == name
        put_back()
        for module in modules.values():
            assert module.eager_attention_forward is _stand_in, "a bridge alive from earlier must keep its hooks"

    def test_a_snapshot_taken_after_the_patch_is_refused_by_name(self, monkeypatch):
        module = importlib.import_module(GEMMA_MODELING_MODULES[1])
        monkeypatch.setattr(module, "eager_attention_forward", _stand_in)
        with pytest.raises(RuntimeError, match="already .*_stand_in; a snapshot taken now cannot vouch"):
            snapshot_gemma_eager_attention()

    def test_the_session_snapshot_is_the_modeling_modules_own(self, gemma_eager_attention_originals):
        for name, fn in gemma_eager_attention_originals.items():
            assert fn.__module__ == name and not hasattr(fn, "__wrapped__")
