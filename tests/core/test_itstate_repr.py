from interpretune.config import ITState
from interpretune.config.transformer_lens import ITLensCustomConfig
from interpretune.session import ITSession
from interpretune.base.modules import BaseITModule


def test_itstate_basic_summary_shows_values(tmp_path):
    st = ITState()
    # set values
    st._current_epoch = 0
    st._global_step = 5
    st._log_dir = tmp_path / "logs"
    st._session_complete = False
    st._init_hparams = {"a": {"nested": 1}, "b": "long-string-value-that-will-be-truncated", "c": 123}

    s = repr(st)
    assert "epoch=0" in s or "epoch=0" in s
    assert "step=5" in s
    # log_dir should be shown as a path string
    assert str(st._log_dir) in s
    # session_complete should be visible as False
    assert "session_complete=False" in s
    # init_hparams should be present (private key preserved) and include keys a,b,c
    assert "_init_hparams" in st.to_dict()
    init = st.to_dict()["_init_hparams"]
    assert "a" in init and "b" in init and "c" in init
    # dict values should be summarized as {...}
    assert init["a"] == "{...}"
    # string value truncated representation
    assert isinstance(init["b"], str) and len(init["b"]) <= 20
    # numeric value preserved
    assert init["c"] == 123


def test_itstate_extensions_and_custom_obj_repr(tmp_path):
    st = ITState()
    st._extensions = {"debug_lm": None, "memprofiler": None}
    # custom config object; avoid running __post_init__ which tries to build a HookedTransformerConfig
    tlc = object.__new__(ITLensCustomConfig)
    tlc.cfg = {"n_layers": 2, "n_heads": 2}
    st._init_hparams = {"tl_cfg": tlc}
    d = st.to_dict()
    assert "_extensions" in d and isinstance(d["_extensions"], dict)
    assert d["_extensions"]["debug_lm"] is None
    # init_hparams should show the custom class with parentheses
    assert "_init_hparams" in d
    assert isinstance(d["_init_hparams"]["tl_cfg"], str)
    assert d["_init_hparams"]["tl_cfg"].startswith("ITLensCustomConfig(")


def test_session_and_module_repr_include_state(tmp_path):
    st = ITState()
    st._current_epoch = 1

    # create a minimal dummy instance that mimics a module with _it_state
    class SimpleModule:
        pass

    simple = SimpleModule()
    simple._it_state = st
    # build a minimal BaseITModule-like instance without calling its __init__
    DummyBase = type("DummyBase", (BaseITModule,), {})
    dummy_inst = object.__new__(DummyBase)
    # set attributes expected by BaseITModule.__repr__
    dummy_inst._it_state = st
    dummy_inst.model = simple

    # ITSession repr using the dummy module
    it_session = object.__new__(ITSession)
    it_session.datamodule = None
    it_session.module = dummy_inst
    srepr = repr(it_session)
    assert "ITSession(" in srepr
    assert "module=DummyBase" in srepr or "module=DummyBase" in srepr

    # BaseITModule repr should include ITState summary and model class
    mod_repr = repr(dummy_inst)
    assert "ITState(" in mod_repr
    assert "model=" in mod_repr
