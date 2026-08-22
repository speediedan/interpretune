"""#273: a collection fetched mid-session must be usable at the PREFERRED surface, `it.<op>`.

`OpWrapper.register_operations` snapshots op names onto the interpretune module when
`interpretune.analysis` is imported. `pull_ops(reload=True)` documents "usable immediately", which
was true at the dispatcher layer and false at the top-level wrapper surface -- precisely the one the
composition guide tells notebook authors to use. Measured before the fix: `DISPATCHER.get_op(name)`
succeeded while `it.<name>` raised AttributeError in the same process.
"""

from __future__ import annotations

import textwrap

import pytest


@pytest.fixture
def scratch_collection(tmp_path):
    (tmp_path / "scratch_ops.yaml").write_text(
        textwrap.dedent(
            """\
            collection:
              name: scratch_reload_ops
              version: 0.0.1

            scratch_reload_probe_op:
              description: Minimal op existing only to prove post-reload wrapper sync
              implementation: scratch_reload_defs.scratch_reload_probe_impl
              input_schema:
                orig_labels:
                  datasets_dtype: int64
                  required: false
              output_schema:
                preds:
                  datasets_dtype: int64
            """
        )
    )
    (tmp_path / "scratch_reload_defs.py").write_text(
        "def scratch_reload_probe_impl(module, analysis_batch, batch, batch_idx, **kwargs):\n"
        "    return analysis_batch\n"
    )
    return tmp_path


def test_reload_makes_new_collection_reachable_as_top_level_attr(scratch_collection, monkeypatch):
    """The notebook runtime order: import first, fetch later, use `it.<op>` immediately."""
    import interpretune as it
    import interpretune.analysis  # -- installs the wrapper surface (the snapshot)
    from interpretune.analysis.ops.dispatcher import DISPATCHER

    assert not hasattr(it, "scratch_reload_probe_op")

    # Appending a source then reloading is the runtime shape pull_ops produces: it downloads into the
    # hub cache (a directory discovery re-scans) and calls reload_definitions. IT_ANALYSIS_OP_PATHS
    # itself cannot be used here -- it is captured at import time into a module constant, so a
    # monkeypatched env var is invisible to a live dispatcher. That is by design for the LOCAL path
    # (set it before starting the session); the mid-session route is the hub pull.
    DISPATCHER.yaml_paths.append(scratch_collection)
    DISPATCHER.reload_definitions()  # what pull_ops(reload=True) runs after fetching
    try:
        wrapper = it.scratch_reload_probe_op  # the surface the guide points notebook authors at
        assert wrapper is not None
        # and the dispatcher agrees it is the same op, so the two surfaces cannot diverge again
        assert DISPATCHER.get_op("scratch_reload_probe_op", lazy=True) is not None
    finally:
        DISPATCHER.yaml_paths.remove(scratch_collection)
        DISPATCHER.reload_definitions()  # restore the bundled-only registry for later tests


def test_reload_without_wrapper_surface_stays_inert(scratch_collection, monkeypatch):
    """`_target_module is None` means the wrapper surface was never installed; reload must not install it."""
    from interpretune.analysis.ops.base import OpWrapper
    from interpretune.analysis.ops.dispatcher import DISPATCHER

    monkeypatch.setattr(OpWrapper, "_target_module", None)
    DISPATCHER.yaml_paths.append(scratch_collection)
    try:
        DISPATCHER.reload_definitions()  # must not raise reaching for a module that is not there
    finally:
        DISPATCHER.yaml_paths.remove(scratch_collection)
        monkeypatch.undo()
        DISPATCHER.reload_definitions()
