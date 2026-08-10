"""4a loader-equivalence harness: baseline leg (hub design v3 §11.4).

Captures every CLI experiment config through the CURRENT jsonargparse session surface and asserts the
capture works. When ``load_session_cfg`` lands, each case gains the second leg: the unified loader's
output must be ``session_spec``-identical to this baseline BEFORE the old path is removed. The
equivalence set also includes a registry/examples configuration exercising ``AutoCompConfig``
``make_dataclass`` synthesis — none of the CLI configs uses AutoComp (they bind explicit
``RTEBoolq*Config`` classes), so that acceptance case cannot come from this parametrization alone.
"""

from __future__ import annotations

import pytest

from tests.core.loader_equivalence import (
    capture_session_cfg_via_cli,
    cli_experiment_configs,
    session_spec,
)

CONFIGS = cli_experiment_configs()


def test_all_cli_experiment_configs_discovered():
    """The harness must cover the full experiment-config surface; a moved/removed config shrinks it loudly."""
    assert len(CONFIGS) == 15, sorted(str(c) for c in CONFIGS)


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_old_path_capture_baseline(config_path):
    """Every experiment config instantiates a complete session_cfg via the current jsonargparse surface."""
    spec = session_spec(capture_session_cfg_via_cli([config_path]))
    assert spec["adapter_ctx"], "adapter_ctx must be non-empty"
    assert spec["datamodule_cfg"]["__class__"] and spec["module_cfg"]["__class__"]
    assert spec["datamodule_cls"] and spec["module_cls"]
    # shared-field link propagation is the CLI behavior the unified loader must reproduce exactly:
    # model_name_or_path is an ITSharedConfig field linked datamodule -> module at parse time
    assert spec["module_cfg"].get("model_name_or_path") == spec["datamodule_cfg"].get("model_name_or_path")
