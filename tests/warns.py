import re
from functools import partial
from typing import List, Optional
from warnings import WarningMessage
from packaging.version import Version
from pkg_resources import get_distribution

from interpretune.utils import dummy_method_warn_fingerprint
from interpretune.protocol import Adapter
from tests.runif import EXTENDED_VER_PAT


EXPECTED_WARNS = [
    "The truth value of an empty array is ambiguous",  # for jsonargparse
    ".*torch.cpu.amp.autocast.*",  # required for PT nightly 20240601, likely can be removed with 2.4 PT release
]

HF_EXPECTED_WARNS = [
    "Please use torch.utils._pytree.register_pytree_node instead",  # temp  allow deprecated call from hf
    "use_reentrant parameter should be",  # hf activation checkpoint warning
    "`resume_download` is deprecated",  # required because of upstream usage
    "`is_compiling` is deprecated",  # required with `transformers` 4.41.2 and PT nightly 20240601
    "The `use_auth_token` argument is deprecated",  # TODO: need to use `token` instead of `use_auth_token`
]

CORE_CTX_WARNS =  EXPECTED_WARNS + HF_EXPECTED_WARNS + [
    dummy_method_warn_fingerprint,  # expected in a core context with modules that use dummy log methods
]

LIGHTING_CTX_WARNS = HF_EXPECTED_WARNS + EXPECTED_WARNS + [
    "does not have many workers",
    "GPU available but",
    "is smaller than the logging interval",
]

TL_EXPECTED_WARNS = [
    "to transform pretrained weights on cpu",  # to support transforming weights on cpu prior to loading to device
    "dtype was not provided. Setting",  # for our HF/TL dtype inference/synchronization warnings
    "Setting both to the specified TL dtype",  # another variant of HF/TL dtype inference/synchronization warning
    "Since an `ITLensCustomConfig` has been provided",  # for custom transformerlens configs
    "Interpretune manages the HF model instantiation via `model_name_or_path`",  # using cust tlens config for fallback
    "Since no datamodule",  # using cust tlens config for fallback
]

TL_CTX_WARNS = TL_EXPECTED_WARNS + CORE_CTX_WARNS
TL_LIGHTNING_CTX_WARNS = TL_CTX_WARNS + LIGHTING_CTX_WARNS

SL_EXPECTED_WARNS = [
    "interactive_bk attribute",  # temporarily triggered in pydevd context
    "open_text is deprecated",
    "SAE has non-empty model_from_pretrained_kwargs",
]

SL_CTX_WARNS = SL_EXPECTED_WARNS + CORE_CTX_WARNS
SL_LIGHTNING_CTX_WARNS = SL_CTX_WARNS + LIGHTING_CTX_WARNS

FTS_CTX_WARNS = [".*currently depends upon.*", "No monitor metric specified for.*",]

EXAMPLE_WARNS = EXPECTED_WARNS + HF_EXPECTED_WARNS + TL_EXPECTED_WARNS

CLI_EXPECTED_WARNS = {
    # cli_adapter, *adapter_ctx
    (Adapter.core, Adapter.core): CORE_CTX_WARNS,
    (Adapter.lightning, Adapter.lightning): LIGHTING_CTX_WARNS,
    (Adapter.lightning, Adapter.lightning, Adapter.transformer_lens): TL_LIGHTNING_CTX_WARNS,
    (Adapter.core, Adapter.core, Adapter.transformer_lens): TL_CTX_WARNS
}

MIN_VERSION_WARNS = "2.2"
MAX_VERSION_WARNS = "2.5"
# torch version-specific warns go here
EXPECTED_VERSION_WARNS = {MIN_VERSION_WARNS: [],
                          MAX_VERSION_WARNS: [
                              ]}
torch_version = get_distribution("torch").version
extended_torch_ver = EXTENDED_VER_PAT.match(torch_version).group() or torch_version
if Version(extended_torch_ver) < Version(MAX_VERSION_WARNS):
    EXPECTED_WARNS.extend(EXPECTED_VERSION_WARNS[MIN_VERSION_WARNS])
else:
    EXPECTED_WARNS.extend(EXPECTED_VERSION_WARNS[MAX_VERSION_WARNS])
ADV_EXPECTED_WARNS = EXPECTED_WARNS + ["Found an `init_pg_lrs` key"]


def multiwarn_check(
    rec_warns: List, expected_warns: List | str, expected_mode: bool = False
) -> List[Optional[WarningMessage]]:
    if isinstance(expected_warns, str):
        expected_warns = [expected_warns]
    msg_search = lambda w1, w2: re.compile(w1).search(w2.message.args[0])  # noqa: E731
    if expected_mode:  # we're directed to check that multiple expected warns are obtained
        return [w_msg for w_msg in expected_warns if not any([msg_search(w_msg, w) for w in rec_warns])]
    else:  # by default we're checking that no unexpected warns are obtained
        return [w_msg for w_msg in rec_warns if not any([msg_search(w, w_msg) for w in expected_warns])]


unexpected_warns = partial(multiwarn_check, expected_mode=False)


unmatched_warns = partial(multiwarn_check, expected_mode=True)
