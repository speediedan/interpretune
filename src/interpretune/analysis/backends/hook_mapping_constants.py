"""Constants shared by the hook-name resolver and the activation-point vocabulary (kept dependency-free)."""

from __future__ import annotations

#: SAE sub-hook suffixes that may trail a base hook name (``blocks.5.hook_resid_post.hook_sae_acts_post``).
_SAE_SUBHOOK_SUFFIXES = (
    "hook_sae_input",
    "hook_sae_acts_pre",
    "hook_sae_acts_post",
    "hook_sae_output",
    "hook_sae_error",
)
SUBHOOK_SUFFIXES = frozenset(_SAE_SUBHOOK_SUFFIXES)
