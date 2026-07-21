# Configuration Guidance

Practical guidance for composing interpretune configurations. See
{doc}`usage/session_module_datamodule_usage` for the session/config API itself.

## Keep the composition space flat

Interpretune deliberately keeps configuration composition **flat**: configs compose by explicit
inclusion/override rather than deep implicit inheritance hierarchies. When authoring new configs,
prefer adding a sibling config (or a small override file) over introducing new nesting levels —
flatness is what keeps the composition space legible and greppable as the config surface grows.

## Group configs by model, then experiment

Configuration files can technically be composed arbitrarily, but the convention used throughout
the examples — and recommended for your own experiments — is grouping by **model level** (shared
per-model substrate settings) and **experiment level** (task/experiment-specific settings layered
on a model config). See `src/it_examples/config/experiments/` for the reference layout
(e.g. `rte_boolq/gemma2/...`, `rte_boolq/gemma3/...`).

## HookedTransformer: `model_name_or_path` can be arbitrary

When instantiating a TransformerLens `HookedTransformer` via interpretune config, the ITLens
config's `model_name_or_path` can be set to an arbitrary identifier — the TL pretrained-model
resolution is driven by the TL config's own model selection, so `model_name_or_path` functions as
a label rather than a required HF-hub pointer in that mode.

## CLI: `auto_comp_cfg` is not supported via jsonargparse

The `auto_comp_cfg` mechanism (automatic composition-config resolution,
`interpretune.config.shared.AutoCompConfig`) is **not supported through the CLI**: jsonargparse's
signature-driven parsing cannot represent its dynamic target-class rebinding. Configure
`auto_comp_cfg` programmatically (or via YAML consumed by the session API) instead of CLI flags.

## Lightning precision plugins only convert floating-point tensors

When composing with the Lightning adapter, note that Lightning precision plugins convert inputs
via `apply_to_collection(..., function=...)` targeting **floating-point tensors only**. Integer
tensors — token IDs, attention masks, embedding indices (`LongTensor`) — pass through unconverted,
which is exactly what embedding layers require. If you see mixed-precision surprises, audit the
float tensors in your batch; the integer inputs are not the issue.
