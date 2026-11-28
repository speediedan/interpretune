# About These Examples

The runnable example notebooks, the published components they load, and the dashboard pipeline configurations live in
the [interpretune/examples](https://github.com/interpretune/examples) repository, and its documentation renders each
notebook from a copy executed ahead of time: see the
[examples gallery](https://interpretune.org/projects/examples/) (a subproject of these docs).

Each notebook there says what it needs: optional adapters (TransformerLens, NNsight, SAELens, circuit-tracer), a GPU,
and in some cases gated model weights. The installs it names come from this repository, which defines the `examples`
extra and the `git-deps` group:

```bash
uv pip install -e ".[examples]"                    # CPU quickstart, SAELens tutorial and the hub-only notebooks
uv pip install -e ".[examples]" --group git-deps   # the circuit-tracer notebooks
```

See {doc}`usage/developer_multi_repo_setup` for environment setup. New here? The CPU quickstart needs no GPU, token or
gated weights, and the {doc}`adapter-selection guide <usage/adapter_selection_guide>` says which adapter combination
fits your task.
