## Profiling and using ``scripts/speedscope_top_packages.py``

The following examples demonstrate the production of `py-spy` profiling artifacts often useful in dev/test analysis
as well as the use of `scripts/speedscope_top_packages.py` to parse and sample stacks for agentic analysis or human inspection.

Notes:
- All commands assume you have activated the relevant Python virtual environment and are in the repository top-level directory. These examples are used for analyzing pytest startup time but can be adapted for other commands.

1) Manually generate a flamegraph of pytest startup (collect-only):

```bash
time py-spy record --subprocesses -o ${HOME}/repos/interpretune/tests/profiling_artifacts/pytest_startup_collect_only.svg -- python -m pytest src/interpretune tests -q --collect-only
```

2) Manually generate a speedscope-format JSON that can be uploaded to speedscope.app:

a) generate the speedscope json file (use --nonblocking for more accurate time estimates; remove that flag to avoid sampling errors at the cost of overhead):

```bash
d=`date +%Y%m%d%H%M%S`
time py-spy record --subprocesses --nonblocking -o ${HOME}/repos/interpretune/tests/profiling_artifacts/pytest_startup_collect_only_speedscope_${d}.json --format speedscope -- python -m pytest src/interpretune tests -q --collect-only
```

b) upload the speedscope json to the remote webapp: https://www.speedscope.app/ (or for offline viewing install speedscope locally via npm: `npm install -g speedscope`). See https://github.com/jlfwong/speedscope/wiki/Importing-from-py-spy-(python)

3) Manually generate a raw collection-format capture:

```bash
time py-spy record --subprocesses -o ${HOME}/repos/interpretune/tests/profiling_artifacts/pytest_startup_collect_only_raw.json --format raw -- python -m pytest src/interpretune tests -q --collect-only
```

4) Parse top packages appearing in a speedscope JSON (useful for agentic or manual analysis):

```bash
python scripts/speedscope_top_packages.py /home/speediedan/repos/interpretune/tests/profiling_artifacts/pytest_startup_collect_only_speedscope_20250910113803.json -p transformer_lens sae_lens torch transformers lightning IPython wandb
```

Example output (summary):

```
| metric        |   value |
|---------------|---------|
| frames_count  |    1816 |
| total_samples |     936 |

| package          |   samples | pct   | top_files                                                                                                |
|------------------|-----------|-------|----------------------------------------------------------------------------------------------------------|
| transformer_lens |       120 | 12.8% | <frozen importlib._bootstrap> (12074), <frozen importlib._bootstrap_external> (2446), __init__.py (1320) |
| sae_lens         |       133 | 14.2% | <frozen importlib._bootstrap> (12678), <frozen importlib._bootstrap_external> (2571), __init__.py (1463) |
| torch            |       344 | 36.8% | <frozen importlib._bootstrap> (30790), <frozen importlib._bootstrap_external> (6110), __init__.py (3718) |
| transformers     |       131 | 14.0% | <frozen importlib._bootstrap> (9825), <frozen importlib._bootstrap_external> (1881), __init__.py (1430)  |
| lightning        |       170 | 18.2% | <frozen importlib._bootstrap> (23143), <frozen importlib._bootstrap_external> (4594), __init__.py (1870) |
| IPython          |         7 | 0.7%  | <frozen importlib._bootstrap> (398), <frozen importlib._bootstrap_external> (83), __init__.py (77)       |
| wandb            |        63 | 6.7%  | <frozen importlib._bootstrap> (7137), <frozen importlib._bootstrap_external> (1469), __init__.py (671)   |
```

5) Parse speedscope stacks and sample examples (filter by frame name and limit examples):

```bash
cd /home/speediedan/repos/interpretune && \
source ~/.venvs/it_latest/bin/activate && \
d=`date +%Y%m%d%H%M%S` && \
python scripts/speedscope_top_packages.py /home/speediedan/repos/interpretune/tests/profiling_artifacts/pytest_startup_collect_only_speedscope_20250910113803.json -p transformer_lens sae_lens torch transformers lightning IPython wandb --sample-stacks --max-examples 2 --name-filter "<module>" > /tmp/speedscope_summary_${d}.out
```

Example sample output (for `torch`):

```
Sample stacks for package: torch
| # | pct total | pct package | stack                                                                                                |
|---|-----------|-------------|------------------------------------------------------------------------------------------------------|
| 1 | 3.31%     | 9.1%        | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/pytest/__main__.py:9)       |
|   |           |             | <module> (/home/speediedan/repos/interpretune/tests/__init__.py:17)                                  |
|   |           |             | <module> (/home/speediedan/repos/interpretune/src/interpretune/__init__.py:21)                       |
|   |           |             | <module> (/home/speediedan/repos/interpretune/src/interpretune/protocol.py:27)                       |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/torch/__init__.py:416)      |
| 2 | 0.11%     | 0.3%        | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/pytest/__main__.py:9)       |
|   |           |             | <module> (/home/speediedan/repos/interpretune/tests/__init__.py:17)                                  |
|   |           |             | <module> (/home/speediedan/repos/interpretune/src/interpretune/__init__.py:21)                       |
|   |           |             | <module> (/home/speediedan/repos/interpretune/src/interpretune/protocol.py:27)                       |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/torch/__init__.py:1837)     |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/torch/_tensor.py:21)        |
```

6) Filter unique stacks by a substring that appears in the stack lines (e.g. to focus on the various `transformer_lens` init codepaths for both for `transformer_lens` and `sae_lens`):

```bash
cd /home/speediedan/repos/interpretune && \
source ~/.venvs/it_latest/bin/activate && \
d=`date +%Y%m%d%H%M%S` && \
python scripts/speedscope_top_packages.py /home/speediedan/repos/interpretune/tests/profiling_artifacts/pytest_startup_collect_only_speedscope_20250910113803.json -p transformer_lens sae_lens --sample-stacks --max-examples 5 --name-filter "<module>" --filter-uniq "transformer_lens/__init__.py" > /tmp/speedscope_summary_${d}.out
```

Example output excerpt:

```
| metric        |   value |
|---------------|---------|
| frames_count  |    1816 |
| total_samples |     936 |

| package          |   samples | pct   | top_files                                                                                                |
|------------------|-----------|-------|----------------------------------------------------------------------------------------------------------|
| transformer_lens |       120 | 12.8% | <frozen importlib._bootstrap> (12074), <frozen importlib._bootstrap_external> (2446), __init__.py (1320) |
| sae_lens         |       133 | 14.2% | <frozen importlib._bootstrap> (12678), <frozen importlib._bootstrap_external> (2571), __init__.py (1463) |

Sample stacks for package: transformer_lens
| # | pct total | pct package | stack                                                                                                                              |
|---|-----------|-------------|------------------------------------------------------------------------------------------------------------------------------------|
| 1 | 6.52%     | 50.8%       | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/pytest/__main__.py:9)                                     |
|   |           |             | <module> (/home/speediedan/repos/interpretune/tests/__init__.py:17)                                                                |
|   |           |             | <module> (/home/speediedan/repos/interpretune/src/interpretune/utils/__init__.py:2)                                                |
|   |           |             | <module> (/home/speediedan/repos/interpretune/src/interpretune/utils/import_utils.py:176)                                          |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/__init__.py:8)                                   |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/saes/__init__.py:1)                              |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/saes/batchtopk_sae.py:8)                         |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/saes/jumprelu_sae.py:10)                         |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/saes/sae.py:27)                                  |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/transformer_lens/__init__.py:21)                          |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/transformer_lens/train.py:12)                             |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/__init__.py:22)                                     |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/sdk/__init__.py:25)                                 |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/sdk/artifacts/artifact.py:30)                       |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/data_types.py:16)                                   |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/sdk/data_types/audio.py:11)                         |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/sdk/data_types/base_types/media.py:13)              |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/sdk/data_types/base_types/wb_value.py:4)            |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/sdk/wandb_setup.py:38)                              |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/sdk/wandb_settings.py:23)                           |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/pydantic/main.py:36)                                      |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/pydantic/_internal/_decorators.py:29)                     |
| 2 | 2.99%     | 23.3%       | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/pytest/__main__.py:9)                                     |
|   |           |             | <module> (/home/speediedan/repos/interpretune/tests/__init__.py:17)                                                                |
|   |           |             | <module> (/home/speediedan/repos/interpretune/src/interpretune/utils/__init__.py:2)                                                |
|   |           |             | <module> (/home/speediedan/repos/interpretune/src/interpretune/utils/import_utils.py:176)                                          |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/__init__.py:8)                                   |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/saes/__init__.py:1)                              |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/saes/batchtopk_sae.py:8)                         |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/saes/jumprelu_sae.py:10)                         |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/saes/sae.py:27)                                  |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/transformer_lens/__init__.py:1)                           |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/transformer_lens/hook_points.py:20)                       |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/transformer_lens/utils.py:23)                             |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/datasets/__init__.py:17)                                  |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/datasets/arrow_dataset.py:75)                             |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/datasets/arrow_reader.py:30)                              |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/datasets/download/__init__.py:9)                          |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/datasets/download/download_manager.py:32)                 |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/datasets/utils/file_utils.py:48)                          |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/aiohttp/__init__.py:6)                                    |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/aiohttp/client.py:38)                                     |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/yarl/__init__.py:2)                                       |
| 3 | 2.46%     | 19.2%       | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/pytest/__main__.py:9)                                     |
...

Sample stacks for package: sae_lens
| # | pct total | pct package | stack                                                                                                                              |
|---|-----------|-------------|------------------------------------------------------------------------------------------------------------------------------------|
| 1 | 6.52%     | 45.9%       | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/pytest/__main__.py:9)                                     |
|   |           |             | <module> (/home/speediedan/repos/interpretune/tests/__init__.py:17)                                                                |
|   |           |             | <module> (/home/speediedan/repos/interpretune/src/interpretune/utils/__init__.py:2)                                                |
|   |           |             | <module> (/home/speediedan/repos/interpretune/src/interpretune/utils/import_utils.py:176)                                          |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/__init__.py:8)                                   |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/saes/__init__.py:1)                              |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/saes/batchtopk_sae.py:8)                         |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/saes/jumprelu_sae.py:10)                         |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/saes/sae.py:27)                                  |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/transformer_lens/__init__.py:21)                          |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/transformer_lens/train.py:12)                             |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/__init__.py:22)                                     |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/sdk/__init__.py:25)                                 |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/sdk/artifacts/artifact.py:30)                       |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/data_types.py:16)                                   |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/sdk/data_types/audio.py:11)                         |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/sdk/data_types/base_types/media.py:13)              |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/sdk/data_types/base_types/wb_value.py:4)            |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/sdk/wandb_setup.py:38)                              |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/wandb/sdk/wandb_settings.py:23)                           |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/pydantic/main.py:36)                                      |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/pydantic/_internal/_decorators.py:29)                     |
| 2 | 2.99%     | 21.1%       | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/pytest/__main__.py:9)                                     |
|...
| 5 | 0.21%     | 1.5%        | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/pytest/__main__.py:9)                                     |
|   |           |             | <module> (/home/speediedan/repos/interpretune/tests/__init__.py:17)                                                                |
|   |           |             | <module> (/home/speediedan/repos/interpretune/src/interpretune/utils/__init__.py:2)                                                |
|   |           |             | <module> (/home/speediedan/repos/interpretune/src/interpretune/utils/import_utils.py:176)                                          |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/__init__.py:8)                                   |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/saes/__init__.py:1)                              |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/saes/batchtopk_sae.py:8)                         |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/saes/jumprelu_sae.py:10)                         |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/sae_lens/saes/sae.py:27)                                  |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/transformer_lens/__init__.py:14)                          |
|   |           |             | <module> (/home/speediedan/.venvs/it_latest/lib/python3.12/site-packages/transformer_lens/SVDInterpreter.py:20)                    |
```
