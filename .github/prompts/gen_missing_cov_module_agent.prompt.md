---
mode: "agent"
tools: ["githubRepo", "codebase", "file_search", "semantic_search", "read_file", "insert_edit_into_file", "create_file"]
description: "Generate code to add missing test coverage for a source module to a test module"
---

### What to change

- We only want to add or extend tests in the target test module to fully cover the missing lines of the current implementation of our source code module.
- We do not want to make any changes to implementations in the source code module at this point,

### New/Existing Fixture Usage

Our preference is to maximize test fidelity and intuitiveness over performance so we usually prefer using small real
objects (except for trivial edge cases), either by using a new fixture or usually by using existing fixtures defined in [conftest](../../tests/conftest.py) to
perform the tests.

It also may make sense to add new fixtures that can be reused to [conftest](../../tests/conftest.py).

#### Existing Fixture examples

Some examples of generated fixtures are below (you can examine [conftest](../../tests/conftest.py)) for more details.
All modules use the default #FingerprintTestITDataModule datamodule unless otherwise specified.

- these fixtures return #AnalysisSessionFixture objects, which include real #AnalysisStore objects useful for many tests.
  - "get_analysis_session\_\_sl_gpt2_logit_diffs_base\_\_initonly_runanalysis": uses a SAELensModule composed user module with GPT2 model to run the "logit_diffs_base" composite operation
  - "get_analysis_session\_\_sl_gpt2_logit_diffs_sae\_\_initonly_runanalysis": uses a SAELensModule composed user module with GPT2 model to run the "logit_diffs_sae" composite operation
  - "get_analysis_session\_\_sl_gpt2_logit_diffs_attr_grad\_\_initonly_runanalysis": uses a SAELensModule user composed module with GPT2 model to run the "logit_diffs_attr_grad" composite operation
  - "get_analysis_session\_\_sl_gpt2_logit_diffs_attr_ablation\_\_initonly_runanalysis": uses a SAELensModule user composed module with GPT2 model to run the "logit_diffs_attr_ablation" composite operation
- these fixtures return regular #ITSessionFixture objects
  - "get_it_session\_\_l_sl_gpt2\_\_initonly": Creates an ITSession with SAELensModule and LightningModule composed user module with the GPT2 model and does not run it hooks (e.g. prepare_data, setup)
  - "get_it_session\_\_tl_gpt2_debug\_\_setup": Creates an ITSession with an ITLensModule composed user module with the GPT2 model and runs the it hooks >= #FixtPhase.setup
  - "get_it_session\_\_core_cust\_\_setup": Creates an ITSession with a basic ITModule and uses a custom model and runs the it hooks >= #FixtPhase.setup
  - get_it_session\_\_core_cust_force_prepare\_\_initonly: Creates an ITSession basic ITModule and uses a custom model and does not run it hooks but configures the datamodule to avoid caching data
- these fixtures return ITSessionCfg objects
  - "get_it_session_cfg\_\_tl_cust": Returns an #ITSessionCfg object only, configured for ITLensModule and custom model usage
  - "get_it_session_cfg\_\_sl_gpt2": Returns an #ITSessionCfg object only, configured for SAELensModule and GPT2 model usage
- this fixtures returns an ITModule objects
  - "get_it_module\_\_core_cust\_\_setup": Returns an #ITModule object that has had its hooks >= #FixtPhase.setup executed

Invoking these existing fixtures can be either direct or indirect (via the request fixture).

For instance, the test
`tests/core/test_analysis_core.py::TestSAEAnalysisDict::test_core_sae_analysis_dict[sl_gpt2_logit_diffs_sae]`
uses a parameterized fixture (`get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis`,
requested via the `request` fixture) that is a session-scoped fixture and returns a #AnalysisStore object. For
session-level fixtures like these, we may want to `deepcopy` the #AnalysisStore that is returned
by the fixture so we don't interfere with other tests.

You can also use these session-level fixtures directly (e.g. for tests that all use the same fixture so don't need
to request specific fixtures based on parameterized values). For example, see the test
`tests/core/test_analysis_core.py::TestAnalysisStore::test_select_columns` for another example of how to properly
request the `get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis` fixture and handle the
#AnalysisSessionFixture artifacts.

### Execution guidance

To run all tests in the target test module:

```bash
cd ~/repos/interpretune && source ~/.venvs/it_latest/bin/activate && python -m pytest tests/core/<test code module name> -v
```

If tests are failing, try to determine why the tests are failing, pausing if the issue would seem to require changes to the implementation code.

Once all tests are passing, you may run the test suite and append to existing coverage with the command below.

Command to append coverage and validate expected lines are hit (or find remaining set of missing lines) once tests are passing:

```bash
cd ${HOME}/repos/interpretune && source ~/.venvs/it_latest/bin/activate && python -m coverage run --append --source src/interpretune -m pytest tests/core/<test code module name>  -v && python -m coverage report -m --include=`<source code module relative path>`.py
```

A concrete example:

```bash
cd ${HOME}/repos/interpretune && source ~/.venvs/it_latest/bin/activate && python -m coverage run --append --source src/interpretune -m pytest tests/core/test_analysis_core.py  -v && python -m coverage report -m --include=src/interpretune/analysis/core.py
```

After collecting updated coverage, if there are no more missing coverage lines, the task is complete.
If there are still missing coverage lines, generate another set of test module updates to cover those lines and re-run the test suite.

### Systematic Coverage Increase Approach

After your initial pass at trying to cover all lines, try updating coverage one source module function at a time.
You should be able to append to existing coverage using only the tests you add or update in the test module which should allow more reliable iteration and enable us to validate we are making progress covering lines. For clarity, progress systematically, lowest line numbers first, source module function/method by function/method. If you are unable to collect test output, indicate that is the case to the user and pause since we don't want to iterate on the test code if we can't collect test output.

### Target Source and Test Modules with starting Coverage State

Following this prompt, the user will provide following additional information to guide the generation of the test code:

- Test code module to update: `${input:test_code_module_name}`

- Source code module with missing lines: `${input:source_code_module_name}`

- List of line numbers and ranges that are currently missing coverage for `${source_code_module_name}` (e.g. 1-2, 3, 30-33):

`${input:missing_coverage_lines}`
