---
mode: "edit"
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

  - "get_it_session\_\_sl_gpt2_analysis\_\_setup": This fixture is a session-scoped fixture that creates an ITSession with a SAELensModule composed user module with the GPT2 model and runs the it hooks >= #FixtPhase.setup. The ITSession is
    preconfigured for analysis tasks (e.g. it has sae_analysis_targets defined) and is especially useful for tests that involve analysis operations. You can examine `tests/unit/test_analysis_ops_definitions.py::TestAnalysisOperationsImplementations::test_op_serialization` for example usage.

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
`tests/unit/test_analysis_core.py::TestSAEAnalysisDict::test_core_sae_analysis_dict[sl_gpt2_logit_diffs_sae]`
uses a parameterized fixture (`get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis`,
requested via the `request` fixture) that is a session-scoped fixture and returns a #AnalysisStore object. For
session-level fixtures like these, we may want to `deepcopy` the #AnalysisStore that is returned
by the fixture so we don't interfere with other tests.

You can also use these session-level fixtures directly (e.g. for tests that all use the same fixture so don't need
to request specific fixtures based on parameterized values). For example, see the test
`tests/unit/test_analysis_core.py::TestAnalysisStore::test_select_columns` for another example of how to properly
request the `get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis` fixture and handle the
#AnalysisSessionFixture artifacts.

If you want to test a specific analysis operation, it's often useful to use the #run_op_with_config function to run the operation with an #OpTestConfig object. See `tests/unit/test_analysis_ops_definitions.py::TestAnalysisOperationsImplementations::test_op_serialization` for example usage on various ops defined in #SERIALIZATION_TEST_CONFIGS

### Target Source and Test Modules with starting Coverage State

Following this prompt, the user will provide following additional information to guide the generation of the test code:

- Test code module to update: `${input:test_code_module_name}`

- Source code module with missing lines: `${input:source_code_module_name}`

- List of line numbers and ranges that are currently missing coverage for `${source_code_module_name}` (e.g. 1-2, 3, 30-33):

`${input:missing_coverage_lines}`
