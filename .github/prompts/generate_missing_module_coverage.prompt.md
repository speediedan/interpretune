Without making any changes to implementations in the code base at this point, we want to add or extend tests in the
target test module to fully cover the missing lines of the current implementation of our source code module.

If may make sense to add new fixtures that can be reused to [conftest](../../tests/conftest.py).
Our preference is to maximize test fidelity and intuitiveness over performance so it is often best to use a small real
object, either by using a new fixture or using existing fixtures defined in [conftest](../../tests/conftest.py) to
perform the tests.

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

Command to run tests:

```bash
cd ~/repos/interpretune && source ~/.venvs/it_latest/bin/activate && python -m pytest tests/unit/<test code module name> -v
```

If tests are failing, pause iteration and ask for user intervention to help determine why the tests are failing before trying to fix them. Do NOT iterate trying to fix tests, ask for user feedback before re-running the test suite.

If all tests are passing, you may run the test suite and append to existing coverage with the command below.

Command to append coverage and validate expected lines are hit (or find remaining set of missing lines) once tests are passing
cd ${HOME}/repos/interpretune && source ~/.venvs/it_latest/bin/activate && python -m coverage run --append --source src/interpretune -m pytest tests/unit/<test code module name> -v && python -m coverage report -m --include=`<source code module relative path>`.py

A concrete example:

```bash
cd ${HOME}/repos/interpretune && source ~/.venvs/it_latest/bin/activate && python -m coverage run --append --source src/interpretune -m pytest tests/unit/test_analysis_core.py  -v && python -m coverage report -m --include=src/interpretune/analysis/core.py
```

After collecting updated coverage, if there are no more missing coverage lines, the task is complete. If there are still missing coverage lines, generate another set of test module updates to cover those lines and re-run the test suite (again, pause if there are test errors and ask for feedback).

Following this prompt, you will find the following additional information (example line numbers are below):

- Test code module to update: `<test code module name>`.py

- Source code module with missing lines: `<source code module name>`.py

- Lines that are currently missing coverage for `<source code module name>`.py:

`<comma-separated list of line numbers and line number ranges>`
