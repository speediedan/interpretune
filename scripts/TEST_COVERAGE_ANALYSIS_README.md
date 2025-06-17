# Test Coverage Redundancy Analyzer

This tool analyzes your test suite to identify redundant tests that can be removed without reducing overall code coverage.

## Overview

The test coverage redundancy analyzer uses coverage.py's dynamic contexts feature to track which lines of code and which branches are covered by which tests. By analyzing this data, it can:

1. Identify "base tests" (at statement and branch levels) that uniquely cover at least one line of code or one branch.
2. Find tests that only cover lines or branches already covered by other tests.
3. Use a greedy algorithm to find a minimal set of tests that maintain full statement and branch coverage.
4. Generate reports to help you decide which tests to keep or remove.

## How It Works

The analyzer works in several steps:

1. Sets up a `.coveragerc` file configured to use dynamic contexts and collect branch coverage (via the pytest-cov plugin)
2. Runs your test suite (both regular pytest tests and special tests) with coverage tracking (including branch coverage)
3. Analyzes the coverage database to extract line-to-test and branch-to-test mappings
4. Identifies base tests (for both statement and branch coverage) that cannot be removed without losing coverage
5. Ranks remaining tests by their redundancy (how much of their statement and branch coverage is also provided by other tests)
6. Uses a greedy algorithm to find a minimal test set that maintains full statement and branch coverage
7. Generates reports with findings and recommendations

## Usage

### Run the Analysis

```bash
# Run with default settings (statement-level analysis)
./scripts/analyze_coverage_overlap.sh

# Run with branch-level analysis
./scripts/analyze_coverage_overlap.sh --branch-level

# Run with custom output directory
./scripts/analyze_coverage_overlap.sh --output-dir=/tmp/test_analysis

# Limit to top 50 candidates for removal
./scripts/analyze_coverage_overlap.sh --max-candidates=50

# Run with subset of tests (filter expressions)
./scripts/analyze_coverage_overlap.sh --normal-subset="test_feature1 or test_feature2"

# Run with complex filter patterns including brackets (make sure to quote properly)
./scripts/analyze_coverage_overlap.sh --normal-subset="test_core[param1] or test_utils[param2]"

# Run analysis for specific test subsets for different mark types
./scripts/analyze_coverage_overlap.sh \
  --normal-subset="test_core_feature" \
  --standalone-subset="test_standalone_utility" \
  --profile-ci-subset="test_ci_profile_critical" \
  --profile-subset="test_specific_profile_scenario" \
  --optional-subset="test_optional_integration"

# Specify which mark types to run (default is "normal,standalone,profile_ci")
# This example runs only normal and profile tests:
./scripts/analyze_coverage_overlap.sh --mark-types-to-run="normal,profile"

# If a subset is provided for a mark type not in --mark-types-to-run,
# that mark type will still be run with a warning.
# Example: only 'normal' is specified, but '--profile-subset' is also given.
# 'profile' tests matching the subset will run.
./scripts/analyze_coverage_overlap.sh --mark-types-to-run="normal" --profile-subset="test_important_profile"

# Run branch-level analysis with specific test subsets
./scripts/analyze_coverage_overlap.sh --branch-level --normal-subset="test_core_feature"

# Just show what would be run (dry run)
./scripts/analyze_coverage_overlap.sh --dryrun
```

### Review Results

After running the analysis, you'll find several files in the output directory. The exact set of files will depend on whether you ran with the default statement-level analysis or with `--branch-level` enabled.

#### Files generated in both modes:
- `test_coverage_analysis_report.txt`: Summary report with key findings for the selected analysis mode.
- `coverage.json`: JSON format coverage report with detailed information.
- `html_report/`: Directory containing interactive HTML coverage reports.

#### Files generated in statement-level mode (default):
- `statement_level_base.json`: Tests that uniquely cover at least one line of code.
- `statement_level_redundant.json`: Tests whose statement coverage could potentially be redundant.
- `statement_minimization_stats.json`: Results of the greedy test minimization for statement coverage.
- `test_to_lines.json`: Mapping of which tests cover which lines (statements).
- `line_to_tests.json`: Mapping of which lines (statements) are covered by which tests.
- `unique_statement_coverage.json`: Lines uniquely covered by each statement-level base test.

#### Files generated in branch-level mode (with `--branch-level`):
- `branch_level_base.json`: Tests that uniquely cover at least one branch.
- `branch_level_redundant.json`: Tests whose branch coverage could potentially be redundant.
- `branch_minimization_stats.json`: Results of the greedy test minimization for branch coverage.
- `test_to_branches.json`: Mapping of which tests cover which branches.
- `branch_to_tests.json`: Mapping of which branches are covered by which tests.
- `unique_branch_coverage.json`: Branches uniquely covered by each branch-level base test.

The HTML report provides an interactive view of your source code with coverage highlighting, making it easy to identify which parts of your code are covered by tests. The JSON report provides machine-readable coverage data that can be used for further processing or integration with other tools.

### Interpreting Results

The main report (`test_coverage_analysis_report.txt`) provides:

1. Statistics about your test suite (including statement and branch coverage metrics).
2. Lists of top candidates for removal, ranked by redundancy (for both statement and branch coverage).
3. Results of the greedy minimization algorithm (for both statement and branch coverage).
4. Recommendations for next steps

Tests are ranked for removal based on:
- **Redundancy ratio**: How much of the test's coverage is also covered by other tests
- **Lines/branches covered**: Fewer lines/branches covered means less impact if removed
- **Unique lines/branches**: Tests with zero unique coverage are prime candidates for removal

## Statement vs. Branch Coverage

The tool supports two coverage analysis modes:

1. **Statement Coverage (default)** - Tracks which lines of code are executed by which tests.
2. **Branch Coverage** - Tracks which branches (code paths) are taken by which tests.

Branch coverage is generally considered more thorough than statement coverage because it ensures that all possible code paths are tested, not just that each line is executed at least once. For example, branch coverage can detect if a conditional path (else branch) is never taken.

When using branch-level analysis:
- Coverage data includes source-to-destination line pairs representing branches
- base tests are those that uniquely cover at least one branch
- The analysis considers redundancy in terms of branches rather than statements

## Implementation Details

This tool consists of two main components:

1. `analyze_test_coverage.py`: The Python script that performs the analysis
2. `analyze_coverage_overlap.sh`: A shell wrapper script for convenient execution

The Python script integrates with your existing test infrastructure by:
- Supporting both regular pytest tests and special marker-based tests
- Using coverage.py's dynamic contexts to associate covered lines with specific tests
- Analyzing the SQLite coverage database to extract detailed coverage information

## Recommendations for Test Removal

When considering test removal:

1. Start with tests that have 100% redundancy (all lines/branches covered by other tests)
2. Verify removal doesn't affect coverage using targeted testing
3. Consider the value of tests beyond just line coverage (e.g., assertion quality)
4. Remove tests in small batches and rerun the analysis to ensure coverage is maintained

## Limitations

- The analysis focuses on line and branch coverage, not assertion quality directly.
- Special test marker extraction might need adjustment based on your specific test organization
- For very large test suites, the analysis may take significant time and memory
- Some tests might still have value beyond coverage (e.g., complex state validations)

## Advanced Usage

The Python script can be used directly for more advanced scenarios:

```bash
python scripts/analyze_test_coverage.py --help
```

You can also integrate this analysis into your CI/CD pipeline to track test redundancy over time.
