# Test Coverage Redundancy Analyzer

This tool analyzes your test suite to identify redundant tests that can be removed without reducing overall code coverage.

## Overview

The test coverage redundancy analyzer uses coverage.py's dynamic contexts feature to track which lines of code are covered by which tests. By analyzing this data, it can:

1. Identify "essential tests" that uniquely cover at least one line of code
2. Find tests that only cover lines already covered by other tests
3. Use a greedy algorithm to find a minimal set of tests that maintain full coverage
4. Generate reports to help you decide which tests to keep or remove

## How It Works

The analyzer works in several steps:

1. Sets up a `.coveragerc` file configured to use dynamic contexts  (via the pytest-cov plugin)
2. Runs your test suite (both regular pytest tests and special tests) with coverage tracking
3. Analyzes the coverage database to extract line-to-test mappings
4. Identifies essential tests that cannot be removed without losing coverage
5. Ranks remaining tests by their redundancy (how much of their coverage is also provided by other tests)
6. Uses a greedy algorithm to find a minimal test set that maintains full coverage
7. Generates reports with findings and recommendations

## Usage

### Run the Analysis

```bash
# Run with default settings
./scripts/analyze_test_redundancy.sh

# Run with custom output directory
./scripts/analyze_test_redundancy.sh --output-dir=/tmp/test_analysis

# Limit to top 50 candidates for removal
./scripts/analyze_test_redundancy.sh --max-candidates=50

# Run with subset of tests (filter expressions)
./scripts/analyze_test_redundancy.sh --normal-subset="test_feature1 or test_feature2"

# Run with complex filter patterns including brackets (make sure to quote properly)
./scripts/analyze_test_redundancy.sh --normal-subset="test_core[param1] or test_utils[param2]"

# Just show what would be run (dry run)
./scripts/analyze_test_redundancy.sh --dryrun
```

### Review Results

After running the analysis, you'll find these files in the output directory:

- `test_coverage_analysis_report.txt`: Summary report with key findings
- `essential_tests.json`: Tests that uniquely cover at least one line
- `removal_candidates.json`: Tests that could potentially be removed
- `minimization_stats.json`: Results of the greedy test minimization
- `test_to_lines.json`: Mapping of which tests cover which lines
- `line_to_tests.json`: Mapping of which lines are covered by which tests
- `unique_coverage.json`: Lines uniquely covered by each essential test

### Interpreting Results

The main report (`test_coverage_analysis_report.txt`) provides:

1. Statistics about your test suite
2. A list of top candidates for removal, ranked by redundancy
3. Results of the greedy minimization algorithm
4. Recommendations for next steps

Tests are ranked for removal based on:
- **Redundancy ratio**: How much of the test's coverage is also covered by other tests
- **Lines covered**: Fewer lines covered means less impact if removed
- **Unique lines**: Tests with zero unique lines are prime candidates for removal

## Implementation Details

This tool consists of two main components:

1. `analyze_test_coverage.py`: The Python script that performs the analysis
2. `analyze_test_redundancy.sh`: A shell wrapper script for convenient execution

The Python script integrates with your existing test infrastructure by:
- Supporting both regular pytest tests and special marker-based tests
- Using coverage.py's dynamic contexts to associate covered lines with specific tests
- Analyzing the SQLite coverage database to extract detailed coverage information

## Recommendations for Test Removal

When considering test removal:

1. Start with tests that have 100% redundancy (all lines covered by other tests)
2. Verify removal doesn't affect coverage using targeted testing
3. Consider the value of tests beyond just line coverage (e.g., assertion quality)
4. Remove tests in small batches and rerun the analysis to ensure coverage is maintained

## Limitations

- The analysis focuses on line coverage only, not branch coverage or assertion quality
- Special test marker extraction might need adjustment based on your specific test organization
- For very large test suites, the analysis may take significant time and memory
- Some tests might still have value beyond coverage (e.g., complex state validations)

## Advanced Usage

The Python script can be used directly for more advanced scenarios:

```bash
python scripts/analyze_test_coverage.py --help
```

You can also integrate this analysis into your CI/CD pipeline to track test redundancy over time.
