#!/usr/bin/env python3
"""Test Coverage Analyzer.

This script integrates with pytest and coverage.py to track which lines are covered by which tests,
then identifies tests that can be removed without reducing overall coverage.

The script uses coverage.py's dynamic contexts feature to associate covered lines with specific tests.
It can be run as a standalone script or integrated into your existing test infrastructure.

Usage:
    python analyze_test_coverage.py [--max-candidates N] [--output-dir PATH]
"""

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


class TestCoverageAnalyzer:
    """Analyzes test coverage and identifies redundant tests."""

    def __init__(self, output_dir=None, max_candidates=None, normal_subset=None,
                 standalone_subset=None, profile_ci_subset=None):
        """Initialize the analyzer with configuration options.

        Args:
            output_dir: Directory to store output files (default: current directory)
            max_candidates: Maximum number of candidate tests to report (default: no limit)
            normal_subset: Filter expression for normal pytest tests (e.g., "test_1 or test_2")
            standalone_subset: Filter expression for standalone tests (e.g., "test_3 or test_4")
            profile_ci_subset: Filter expression for profile_ci tests (e.g., "test_5 or test_6")
        """
        self.output_dir = Path(output_dir or '.')
        self.max_candidates = max_candidates
        self.normal_subset = normal_subset
        self.standalone_subset = standalone_subset
        self.profile_ci_subset = profile_ci_subset
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up paths for coverage data
        self.coverage_db = self.output_dir / '.coverage'
        self.coverage_config = self.output_dir / '.coveragerc'

        # Store analysis results
        self.test_to_lines = defaultdict(set)
        self.line_to_tests = defaultdict(set)
        self.all_covered_lines = set()
        self.all_tests = set()
        self.essential_tests = set()
        self.removal_candidates = []

    def setup_coverage_config(self):
        """Create a .coveragerc file for the analysis."""
        config_content = """
[run]
source = src/interpretune
data_file = {}

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
    pass
    raise ImportError

[html]
show_contexts = True
""".format(self.coverage_db)

        with open(self.coverage_config, 'w') as f:
            f.write(config_content.strip())
        print(f"Created coverage config at {self.coverage_config}")

    def collect_coverage_data(self):
        """Run pytest with coverage to collect test-specific coverage data."""
        # Ensure coverage data file is created in the output directory

        # Run regular tests with pytest-cov
        cmd = [
            "python", "-m", "pytest", "--cov", f"--cov-config={self.coverage_config}", "--cov-append",
            "--cov-context=test", "tests", "-v"
        ]

        # Add normal subset filter if specified
        if self.normal_subset:
            # Remove any surrounding quotes that might have been carried over from the shell
            normal_subset = self.normal_subset.strip("'\"")
            # Use the filter expression directly as a pytest -k expression
            cmd.extend(["-k", normal_subset])

        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # Run special tests with different mark types
        special_tests_mark_types = ["standalone", "profile_ci"]
        for mark_type in special_tests_mark_types:
            env = os.environ.copy()

            # Set up environment variables needed by special_tests.sh
            if mark_type == "standalone":
                env["IT_RUN_STANDALONE_TESTS"] = "1"
                subset = self.standalone_subset.strip("'\"") if self.standalone_subset else None
            elif mark_type == "profile_ci":
                env["IT_RUN_PROFILING_TESTS"] = "1"
                subset = self.profile_ci_subset.strip("'\"") if self.profile_ci_subset else None
            else:
                subset = None

            # Pass the coverage data file path as an environment variable for special_tests.sh
            env["COVERAGE_ANALYSIS_DATAFILE"] = str(self.coverage_config)
            cmd = ["bash", "tests/special_tests.sh", f"--mark_type={mark_type}"]

            # Add filter pattern if subset is specified
            if subset:
                cmd.extend([f"--filter_pattern={subset}"])

            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, env=env, check=True)

    def analyze_coverage_database(self):
        """Analyze the coverage database to extract line-to-test mappings."""
        if not self.coverage_db.exists():
            raise FileNotFoundError(f"Coverage database not found at {self.coverage_db}")

        conn = sqlite3.connect(self.coverage_db)
        # Enable column names in row factory to get results as dictionaries
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            # First, verify we can access the database at all
            cursor.execute("SELECT sqlite_version()")
            sqlite_version = cursor.fetchone()[0]
            print(f"Using SQLite version: {sqlite_version}")

            # Get all available tables to help diagnose the issue
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            print(f"Available tables: {', '.join(tables)}")

            # Check if line_bits table exists and get its schema
            if 'line_bits' not in tables:
                raise ValueError("line_bits table not found in the coverage database")

            cursor.execute("PRAGMA table_info(line_bits)")
            columns = cursor.fetchall()
            column_names = [column[1] for column in columns]
            print(f"line_bits columns: {', '.join(column_names)}")

            # Print all contexts to see their format
            cursor.execute("SELECT id, context FROM context")
            contexts_data = cursor.fetchall()
            print(f"Found {len(contexts_data)} contexts in the database")
            for i, row in enumerate(contexts_data):  # Print all contexts since there may be few
                print(f"Context {i}: {row['id']} -> {row['context']}")

            # Print some file info to see paths
            cursor.execute("SELECT id, path FROM file LIMIT 10")
            file_data = cursor.fetchall()
            print("File paths sample:")
            for row in file_data:
                print(f"  {row['id']}: {row['path']}")

            # Try direct SELECT * to avoid column name issues
            try:
                # Process all line_bits rows using dictionary access
                cursor.execute("SELECT * FROM line_bits")
                line_bits_data = cursor.fetchall()
                print(f"Found {len(line_bits_data)} line_bits entries")

                # Get file paths and contexts
                file_paths = {}
                cursor.execute("SELECT id, path FROM file")
                for row in cursor.fetchall():
                    file_paths[row['id']] = row['path']

                contexts = {}
                cursor.execute("SELECT id, context FROM context")
                for row in cursor.fetchall():
                    contexts[row['id']] = row['context']

                # Process data
                for row in line_bits_data:
                    # Extract using dictionary keys which avoids column name issues
                    file_id = row['file_id']
                    context_id = row['context_id']
                    numbits_blob = row['numbits']

                    file_path = file_paths.get(file_id)
                    if not file_path:
                        continue

                    context = contexts.get(context_id)
                    if not context:
                        continue

                    # Extract test name from context
                    test_name = self._extract_test_name(context)
                    if not test_name:
                        continue

                    # Decode the bitmap in numbits to get actual line numbers
                    if isinstance(numbits_blob, bytes):
                        # Convert blob to binary string
                        binary_str = ''.join(format(byte, '08b') for byte in numbits_blob)

                        # Each '1' bit represents a covered line
                        for i, bit in enumerate(binary_str):
                            if bit == '1':
                                line_no = i + 1  # Line numbers are 1-based

                                # Store mappings
                                line_key = f"{file_path}:{line_no}"
                                self.test_to_lines[test_name].add(line_key)
                                self.line_to_tests[line_key].add(test_name)
                                self.all_covered_lines.add(line_key)
                                self.all_tests.add(test_name)


            except sqlite3.OperationalError as e:
                print(f"Error with dictionary-based access: {e}")
        except Exception as e:
            # Get detailed schema information for diagnostics
            schema_info = {}
            for table in tables:
                try:
                    cursor.execute(f"PRAGMA table_info({table})")
                    schema_info[table] = [column[1] for column in cursor.fetchall()]
                except Exception as t_info_e:
                    print(f"Error getting schema for table {table}: {t_info_e}")
                    schema_info[table] = ["<error getting columns>"]

            error_msg = f"Error analyzing coverage database: {e}\n"
            error_msg += "Database schema:\n"
            for table, columns in schema_info.items():
                error_msg += f"Table '{table}': {', '.join(columns)}\n"

            raise RuntimeError(error_msg) from e
        finally:
            conn.close()

        print(f"Processed {len(self.all_tests)} tests covering {len(self.all_covered_lines)} lines")

        # Save the raw mappings for debugging
        self._save_mappings()

    def _extract_test_name(self, context):
        """Extract test name from coverage context string."""
        # Regular test format: test_file.py::test_name|run
        match = re.match(r'(.+?)\|(setup|run|teardown)$', context)
        if match:
            return match.group(1)

        # Special test format: special:mark_type:test_name
        match = re.match(r'special:(\w+):(.+)$', context)
        if match:
            return match.group(2)

        # Example: tests/unit/test_analysis_core.py::TestSAEAnalysisDict::test_core_sae_analysis_dict
        match = re.match(r'(tests/.*?::.*?)(?:\[|$)', context)
        if match:
            return match.group(1)

        # Direct context is the test name pattern itself
        match = re.search(r'(test_\w+(?:\[\w+\])?)', context)
        if match:
            return match.group(1)

        # If all patterns fail but looks like a test name
        if context and 'test_' in context:
            return context

        return None

    def _save_mappings(self):
        """Save the test-to-lines and line-to-tests mappings for debugging."""
        # Convert to serializable format
        test_to_lines = {k: list(v) for k, v in self.test_to_lines.items()}
        line_to_tests = {k: list(v) for k, v in self.line_to_tests.items()}

        with open(self.output_dir / 'test_to_lines.json', 'w') as f:
            json.dump(test_to_lines, f, indent=2)

        with open(self.output_dir / 'line_to_tests.json', 'w') as f:
            json.dump(line_to_tests, f, indent=2)

    def find_essential_tests(self):
        """Identify tests that uniquely cover at least one line."""
        self.essential_tests = set()
        unique_coverage = defaultdict(list)

        # Find tests that uniquely cover a line
        for line, tests in self.line_to_tests.items():
            if len(tests) == 1:
                test = list(tests)[0]
                self.essential_tests.add(test)
                unique_coverage[test].append(line)

        # Save the essential tests information
        essential_tests_info = {
            "count": len(self.essential_tests),
            "tests": {test: len(self.test_to_lines[test]) for test in self.essential_tests}
        }

        with open(self.output_dir / 'essential_tests.json', 'w') as f:
            json.dump(essential_tests_info, f, indent=2)

        with open(self.output_dir / 'unique_coverage.json', 'w') as f:
            json.dump({k: v for k, v in unique_coverage.items()}, f, indent=2)

        print(f"Found {len(self.essential_tests)} essential tests that uniquely cover at least one line")

    def find_removal_candidates(self):
        """Find tests that can be removed without reducing coverage."""
        # Start with all tests except essential ones
        removal_candidates = sorted(self.all_tests - self.essential_tests)

        # Calculate coverage statistics for each test
        test_stats = []
        for test in removal_candidates:
            lines_covered = len(self.test_to_lines[test])
            # Check what percentage of lines is also covered by other tests
            redundant_lines = 0
            for line in self.test_to_lines[test]:
                if len(self.line_to_tests[line]) > 1:
                    redundant_lines += 1

            redundancy_ratio = redundant_lines / lines_covered if lines_covered > 0 else 1.0

            test_stats.append({
                "test": test,
                "lines_covered": lines_covered,
                "redundant_lines": redundant_lines,
                "redundancy_ratio": redundancy_ratio,
                "unique_lines": lines_covered - redundant_lines
            })

        # Sort by redundancy ratio (highest first) and then by lines covered (lowest first)
        test_stats.sort(key=lambda x: (-x["redundancy_ratio"], x["lines_covered"]))

        # Apply max_candidates limit if specified
        self.removal_candidates = test_stats[:self.max_candidates] if self.max_candidates else test_stats

        # Save the results
        with open(self.output_dir / 'removal_candidates.json', 'w') as f:
            json.dump(self.removal_candidates, f, indent=2)

        print(f"Found {len(self.removal_candidates)} potential candidates for removal")

    def greedy_test_minimization(self):
        """Use a greedy algorithm to find a minimal set of tests that maintain coverage."""
        remaining_lines = set(self.all_covered_lines)
        selected_tests = set()

        # First, include all essential tests
        for test in self.essential_tests:
            selected_tests.add(test)
            remaining_lines -= self.test_to_lines[test]

        # Then greedily select tests that cover the most remaining lines
        non_essential_tests = list(self.all_tests - self.essential_tests)

        while remaining_lines and non_essential_tests:
            # Find the test that covers the most remaining lines
            best_test = None
            best_coverage = 0

            for test in non_essential_tests:
                coverage = len(remaining_lines.intersection(self.test_to_lines[test]))
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_test = test

            if best_test is None or best_coverage == 0:
                # No test covers any remaining lines
                break

            # Add the best test and update remaining lines
            selected_tests.add(best_test)
            remaining_lines -= self.test_to_lines[best_test]
            non_essential_tests.remove(best_test)

        # Tests that weren't selected are candidates for removal
        removal_candidates = self.all_tests - selected_tests

        # Calculate statistics for the minimization
        minimization_stats = {
            "total_tests": len(self.all_tests),
            "essential_tests": len(self.essential_tests),
            "total_tests_needed": len(selected_tests),
            "removal_candidates": len(removal_candidates),
            "removal_candidates_list": sorted(list(removal_candidates))
        }

        with open(self.output_dir / 'minimization_stats.json', 'w') as f:
            json.dump(minimization_stats, f, indent=2)

        print(f"Minimization results: {len(selected_tests)} tests needed, {len(removal_candidates)} can be removed")

        return selected_tests, removal_candidates

    def generate_report(self):
        """Generate a human-readable report."""
        report_path = self.output_dir / 'test_coverage_analysis_report.txt'

        with open(report_path, 'w') as f:
            f.write("Test Coverage Analysis Report\n")
            f.write("============================\n\n")

            f.write(f"Total number of tests: {len(self.all_tests)}\n")
            f.write(f"Total lines covered: {len(self.all_covered_lines)}\n")
            f.write(f"Essential tests (uniquely cover at least one line): {len(self.essential_tests)}\n\n")

            f.write("Top candidates for removal:\n")
            f.write("-------------------------\n\n")

            for i, candidate in enumerate(self.removal_candidates[:20], 1):
                f.write(f"{i}. {candidate['test']}\n")
                f.write(f"   - Lines covered: {candidate['lines_covered']}\n")
                f.write(f"   - Redundancy ratio: {candidate['redundancy_ratio']:.2f}\n")
                f.write(f"   - Unique lines: {candidate['unique_lines']}\n\n")

            f.write("\nGreedy Test Minimization Results:\n")
            f.write("-------------------------------\n\n")

            # Load minimization stats
            with open(self.output_dir / 'minimization_stats.json', 'r') as mf:
                stats = json.load(mf)

            f.write(f"Tests needed for full coverage: {stats['total_tests_needed']} ")
            f.write(f"Tests that can be removed: {stats['removal_candidates']}\n\n")
            f.write("Recommended action: Review the candidates for removal and verify through targeted testing.\n")

        print(f"Report generated at {report_path}")

    def run_analysis(self):
        """Run the complete analysis pipeline."""
        self.setup_coverage_config()
        self.collect_coverage_data()
        self.analyze_coverage_database()
        self.find_essential_tests()
        self.find_removal_candidates()
        self.greedy_test_minimization()
        self.generate_report()
        print("Analysis complete!")


def main():
    """Parse arguments and run the analyzer."""
    parser = argparse.ArgumentParser(description='Analyze test coverage and identify redundant tests.')
    parser.add_argument('--output-dir', help='Directory to store output files', default='test_coverage_analysis')
    parser.add_argument('--max-candidates', type=int, help='Maximum number of removal candidates to report')
    parser.add_argument(
        '--normal-subset',
        help='Filter expression for normal pytest tests (e.g., "test_feature1 or test_feature2")'
    )
    parser.add_argument(
        '--standalone-subset',
        help='Filter expression for standalone tests (e.g., "test_standalone1 or test_standalone2")'
    )
    parser.add_argument(
        '--profile-ci-subset',
        help='Filter expression for profile_ci tests (e.g., "test_profile1 or test_profile2")'
    )
    args = parser.parse_args()

    analyzer = TestCoverageAnalyzer(
        output_dir=args.output_dir,
        max_candidates=args.max_candidates,
        normal_subset=args.normal_subset,
        standalone_subset=args.standalone_subset,
        profile_ci_subset=args.profile_ci_subset
    )

    try:
        analyzer.run_analysis()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
