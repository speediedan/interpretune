#!/usr/bin/env python3
"""Test Coverage Analyzer.

This script integrates with pytest and coverage.py to track which lines are covered by which tests,
then identifies tests that can be removed without reducing overall coverage.

The script uses coverage.py's dynamic contexts feature to associate covered lines with specific tests.
It can be run as a standalone script or integrated into your existing test infrastructure.

Usage:
    python analyze_test_coverage.py [--max-candidates N] [--output-dir PATH] [--branch-level]
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
                 standalone_subset=None, profile_ci_subset=None,
                 profile_subset=None, optional_subset=None, mark_types_to_run=None,
                 branch_level=False):
        """Initialize the analyzer with configuration options.

        Args:
            output_dir: Directory to store output files (default: current directory)
            max_candidates: Maximum number of candidate tests to report (default: no limit)
            normal_subset: Filter expression for normal pytest tests (e.g., "test_1 or test_2")
            standalone_subset: Filter expression for standalone tests (e.g., "test_3 or test_4")
            profile_ci_subset: Filter expression for profile_ci tests (e.g., "test_5 or test_6")
            profile_subset: Filter expression for profile tests (e.g., "test_7 or test_8")
            optional_subset: Filter expression for optional tests (e.g., "test_9 or test_10")
            mark_types_to_run: Comma-separated string of mark types to run (default: "normal,standalone,profile_ci")
            branch_level: Whether to perform branch-level analysis instead of statement-level
        """
        self.output_dir = Path(output_dir or '.')
        self.max_candidates = max_candidates
        self.normal_subset = normal_subset
        self.standalone_subset = standalone_subset
        self.profile_ci_subset = profile_ci_subset
        self.profile_subset = profile_subset
        self.optional_subset = optional_subset
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.branch_level = branch_level

        # Parse mark_types_to_run
        # Default comes from argparse, so mark_types_to_run is always a string.
        if mark_types_to_run and mark_types_to_run.strip():
            self.mark_types_to_run = {s.strip() for s in mark_types_to_run.split(',') if s.strip()}
        else: # User provided an empty string e.g. --mark-types-to-run ""
            self.mark_types_to_run = set()


        # Set up paths for coverage data
        self.coverage_db = self.output_dir / '.coverage'
        self.coverage_config = self.output_dir / '.coveragerc'

        # Store analysis results for statement-level (default)
        self.test_to_lines = defaultdict(set)
        self.line_to_tests = defaultdict(set)
        self.all_covered_lines = set()

        # Store analysis results for branch-level
        self.test_to_branches = defaultdict(set)
        self.branch_to_tests = defaultdict(set)
        self.all_covered_branches = set()

        # Common data structures
        self.all_tests = set()
        self.statement_level_base = set()
        self.statement_level_redundant = []
        self.branch_level_base = set()
        self.branch_level_redundant = []

    def setup_coverage_config(self):
        """Create a .coveragerc file for the analysis."""
        # Add branch=True to the config if branch-level analysis is enabled
        branch_option = "branch = True" if self.branch_level else ""

        config_content = """
[run]
source = src/interpretune
data_file = {}
{}

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
""".format(self.coverage_db, branch_option)

        with open(self.coverage_config, 'w') as f:
            f.write(config_content.strip())
        print(f"Created coverage config at {self.coverage_config}")
        if self.branch_level:
            print("Branch coverage analysis mode enabled")
        else:
            print("Statement coverage analysis mode enabled (default)")

    def collect_coverage_data(self):
        """Run pytest with coverage to collect test-specific coverage data."""
        # Ensure coverage data file is created in the output directory

        # Determine if normal tests should run
        run_normal_tests = "normal" in self.mark_types_to_run
        if self.normal_subset and "normal" not in self.mark_types_to_run:
            print(
                f"Warning: --normal-subset is provided, but 'normal' is not in --mark-types-to-run "
                f"('{','.join(sorted(list(self.mark_types_to_run)))}'). "
                "Running 'normal' tests due to the subset filter."
            )
            run_normal_tests = True

        if run_normal_tests:
            # Run regular tests with pytest-cov
            cmd = [
                "python", "-m", "pytest", "--cov", f"--cov-config={self.coverage_config}", "--cov-append",
                "--cov-context=test", "--cov-report=", "tests", "-v"
            ]

            # Add normal subset filter if specified
            if self.normal_subset:
                # Remove any surrounding quotes that might have been carried over from the shell
                normal_subset_cleaned = self.normal_subset.strip("'\"")
                # Use the filter expression directly as a pytest -k expression
                cmd.extend(["-k", normal_subset_cleaned])

            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        else:
            print(
                "Skipping normal tests as 'normal' is not in --mark-types-to-run and "
                "--normal-subset is not provided."
            )


        # Run special tests with different mark types
        all_special_mark_types = ["standalone", "profile_ci", "profile", "optional"]

        env_vars_config = {
            "standalone": {"IT_RUN_STANDALONE_TESTS": "1"},
            "profile_ci": {"IT_RUN_PROFILING_TESTS": "1"},
            "profile": {"IT_RUN_PROFILING_TESTS": "2"},
            "optional": {"IT_RUN_OPTIONAL_TESTS": "1"},
        }

        subset_config = {
            "standalone": self.standalone_subset,
            "profile_ci": self.profile_ci_subset,
            "profile": self.profile_subset,
            "optional": self.optional_subset,
        }

        for mark_type in all_special_mark_types:
            env = os.environ.copy()
            current_subset_str = subset_config.get(mark_type)

            run_this_mark_type = mark_type in self.mark_types_to_run

            if current_subset_str and mark_type not in self.mark_types_to_run:
                print(
                    f"Warning: --{mark_type}-subset is provided, but '{mark_type}' is not in "
                    f"--mark-types-to-run ('{','.join(sorted(list(self.mark_types_to_run)))}'). "
                    f"Running '{mark_type}' tests due to the subset filter."
                )
                run_this_mark_type = True

            if not run_this_mark_type:
                print(
                    f"Skipping '{mark_type}' tests as it's not in --mark-types-to-run and "
                    f"--{mark_type}-subset is not provided."
                )
                continue

            # Set up environment variables needed by special_tests.sh
            for var, value in env_vars_config.get(mark_type, {}).items():
                env[var] = value

            subset_cleaned = current_subset_str.strip("'\"") if current_subset_str else None

            # Pass the coverage data file path as an environment variable for special_tests.sh
            env["COVERAGE_ANALYSIS_CONFIG_FILE"] = str(self.coverage_config)
            cmd = ["bash", "tests/special_tests.sh", f"--mark_type={mark_type}"]

            # Add filter pattern if subset is specified
            if subset_cleaned:
                cmd.extend([f"--filter_pattern={subset_cleaned}"])

            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, env=env, check=True)

    def analyze_coverage_database(self):
        """Analyze the coverage database to extract line-to-test and branch-to-test mappings."""
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

            # Check if we're doing branch-level analysis
            has_arcs = False
            if 'meta' in tables:
                cursor.execute("SELECT value FROM meta WHERE key = 'has_arcs'")
                row = cursor.fetchone()
                if row and row[0].lower() == 'true':
                    has_arcs = True
                    print("Database indicates branch coverage was collected (has_arcs=true)")

            # Verify branch mode matches database
            if self.branch_level and not has_arcs:
                print("WARNING: Branch-level analysis requested but coverage database doesn't contain branch data")
            elif not self.branch_level and has_arcs:
                print("WARNING: Statement-level analysis requested but coverage database contains branch data")

            # Get file paths and contexts
            file_paths = {}
            cursor.execute("SELECT id, path FROM file")
            for row in cursor.fetchall():
                file_paths[row['id']] = row['path']

            contexts = {}
            cursor.execute("SELECT id, context FROM context")
            for row in cursor.fetchall():
                contexts[row['id']] = row['context']

            # Print contexts info
            print(f"Found {len(contexts)} contexts in the database")
            for i, (ctx_id, ctx) in enumerate(list(contexts.items())[:10]):  # Print sample contexts
                print(f"Context {i}: {ctx_id} -> {ctx}")
            if len(contexts) > 10:
                print(f"... and {len(contexts) - 10} more")

            # Print some file info
            print("File paths sample:")
            for i, (file_id, path) in enumerate(list(file_paths.items())[:10]):
                print(f"  {file_id}: {path}")
            if len(file_paths) > 10:
                print(f"... and {len(file_paths) - 10} more")

            # Process statement coverage if not in branch mode
            if not self.branch_level:
                if 'line_bits' not in tables:
                    raise ValueError("line_bits table not found in the coverage database")

                # Process line_bits data for statement coverage
                cursor.execute("SELECT * FROM line_bits")
                line_bits_data = cursor.fetchall()
                print(f"Found {len(line_bits_data)} line_bits entries")

                # Process statement-level data
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

                print(f"Processed {len(self.all_tests)} tests covering {len(self.all_covered_lines)} lines")

            # Process branch coverage if in branch mode
            elif self.branch_level:
                if 'arc' not in tables:
                    raise ValueError("arc table not found in the coverage database")

                # Process arc data for branch coverage
                cursor.execute("SELECT * FROM arc")
                arc_data = cursor.fetchall()
                print(f"Found {len(arc_data)} arc entries")

                # Process branch-level data
                for row in arc_data:
                    file_id = row['file_id']
                    context_id = row['context_id']
                    fromno = row['fromno']
                    tono = row['tono']

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

                    # Create a unique identifier for this branch
                    branch_key = f"{file_path}:{fromno}->{tono}"

                    # Store mappings
                    self.test_to_branches[test_name].add(branch_key)
                    self.branch_to_tests[branch_key].add(test_name)
                    self.all_covered_branches.add(branch_key)
                    self.all_tests.add(test_name)

                print(f"Processed {len(self.all_tests)} tests covering {len(self.all_covered_branches)} branches")

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
        if not self.branch_level:
            # Save statement coverage mappings
            test_to_lines = {k: list(v) for k, v in self.test_to_lines.items()}
            line_to_tests = {k: list(v) for k, v in self.line_to_tests.items()}

            with open(self.output_dir / 'test_to_lines.json', 'w') as f:
                json.dump(test_to_lines, f, indent=2)

            with open(self.output_dir / 'line_to_tests.json', 'w') as f:
                json.dump(line_to_tests, f, indent=2)
        else:
            # Save branch coverage mappings
            test_to_branches = {k: list(v) for k, v in self.test_to_branches.items()}
            branch_to_tests = {k: list(v) for k, v in self.branch_to_tests.items()}

            with open(self.output_dir / 'test_to_branches.json', 'w') as f:
                json.dump(test_to_branches, f, indent=2)

            with open(self.output_dir / 'branch_to_tests.json', 'w') as f:
                json.dump(branch_to_tests, f, indent=2)

    def find_base_tests(self):
        """Identify tests that uniquely cover at least one line or branch."""
        if not self.branch_level:
            # Statement-level analysis
            self.statement_level_base = set()
            unique_coverage = defaultdict(list)

            # Find tests that uniquely cover a line
            for line, tests in self.line_to_tests.items():
                if len(tests) == 1:
                    test = list(tests)[0]
                    self.statement_level_base.add(test)
                    unique_coverage[test].append(line)

            # Save the base tests information
            base_tests_info = {
                "count": len(self.statement_level_base),
                "tests": {test: len(self.test_to_lines[test]) for test in self.statement_level_base}
            }

            with open(self.output_dir / 'statement_level_base.json', 'w') as f:
                json.dump(base_tests_info, f, indent=2)

            with open(self.output_dir / 'unique_statement_coverage.json', 'w') as f:
                json.dump({k: v for k, v in unique_coverage.items()}, f, indent=2)

            print(f"Found {len(self.statement_level_base)} base tests that uniquely cover at least one line")
        else:
            # Branch-level analysis
            self.branch_level_base = set()
            unique_coverage = defaultdict(list)

            # Find tests that uniquely cover a branch
            for branch, tests in self.branch_to_tests.items():
                if len(tests) == 1:
                    test = list(tests)[0]
                    self.branch_level_base.add(test)
                    unique_coverage[test].append(branch)

            # Save the base tests information
            base_tests_info = {
                "count": len(self.branch_level_base),
                "tests": {test: len(self.test_to_branches[test]) for test in self.branch_level_base}
            }

            with open(self.output_dir / 'branch_level_base.json', 'w') as f:
                json.dump(base_tests_info, f, indent=2)

            with open(self.output_dir / 'unique_branch_coverage.json', 'w') as f:
                json.dump({k: v for k, v in unique_coverage.items()}, f, indent=2)

            print(f"Found {len(self.branch_level_base)} base tests that uniquely cover at least one branch")

    def find_removal_candidates(self):
        """Find tests that can be removed without reducing coverage."""
        if not self.branch_level:
            # Statement-level analysis
            # Start with all tests except base ones
            removal_candidates = sorted(self.all_tests - self.statement_level_base)

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
            self.statement_level_redundant = test_stats[:self.max_candidates] if self.max_candidates else test_stats

            # Sort by test name before saving
            sorted_redundant = sorted(self.statement_level_redundant, key=lambda x: x["test"])

            # Save the results
            with open(self.output_dir / 'statement_level_redundant.json', 'w') as f:
                json.dump(sorted_redundant, f, indent=2)

            print(f"Found {len(self.statement_level_redundant)} potential statement-level candidates for removal")

        else:
            # Branch-level analysis
            # Start with all tests except base ones
            removal_candidates = sorted(self.all_tests - self.branch_level_base)

            # Calculate coverage statistics for each test
            test_stats = []
            for test in removal_candidates:
                branches_covered = len(self.test_to_branches[test])
                # Check what percentage of branches is also covered by other tests
                redundant_branches = 0
                for branch in self.test_to_branches[test]:
                    if len(self.branch_to_tests[branch]) > 1:
                        redundant_branches += 1

                redundancy_ratio = redundant_branches / branches_covered if branches_covered > 0 else 1.0

                test_stats.append({
                    "test": test,
                    "branches_covered": branches_covered,
                    "redundant_branches": redundant_branches,
                    "redundancy_ratio": redundancy_ratio,
                    "unique_branches": branches_covered - redundant_branches
                })

            # Sort by redundancy ratio (highest first) and then by branches covered (lowest first)
            test_stats.sort(key=lambda x: (-x["redundancy_ratio"], x["branches_covered"]))

            # Apply max_candidates limit if specified
            self.branch_level_redundant = test_stats[:self.max_candidates] if self.max_candidates else test_stats

            # Sort by test name before saving
            sorted_redundant = sorted(self.branch_level_redundant, key=lambda x: x["test"])

            # Save the results
            with open(self.output_dir / 'branch_level_redundant.json', 'w') as f:
                json.dump(sorted_redundant, f, indent=2)

            print(f"Found {len(self.branch_level_redundant)} potential branch-level candidates for removal")

    def greedy_test_minimization(self):
        """Use a greedy algorithm to find a minimal set of tests that maintain coverage."""
        if not self.branch_level:
            # Statement-level minimization
            remaining_lines = set(self.all_covered_lines)
            selected_tests = set()

            # First, include all base tests
            for test in self.statement_level_base:
                selected_tests.add(test)
                remaining_lines -= self.test_to_lines[test]

            # Then greedily select tests that cover the most remaining lines
            non_base_tests = list(self.all_tests - self.statement_level_base)

            while remaining_lines and non_base_tests:
                # Find the test that covers the most remaining lines
                best_test = None
                best_coverage = 0

                for test in non_base_tests:
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
                non_base_tests.remove(best_test)

            # Tests that weren't selected are candidates for removal
            removal_candidates = self.all_tests - selected_tests

            # Calculate statistics for the minimization
            minimization_stats = {
                "total_tests": len(self.all_tests),
                "base_tests": len(self.statement_level_base),
                "total_tests_needed": len(selected_tests),
                "removal_candidates": len(removal_candidates),
                "removal_candidates_list": sorted(list(removal_candidates))
            }

            with open(self.output_dir / 'statement_minimization_stats.json', 'w') as f:
                json.dump(minimization_stats, f, indent=2)

            print(
                f"Statement minimization results: {len(selected_tests)} tests needed, "
                f"{len(removal_candidates)} can be removed"
            )

            return selected_tests, removal_candidates
        else:
            # Branch-level minimization
            remaining_branches = set(self.all_covered_branches)
            selected_tests = set()

            # First, include all base tests
            for test in self.branch_level_base:
                selected_tests.add(test)
                remaining_branches -= self.test_to_branches[test]

            # Then greedily select tests that cover the most remaining branches
            non_base_tests = list(self.all_tests - self.branch_level_base)

            while remaining_branches and non_base_tests:
                # Find the test that covers the most remaining branches
                best_test = None
                best_coverage = 0

                for test in non_base_tests:
                    coverage = len(remaining_branches.intersection(self.test_to_branches[test]))
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_test = test

                if best_test is None or best_coverage == 0:
                    # No test covers any remaining branches
                    break

                # Add the best test and update remaining branches
                selected_tests.add(best_test)
                remaining_branches -= self.test_to_branches[best_test]
                non_base_tests.remove(best_test)

            # Tests that weren't selected are candidates for removal
            removal_candidates = self.all_tests - selected_tests

            # Calculate statistics for the minimization
            minimization_stats = {
                "total_tests": len(self.all_tests),
                "base_tests": len(self.branch_level_base),
                "total_tests_needed": len(selected_tests),
                "removal_candidates": len(removal_candidates),
                "removal_candidates_list": sorted(list(removal_candidates))
            }

            with open(self.output_dir / 'branch_minimization_stats.json', 'w') as f:
                json.dump(minimization_stats, f, indent=2)

            print(
                f"Branch minimization results: {len(selected_tests)} tests needed, "
                f"{len(removal_candidates)} can be removed"
            )

            return selected_tests, removal_candidates

    def generate_report(self):
        """Generate a human-readable report."""
        report_path = self.output_dir / 'test_coverage_analysis_report.txt'

        with open(report_path, 'w') as f:
            f.write("Test Coverage Analysis Report\n")
            f.write("============================\n\n")

            f.write(f"Analysis mode: {'Branch-level' if self.branch_level else 'Statement-level'}\n\n")
            f.write(f"Total number of tests: {len(self.all_tests)}\n")

            if not self.branch_level:
                # Statement-level reporting
                f.write(f"Total lines covered: {len(self.all_covered_lines)}\n")
                f.write(f"base tests (uniquely cover at least one line): {len(self.statement_level_base)}\n\n")

                f.write("All statement-level candidates for removal (sorted by name):\n")
                f.write("---------------------------------------\n\n")

                # Load the sorted redundant tests
                with open(self.output_dir / 'statement_level_redundant.json', 'r') as rf:
                    redundant_tests = json.load(rf)

                # Just list the test names
                for i, candidate in enumerate(redundant_tests, 1):
                    f.write(f"{i}. {candidate['test']}\n")

                f.write("\nGreedy Test Minimization Results (Statement-level):\n")
                f.write("------------------------------------------------\n\n")

                # Load minimization stats
                with open(self.output_dir / 'statement_minimization_stats.json', 'r') as mf:
                    stats = json.load(mf)
            else:
                # Branch-level reporting
                f.write(f"Total branches covered: {len(self.all_covered_branches)}\n")
                f.write(f"base tests (uniquely cover at least one branch): {len(self.branch_level_base)}\n\n")

                f.write("All branch-level candidates for removal (sorted by name):\n")
                f.write("-------------------------------------\n\n")

                # Load the sorted redundant tests
                with open(self.output_dir / 'branch_level_redundant.json', 'r') as rf:
                    redundant_tests = json.load(rf)

                # Just list the test names
                for i, candidate in enumerate(redundant_tests, 1):
                    f.write(f"{i}. {candidate['test']}\n")

                f.write("\nGreedy Test Minimization Results (Branch-level):\n")
                f.write("--------------------------------------------\n\n")

                # Load minimization stats
                with open(self.output_dir / 'branch_minimization_stats.json', 'r') as mf:
                    stats = json.load(mf)

            f.write(f"Tests needed for full coverage: {stats['total_tests_needed']} ")
            f.write(f"Tests that can be removed: {stats['removal_candidates']}\n\n")
            f.write("Recommended action: Review the candidates for removal and verify through targeted testing.\n")

        print(f"Report generated at {report_path}")

    def generate_coverage_reports(self):
        """Generate HTML and JSON coverage reports."""
        # Create subdirectories for HTML reports
        html_dir = self.output_dir / 'html_report'
        html_dir.mkdir(exist_ok=True)

        # Define JSON output file
        json_file = self.output_dir / 'coverage.json'

        # Determine report type label
        report_type = "branch" if self.branch_level else "statement"

        # Generate HTML report
        print(f"Generating {report_type}-level HTML coverage report...")
        html_cmd = [
            "python", "-m", "coverage", "html",
            f"--rcfile={self.coverage_config}",
            f"--data-file={self.coverage_db}",
            "-d", f"{html_dir}"
        ]
        subprocess.run(html_cmd, check=True)

        # Generate JSON report
        print(f"Generating {report_type}-level JSON coverage report...")
        json_cmd = [
            "python", "-m", "coverage", "json",
            f"--rcfile={self.coverage_config}",
            f"--data-file={self.coverage_db}",
            "-o", f"{json_file}"
        ]
        subprocess.run(json_cmd, check=True)

        print(f"HTML report available at: {html_dir}")
        print(f"JSON report available at: {json_file}")

    def run_analysis(self):
        """Run the complete analysis pipeline."""
        self.setup_coverage_config()
        self.collect_coverage_data()
        self.analyze_coverage_database()
        self.find_base_tests()
        self.find_removal_candidates()
        self.greedy_test_minimization()
        self.generate_report()
        # Generate coverage reports (HTML and JSON)
        self.generate_coverage_reports()
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
    parser.add_argument(
        '--profile-subset',
        help='Filter expression for profile tests (e.g., "test_profile_A or test_profile_B")'
    )
    parser.add_argument(
        '--optional-subset',
        help='Filter expression for optional tests (e.g., "test_optional_X or test_optional_Y")'
    )
    parser.add_argument(
        '--mark-types-to-run',
        default="normal,standalone,profile_ci",
        help='Comma-separated list of mark types to run (e.g., "normal,standalone,profile"). '
             'Supported types: normal, standalone, profile_ci, profile, optional. '
             'Default: "normal,standalone,profile_ci". '
             'If a specific subset (e.g. --profile-subset) is provided for a mark type not listed here, '
             'that mark type will still be run with a warning.'
    )
    parser.add_argument(
        '--branch-level',
        action='store_true',
        help='Perform branch-level analysis instead of statement-level (default)'
    )
    args = parser.parse_args()

    analyzer = TestCoverageAnalyzer(
        output_dir=args.output_dir,
        max_candidates=args.max_candidates,
        normal_subset=args.normal_subset,
        standalone_subset=args.standalone_subset,
        profile_ci_subset=args.profile_ci_subset,
        profile_subset=args.profile_subset,
        optional_subset=args.optional_subset,
        mark_types_to_run=args.mark_types_to_run,
        branch_level=args.branch_level
    )

    try:
        analyzer.run_analysis()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
