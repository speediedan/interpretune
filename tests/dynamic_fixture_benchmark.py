#!/usr/bin/env python3
"""Dynamic Fixture Benchmarking Script for Interpretune.

This script provides comprehensive analysis of all fixtures in the test suite, including dynamic discovery,
usage counts, categorization by computational requirements, and fixture metadata analysis.

Features:
- Dynamically discovers all generated fixtures from conftest.py FIXTURE_CFGS
- Categorizes fixtures by model type (Real Model vs Custom Test vs Config-Only)
- Counts actual fixture usage across all test files
- Analyzes fixture scopes and dependencies
- Generates detailed markdown reports for developer reference

Usage:
    # From repository root:
    cd /path/to/interpretune
    PYTHONPATH=/path/to/interpretune python tests/dynamic_fixture_benchmark.py

    # This will generate: tests/fixture_benchmark_report.md

Categories:
- Real Model: Fixtures using actual models (gpt2, llama3, gemma2)
- Custom Model: Fixtures using trivial test models (cust patterns)
- Config Only: Configuration-only fixtures (it_session_cfg type)
- Static: Traditional pytest fixtures defined with @pytest.fixture

The generated report includes:
1. Summary statistics by category and scope
2. Detailed fixture table with usage counts
3. High-impact optimization targets (heavily used session-scope fixtures)
4. Unused fixtures that could be considered for removal
"""

import ast
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import psutil
import torch


class FixtureCategory:
    """Categories for fixture computational requirements."""

    REAL_MODEL = "Real Model"  # gpt2, llama3, gemma2, etc.
    CUSTOM_MODEL = "Custom Model"  # cust (trivial test models)
    CONFIG_ONLY = "Config Only"  # configuration-only fixtures
    STATIC = "Static"  # traditional static fixtures


class FixtureMetrics:
    """Container for fixture benchmarking metrics."""

    def __init__(self):
        self.init_time: float = 0.0
        self.memory_before: float = 0.0
        self.memory_after: float = 0.0
        self.peak_memory: float = 0.0
        self.gpu_memory_before: float = 0.0
        self.gpu_memory_after: float = 0.0
        self.gpu_peak_memory: float = 0.0
        self.error: Optional[str] = None

    @property
    def memory_delta(self) -> float:
        """Memory increase during fixture instantiation (MB)."""
        return self.memory_after - self.memory_before

    @property
    def gpu_memory_delta(self) -> float:
        """GPU memory increase during fixture instantiation (MB)."""
        return self.gpu_memory_after - self.gpu_memory_before


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_gpu_memory_usage_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def _parse_conftest_ast(conftest_path: Path) -> Dict[str, Tuple[Any, str]]:
    """Parse conftest.py using AST to find fixture definitions."""
    fixtures = {}

    try:
        with open(conftest_path, "r") as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if this function has a pytest.fixture decorator
                for decorator in node.decorator_list:
                    scope = "function"  # default scope

                    if (
                        isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Attribute)
                        and isinstance(decorator.func.value, ast.Name)
                        and decorator.func.value.id == "pytest"
                        and decorator.func.attr == "fixture"
                    ):
                        # Check for scope argument
                        for keyword in decorator.keywords:
                            if keyword.arg == "scope" and isinstance(keyword.value, ast.Constant):
                                scope = keyword.value.value

                        fixtures[node.name] = (None, scope)  # No function object available from AST
                        break

                    elif (
                        isinstance(decorator, ast.Attribute)
                        and isinstance(decorator.value, ast.Name)
                        and decorator.value.id == "pytest"
                        and decorator.attr == "fixture"
                    ):
                        fixtures[node.name] = (None, scope)
                        break

    except Exception as e:
        print(f"Error parsing conftest.py: {e}")

    return fixtures


def discover_generated_fixtures() -> Dict[str, Tuple[Any, str]]:
    """Discover all dynamically generated fixtures from conftest.py."""
    fixtures = {}

    try:
        # Use pytest's own fixture discovery mechanism
        print("Discovering dynamic fixtures using pytest...")

        # Run pytest to get fixture information
        cmd = [sys.executable, "-m", "pytest", "--fixtures", "-q"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent), timeout=30)

        if result.returncode == 0:
            output_lines = result.stdout.split("\n")
            current_section = None

            for line in output_lines:
                line = line.strip()

                # Remove ANSI color codes
                import re

                line = re.sub(r"\x1b\[[0-9;]*m", "", line)

                # Look for fixtures defined in conftest
                if "fixtures defined from tests.conftest" in line:
                    current_section = "conftest"
                    continue
                elif "fixtures defined from" in line and "conftest" not in line:
                    current_section = None
                    continue

                if current_section == "conftest" and line:
                    # Parse fixture lines like: "get_it_session__core_cust__initonly [session scope] -- conftest.py:354"
                    if " [" in line and " scope]" in line and "--" in line:
                        try:
                            parts = line.split(" [")
                            fixture_name = parts[0].strip()
                            scope_part = parts[1].split(" scope]")[0]

                            # Only include dynamic fixtures (get_* patterns)
                            dynamic_patterns = [
                                "get_it_session__",
                                "get_it_module__",
                                "get_analysis_session__",
                                "get_it_session_cfg__",
                            ]
                            if any(fixture_name.startswith(pattern) for pattern in dynamic_patterns):
                                fixtures[fixture_name] = (None, scope_part)  # No function object from this method
                                print(f"  Found dynamic fixture: {fixture_name} (scope: {scope_part})")
                        except Exception as e:
                            print(f"  Error parsing fixture line '{line}': {e}")
                            continue
        else:
            print(f"pytest --fixtures failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        print("pytest --fixtures timed out")
    except Exception as e:
        print(f"Error running pytest --fixtures: {e}")

    # Fallback: Try direct import if pytest discovery didn't work
    if not fixtures:
        print("Falling back to direct conftest import...")
        try:
            # Add the parent directory to Python path for imports
            repo_root = Path(__file__).parent.parent
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))

            # Import conftest from the tests directory - suppress import warnings
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import tests.conftest as conftest

            # Get all dynamically generated fixtures using pytest's fixture registry
            dynamic_fixture_patterns = [
                "get_it_session__",
                "get_it_module__",
                "get_analysis_session__",
                "get_it_session_cfg__",
            ]

            for attr_name in dir(conftest):
                if any(attr_name.startswith(pattern) for pattern in dynamic_fixture_patterns):
                    try:
                        fixture_func = getattr(conftest, attr_name)
                        if hasattr(fixture_func, "_pytestfixturefunction"):
                            scope = getattr(fixture_func._pytestfixturefunction, "scope", "function")
                            fixtures[attr_name] = (fixture_func, scope)
                            print(f"  Found dynamic fixture: {attr_name} (scope: {scope})")
                    except Exception as e:
                        print(f"  Error accessing {attr_name}: {e}")
                        continue

        except ImportError as e:
            print(f"Error importing conftest: {e}")
        except Exception as e:
            print(f"Error during fallback discovery: {e}")

    print(f"Total discovered dynamic fixtures: {len(fixtures)}")
    return fixtures


def categorize_fixture(fixt_key: str, fixt_type: str) -> str:
    """Categorize fixture based on its configuration key and type."""
    # Config-only fixtures
    if fixt_type == "it_session_cfg":
        return FixtureCategory.CONFIG_ONLY

    # Real model fixtures (gpt2, llama3, gemma2)
    real_model_patterns = ["gpt2", "llama3", "gemma2"]
    if any(pattern in fixt_key.lower() for pattern in real_model_patterns):
        return FixtureCategory.REAL_MODEL

    # Custom test model fixtures (cust)
    if "cust" in fixt_key.lower():
        return FixtureCategory.CUSTOM_MODEL

    # Default to custom for other patterns
    return FixtureCategory.CUSTOM_MODEL


def count_fixture_usage() -> Dict[str, int]:
    """Count how many times each fixture is used across test files."""
    usage_counts = defaultdict(int)
    tests_dir = Path(__file__).parent

    # Search for fixture usage in all Python test files
    for test_file in tests_dir.rglob("*.py"):
        if test_file.name.startswith("test_") or test_file.name == "conftest.py":
            try:
                content = test_file.read_text()
                # Find fixture usage patterns
                fixture_patterns = [
                    r"get_it_session__\w+(?:__\w+)?",
                    r"get_it_module__\w+(?:__\w+)?",
                    r"get_analysis_session__\w+(?:__\w+)",
                    r"get_it_session_cfg__\w+",
                ]

                for pattern in fixture_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        usage_counts[match] += 1
            except Exception:
                continue  # Skip files that can't be read

    return dict(usage_counts)


def measure_baseline_pytest_startup() -> float:
    """Measure baseline pytest startup time by running a minimal test."""
    tests_dir = Path(__file__).parent
    baseline_test_path = tests_dir / "temp_baseline_test.py"

    # Create a minimal test file
    baseline_content = '''
def test_minimal():
    """Minimal test to measure pytest baseline startup time."""
    assert True
'''

    try:
        with open(baseline_test_path, "w") as f:
            f.write(baseline_content)

        # Measure pytest execution time for minimal test
        start_time = time.perf_counter()
        cmd = [sys.executable, "-m", "pytest", str(baseline_test_path), "-v", "--tb=short", "-p", "no:warnings"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(tests_dir),
            timeout=30,
        )
        end_time = time.perf_counter()

        if result.returncode == 0:
            return end_time - start_time
        else:
            return 0.0  # Fallback if baseline test fails

    except Exception:
        return 0.0
    finally:
        # Clean up
        Path(baseline_test_path).unlink(missing_ok=True)


def measure_baseline_pytest_startup_with_profiling() -> Tuple[float, Dict[str, str]]:
    """Measure baseline pytest startup time with import profiling.

    Returns:
        Tuple of (startup_time, profiling_artifacts) where profiling_artifacts contains
        paths to generated profiling files.
    """
    tests_dir = Path(__file__).parent
    profiling_dir = tests_dir / "profiling_artifacts"
    profiling_dir.mkdir(exist_ok=True)

    baseline_test_path = tests_dir / "temp_baseline_test.py"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a minimal test file
    baseline_content = '''
def test_minimal():
    """Minimal test to measure pytest baseline startup time."""
    assert True
'''

    artifacts = {}

    try:
        with open(baseline_test_path, "w") as f:
            f.write(baseline_content)

        # Run pytest with import time profiling
        start_time = time.perf_counter()
        cmd = [
            sys.executable,
            "-X",
            "importtime",
            "-m",
            "pytest",
            str(baseline_test_path),
            "-v",
            "--tb=short",
            "-p",
            "no:warnings",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(tests_dir),
            timeout=30,
        )
        end_time = time.perf_counter()

        startup_time = end_time - start_time if result.returncode == 0 else 0.0

        # Save raw import time data from stderr
        if result.stderr:
            raw_importtime_path = profiling_dir / f"pytest_importtime_raw_{timestamp}.txt"
            with open(raw_importtime_path, "w") as f:
                f.write(result.stderr)
            artifacts["raw_importtime"] = str(raw_importtime_path)

            # Convert to flamegraph format using importtime-convert
            try:
                flamegraph_path = profiling_dir / f"pytest_importtime_flamegraph_{timestamp}.txt"
                convert_cmd = ["importtime-convert", "--output-format", "flamegraph.pl"]

                convert_result = subprocess.run(
                    convert_cmd,
                    input=result.stderr,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                if convert_result.returncode == 0:
                    with open(flamegraph_path, "w") as f:
                        f.write(convert_result.stdout)
                    artifacts["flamegraph"] = str(flamegraph_path)

            except (subprocess.TimeoutExpired, FileNotFoundError):
                # importtime-convert might not be available, that's okay
                pass

        return startup_time, artifacts

    except Exception:
        return 0.0, {}
    finally:
        # Clean up temporary test file
        Path(baseline_test_path).unlink(missing_ok=True)


def benchmark_fixture(fixture_name: str, fixture_func: Any, baseline_time: float = 0.0) -> FixtureMetrics:
    """Benchmark a single fixture by creating a comparative memory measurement."""
    metrics = FixtureMetrics()

    try:
        # Create two test files: one that uses the fixture, one that doesn't
        minimal_test_content = '''
import pytest
import time
import psutil
import torch

def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def get_gpu_memory_usage_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

def test_minimal():
    """Minimal test for baseline memory measurement."""
    # Record memory state without any fixtures
    memory_usage = get_memory_usage_mb()
    gpu_usage = get_gpu_memory_usage_mb()

    # Store baseline results
    import os
    results_file = "/tmp/baseline_memory_results.txt"
    with open(results_file, "w") as f:
        f.write(str(memory_usage) + "\\n")
        f.write(str(gpu_usage) + "\\n")

    assert True
'''

        fixture_test_content = f'''
import pytest
import time
import psutil
import torch

def get_memory_usage_mb():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def get_gpu_memory_usage_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

def test_fixture_benchmark({fixture_name}):
    """Test that benchmarks the fixture by measuring memory after loading."""
    # Record start time
    start_time = time.perf_counter()

    # Access the fixture to ensure it's fully instantiated
    fixture_value = {fixture_name}

    # Force any lazy evaluation
    if hasattr(fixture_value, '__next__'):
        # It's a generator, get the yielded value
        try:
            fixture_value = next(fixture_value)
        except StopIteration:
            pass

    end_time = time.perf_counter()

    # Record memory state after fixture is loaded
    memory_usage = get_memory_usage_mb()
    gpu_usage = get_gpu_memory_usage_mb()

    # Store results
    import os
    results_file = "/tmp/fixture_benchmark_results.txt"
    with open(results_file, "w") as f:
        f.write(str(end_time - start_time) + "\\n")
        f.write(str(memory_usage) + "\\n")
        f.write(str(gpu_usage) + "\\n")
'''

        # Write the test files in the tests directory
        tests_dir = Path(__file__).parent
        baseline_test_path = tests_dir / f"temp_baseline_{fixture_name}.py"
        fixture_test_path = tests_dir / f"temp_benchmark_{fixture_name}.py"

        try:
            # Run baseline test first
            with open(baseline_test_path, "w") as f:
                f.write(minimal_test_content)

            cmd = [sys.executable, "-m", "pytest", str(baseline_test_path), "-v", "--tb=short", "-p", "no:warnings"]
            baseline_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(tests_dir),
                timeout=60,
            )

            baseline_memory = 0.0
            baseline_gpu = 0.0
            if baseline_result.returncode == 0 and Path("/tmp/baseline_memory_results.txt").exists():
                with open("/tmp/baseline_memory_results.txt", "r") as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        baseline_memory = float(lines[0].strip())
                        baseline_gpu = float(lines[1].strip())
                Path("/tmp/baseline_memory_results.txt").unlink(missing_ok=True)

            # Now run fixture test
            with open(fixture_test_path, "w") as f:
                f.write(fixture_test_content)

            # Run the test with pytest
            start_time = time.perf_counter()
            cmd = [sys.executable, "-m", "pytest", str(fixture_test_path), "-v", "--tb=short", "-p", "no:warnings"]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(tests_dir),
                timeout=120,  # 2 minute timeout per fixture
            )

            end_time = time.perf_counter()
            raw_time = end_time - start_time
            # Subtract baseline pytest startup time to get fixture-only time
            metrics.init_time = max(raw_time - baseline_time, 0.0)

            # Check if the test passed and results file was created
            results_file = "/tmp/fixture_benchmark_results.txt"
            if result.returncode == 0 and Path(results_file).exists():
                # Read the benchmark results
                with open(results_file, "r") as f:
                    lines = f.readlines()
                    if len(lines) >= 3:
                        fixture_memory = float(lines[1].strip())
                        fixture_gpu = float(lines[2].strip())

                        # Calculate memory deltas (positive = fixture uses more memory)
                        metrics.memory_before = baseline_memory
                        metrics.memory_after = fixture_memory
                        metrics.gpu_memory_before = baseline_gpu
                        metrics.gpu_memory_after = fixture_gpu

                        # Clean up
                        Path(results_file).unlink(missing_ok=True)
                    else:
                        metrics.error = "Incomplete benchmark results"
            else:
                # Test failed, capture error
                error_msg = result.stderr.strip() if result.stderr else result.stdout.strip()
                # Extract the actual error without the pytest formatting
                if "fixture" in error_msg and "not found" in error_msg:
                    metrics.error = "Fixture not found"
                elif "ERRORS" in error_msg:
                    # Try to extract specific error
                    error_lines = error_msg.split("\n")
                    for line in error_lines:
                        if "Error" in line or "Exception" in line:
                            metrics.error = f"Error: {line.strip()[:50]}..."
                            break
                    else:
                        metrics.error = "Test had errors"
                else:
                    metrics.error = f"Test failed: {error_msg[:100]}..."

        except subprocess.TimeoutExpired:
            metrics.error = "Timeout (>2min)"
        except Exception as e:
            metrics.error = f"Benchmark error: {str(e)[:50]}..."
        finally:
            # Clean up temporary files
            Path(baseline_test_path).unlink(missing_ok=True)
            Path(fixture_test_path).unlink(missing_ok=True)
            Path("/tmp/fixture_benchmark_results.txt").unlink(missing_ok=True)
            Path("/tmp/baseline_memory_results.txt").unlink(missing_ok=True)

    except Exception as e:
        metrics.error = f"Setup error: {str(e)[:50]}..."

    return metrics


def get_fixture_analysis() -> Dict[str, Dict[str, Any]]:
    """Complete analysis of all fixtures with dynamic discovery and usage counts."""
    raw_fixtures = discover_generated_fixtures()
    usage_counts = count_fixture_usage()

    # Convert tuple format to dictionary format with usage counts and categorization
    processed_fixtures = {}
    for name, (func, scope) in raw_fixtures.items():
        # Categorize the dynamic fixture
        if "gpt2" in name.lower():
            category = FixtureCategory.REAL_MODEL
        elif "llama3" in name.lower() or "gemma2" in name.lower():
            category = FixtureCategory.REAL_MODEL
        elif "cust" in name.lower():
            category = FixtureCategory.CUSTOM_MODEL
        elif "cfg" in name:
            category = FixtureCategory.CONFIG_ONLY
        else:
            category = FixtureCategory.CUSTOM_MODEL

        processed_fixtures[name] = {
            "func": func,
            "scope": scope,
            "uses": usage_counts.get(name, 0),
            "category": category,
            "type": "dynamic",
        }

    return processed_fixtures


def discover_static_fixtures() -> Dict[str, Dict[str, Any]]:
    """Discover static fixtures from conftest.py."""
    conftest_path = Path(__file__).parent / "conftest.py"
    static_fixtures = {}

    try:
        with open(conftest_path, "r") as f:
            content = f.read()

        # Parse AST to find pytest fixtures
        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function has pytest.fixture decorator
                for decorator in node.decorator_list:
                    if (
                        isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Attribute)
                        and decorator.func.attr == "fixture"
                    ):
                        # Extract scope from decorator arguments
                        scope = "function"  # default
                        for keyword in decorator.keywords:
                            if keyword.arg == "scope":
                                if isinstance(keyword.value, ast.Constant):
                                    scope = keyword.value.value

                        static_fixtures[node.name] = {
                            "type": "static",
                            "scope": scope,
                            "category": FixtureCategory.STATIC,
                        }
                        break
                    elif isinstance(decorator, ast.Attribute) and decorator.attr == "fixture":
                        static_fixtures[node.name] = {
                            "type": "static",
                            "scope": "function",
                            "category": FixtureCategory.STATIC,
                        }
                        break

    except Exception as e:
        print(f"Warning: Could not parse static fixtures: {e}")

    return static_fixtures


def run_full_benchmark(max_fixtures: Optional[int] = None) -> Dict[str, Tuple[Dict[str, Any], FixtureMetrics]]:
    """Run comprehensive benchmark of all fixtures."""
    fixtures = get_fixture_analysis()
    static_fixtures = discover_static_fixtures()

    # Add usage counts to static fixtures
    usage_counts = count_fixture_usage()
    for name, metadata in static_fixtures.items():
        metadata["uses"] = usage_counts.get(name, 0)

    all_fixtures = {**fixtures, **static_fixtures}

    # Apply max_fixtures limit for testing
    if max_fixtures:
        fixture_items = list(all_fixtures.items())[:max_fixtures]
        all_fixtures = dict(fixture_items)
        print(f"Limited to first {len(all_fixtures)} fixtures for testing")

    results = {}

    print(f"Benchmarking {len(all_fixtures)} fixtures...")

    # Measure baseline pytest startup time with profiling
    print("Measuring baseline pytest startup time with import profiling...")
    baseline_time, baseline_artifacts = measure_baseline_pytest_startup_with_profiling()
    print(f"Baseline pytest startup time: {baseline_time:.3f}s")
    if baseline_artifacts:
        profiling_dir = Path(__file__).parent / "profiling_artifacts"
        print(f"Import profiling data saved to: {profiling_dir}")
        for artifact_type, path in baseline_artifacts.items():
            relative_path = Path(path).relative_to(Path(__file__).parent)
            print(f"  {artifact_type}: {relative_path}")

    # Import conftest to get access to fixture functions
    sys.path.insert(0, str(Path(__file__).parent.parent))  # Add repo root to path
    sys.path.insert(0, str(Path(__file__).parent))  # Add tests dir to path
    try:
        # Import conftest with proper path setup
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import conftest

        for fixture_name, metadata in all_fixtures.items():
            print(f"Benchmarking {fixture_name}...")

            # Try to get the fixture function - now from metadata dict
            fixture_func = metadata.get("func")
            if fixture_func is None:
                fixture_func = getattr(conftest, fixture_name, None)

            if fixture_func is not None:
                metrics = benchmark_fixture(fixture_name, fixture_func, baseline_time)
            else:
                # For fixtures we can't find, create empty metrics with error
                metrics = FixtureMetrics()
                metrics.error = "Fixture function not found"

            results[fixture_name] = (metadata, metrics)

    finally:
        # Clean up path modifications
        for _ in range(2):  # Remove both paths we added
            if sys.path and str(Path(__file__).parent.parent) in sys.path[0:2]:
                sys.path.pop(0)
            elif sys.path and str(Path(__file__).parent) in sys.path[0:2]:
                sys.path.pop(0)

    return results


def generate_markdown_report(results: Dict[str, Tuple[Dict[str, Any], FixtureMetrics]]) -> str:
    """Generate comprehensive markdown report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name() if gpu_available else "None"

    report = [
        "# Interpretune Fixture Benchmark Report",
        "",
        f"**Generated:** {timestamp}",
        f"**GPU Available:** {gpu_available}",
        f"**GPU Device:** {gpu_name}",
        f"**Total Fixtures Analyzed:** {len(results)}",
        "",
        "## Summary Statistics",
        "",
        "> **Note:** Fixture benchmarking is limited due to pytest fixture dependency",
        "> injection requirements. This report focuses on fixture discovery, usage",
        "> analysis, and categorization. For accurate performance benchmarking,",
        "> fixtures should be tested within a proper pytest session context.",
        "",
    ]

    # Calculate summary statistics
    categories = defaultdict(list)
    scopes = defaultdict(list)
    total_used = 0

    for fixture_name, (metadata, metrics) in results.items():
        categories[metadata["category"]].append(fixture_name)
        scopes[metadata["scope"]].append(fixture_name)
        if metadata["uses"] > 0:
            total_used += 1

    report.extend(
        [
            f"- **Total Fixtures:** {len(results)}",
            f"- **Used Fixtures:** {total_used} ({total_used / len(results) * 100:.1f}%)",
            f"- **Unused Fixtures:** {len(results) - total_used}",
            "",
        ]
    )

    # Category breakdown
    report.extend(
        [
            "### By Category",
            "",
            "| Category | Count | Used | Usage % |",
            "|----------|-------|------|---------|",
        ]
    )

    for category, fixture_list in categories.items():
        used_count = sum(1 for f in fixture_list if results[f][0]["uses"] > 0)
        usage_pct = used_count / len(fixture_list) * 100 if fixture_list else 0
        report.append(f"| {category} | {len(fixture_list)} | {used_count} | {usage_pct:.1f}% |")

    report.extend(
        [
            "",
            "### By Scope",
            "",
            "| Scope | Count | Used | Usage % |",
            "|-------|-------|------|---------|",
        ]
    )

    for scope, fixture_list in scopes.items():
        used_count = sum(1 for f in fixture_list if results[f][0]["uses"] > 0)
        usage_pct = used_count / len(fixture_list) * 100 if fixture_list else 0
        report.append(f"| {scope} | {len(fixture_list)} | {used_count} | {usage_pct:.1f}% |")

    # Detailed fixture table
    report.extend(
        [
            "",
            "## Detailed Fixture Analysis",
            "",
            "| Fixture Name | Uses | Type | Scope | Category | Init Time (s) | Memory Δ (MB) | GPU Δ (MB) | Status |",
            "|--------------|------|------|-------|----------|---------------|---------------|------------|--------|",
        ]
    )

    # Sort by usage count descending, then by name
    sorted_fixtures = sorted(results.items(), key=lambda x: (-x[1][0]["uses"], x[0]))

    for fixture_name, (metadata, metrics) in sorted_fixtures:
        uses = metadata["uses"]
        fixture_type = metadata.get("type", "unknown")
        scope = metadata["scope"]
        category = metadata["category"]

        # Format metrics
        init_time = f"{metrics.init_time:.3f}" if metrics.error is None else "N/A"
        memory_delta = f"{metrics.memory_delta:.1f}" if metrics.error is None else "N/A"
        gpu_delta = f"{metrics.gpu_memory_delta:.1f}" if metrics.error is None and gpu_available else "N/A"
        status = "✅ Success" if metrics.error is None else f"❌ {metrics.error[:30]}..."

        report.append(
            f"| {fixture_name} | {uses} | {fixture_type} | {scope} | {category} | "
            f"{init_time} | {memory_delta} | {gpu_delta} | {status} |"
        )

    # Performance insights
    report.extend(
        [
            "",
            "## Performance Insights",
            "",
        ]
    )

    # Find most expensive fixtures
    successful_benchmarks = [
        (name, metadata, metrics) for name, (metadata, metrics) in results.items() if metrics.error is None
    ]

    if successful_benchmarks:
        # Slowest fixtures
        slowest = sorted(successful_benchmarks, key=lambda x: x[2].init_time, reverse=True)[:5]
        report.extend(
            [
                "### Slowest Fixtures (Top 5)",
                "",
            ]
        )
        for name, metadata, metrics in slowest:
            report.append(f"- **{name}**: {metrics.init_time:.3f}s (Category: {metadata['category']})")

        # Most memory intensive
        memory_intensive = sorted(successful_benchmarks, key=lambda x: x[2].memory_delta, reverse=True)[:5]
        report.extend(
            [
                "",
                "### Most Memory Intensive (Top 5)",
                "",
            ]
        )
        for name, metadata, metrics in memory_intensive:
            report.append(f"- **{name}**: {metrics.memory_delta:.1f}MB (Category: {metadata['category']})")

    # Optimization recommendations
    report.extend(
        [
            "",
            "## Optimization Recommendations",
            "",
        ]
    )

    # High-impact optimization targets
    high_impact_fixtures = [
        name
        for name, (metadata, metrics) in results.items()
        if metadata["uses"] >= 3
        and metadata["scope"] == "session"
        and metadata["category"] in [FixtureCategory.REAL_MODEL, FixtureCategory.CUSTOM_MODEL]
    ]

    if high_impact_fixtures:
        report.extend(
            [
                "### High-Impact Optimization Targets",
                "",
                "These fixtures are heavily used and have session scope, "
                "making them prime candidates for optimization:",
                "",
            ]
        )
        for fixture_name in high_impact_fixtures:
            metadata, metrics = results[fixture_name]
            report.append(f"- **{fixture_name}**: {metadata['uses']} uses, {metadata['category']}")

    # Unused fixtures
    unused_fixtures = [name for name, (metadata, metrics) in results.items() if metadata["uses"] == 0]

    if unused_fixtures:
        report.extend(
            [
                "",
                "### Unused Fixtures Consider for Removal",
                "",
                f"Found {len(unused_fixtures)} unused fixtures that could potentially be removed:",
                "",
            ]
        )
        for fixture_name in unused_fixtures[:10]:  # Show first 10
            metadata, metrics = results[fixture_name]
            report.append(f"- **{fixture_name}**: {metadata['category']}, {metadata['scope']} scope")

        if len(unused_fixtures) > 10:
            report.append(f"- ... and {len(unused_fixtures) - 10} more")

    return "\n".join(report)


def main():
    """Run comprehensive fixture analysis and benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Interpretune fixtures")
    parser.add_argument("--max-fixtures", type=int, help="Maximum number of fixtures to benchmark (for testing)")
    parser.add_argument("--benchmark-startup", action="store_true", help="Profile pytest startup time")
    args = parser.parse_args()

    print("Starting Interpretune Fixture Benchmark...")
    print("=" * 60)

    if args.max_fixtures:
        print(f"Running with --max-fixtures={args.max_fixtures} for quick iteration")

    try:
        if args.benchmark_startup:
            print("Profiling pytest startup time with import analysis...")
            print("=" * 50)
            startup_times = []
            all_artifacts = []
            num_runs = 5

            for i in range(num_runs):
                print(f"Run {i + 1}/{num_runs}...")
                startup_time, artifacts = measure_baseline_pytest_startup_with_profiling()
                startup_times.append(startup_time)
                all_artifacts.append(artifacts)
                print(f"  Startup time: {startup_time:.3f}s")
                if artifacts:
                    print(f"  Profiling artifacts: {len(artifacts)} files generated")

            # Calculate statistics
            avg_time = sum(startup_times) / len(startup_times)
            min_time = min(startup_times)
            max_time = max(startup_times)

            print("\nPytest Startup Profiling Results:")
            print(f"  Average: {avg_time:.3f}s")
            print(f"  Minimum: {min_time:.3f}s")
            print(f"  Maximum: {max_time:.3f}s")
            print(f"  Variance: {max_time - min_time:.3f}s")

            # Report on profiling artifacts
            profiling_dir = Path(__file__).parent / "profiling_artifacts"
            print(f"\nProfiling artifacts saved to: {profiling_dir}")
            for i, artifacts in enumerate(all_artifacts, 1):
                if artifacts:
                    print(f"  Run {i}:")
                    for artifact_type, path in artifacts.items():
                        relative_path = Path(path).relative_to(Path(__file__).parent)
                        print(f"    {artifact_type}: {relative_path}")

            return 0

        # Run full benchmark
        results = run_full_benchmark(max_fixtures=args.max_fixtures)

        # Generate markdown report
        report = generate_markdown_report(results)

        # Write report to file
        report_path = Path(__file__).parent / "fixture_benchmark_report.md"
        with open(report_path, "w") as f:
            f.write(report)

        print("\nBenchmark complete!")
        print(f"Report saved to: {report_path}")
        print(f"Analyzed {len(results)} fixtures")

        # Print summary to console
        used_count = sum(1 for _, (metadata, _) in results.items() if metadata["uses"] > 0)
        print(f"Used fixtures: {used_count}/{len(results)} ({used_count / len(results) * 100:.1f}%)")

    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
