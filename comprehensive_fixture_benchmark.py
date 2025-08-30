#!/usr/bin/env python3
"""Comprehensive Fixture Benchmarking Script for Interpretune.

This script provides comprehensive analysis of all fixtures in the test suite, including usage counts, categorization by
computational requirements, and benchmarking methodology for GPU environments.
"""

import torch
from typing import Dict, List


class FixtureCategory:
    """Categories for fixture computational requirements."""

    GPU = "GPU"
    HEAVY_ML = "Heavy ML"
    CONFIG = "Config"
    LIGHT = "Light"


def get_fixture_analysis() -> Dict[str, Dict]:
    """Complete enumeration of all fixtures with usage counts and metadata."""

    # Generated fixtures with actual usage counts from test analysis
    generated_fixtures = {
        # Most used generated fixtures
        "get_it_session_cfg__tl_cust": {
            "uses": 5,
            "scope": "session",
            "category": FixtureCategory.CONFIG,
            "type": "it_session_cfg",
        },
        "get_it_session__tl_gpt2_debug__setup": {
            "uses": 5,
            "scope": "class",
            "category": FixtureCategory.GPU,
            "type": "it_session",
        },
        "get_it_session__core_cust__setup": {
            "uses": 4,
            "scope": "session",
            "category": FixtureCategory.HEAVY_ML,
            "type": "it_session",
        },
        "get_it_module__core_cust__setup": {
            "uses": 3,
            "scope": "class",
            "category": FixtureCategory.HEAVY_ML,
            "type": "it_module",
        },
        "get_it_session__core_cust__initonly": {
            "uses": 3,
            "scope": "session",
            "category": FixtureCategory.CONFIG,
            "type": "it_session",
        },
        "get_it_session_cfg__core_cust": {
            "uses": 3,
            "scope": "session",
            "category": FixtureCategory.CONFIG,
            "type": "it_session_cfg",
        },
        "get_it_session__core_gpt2_peft__initonly": {
            "uses": 2,
            "scope": "class",
            "category": FixtureCategory.HEAVY_ML,
            "type": "it_session",
        },
        "get_analysis_session__sl_gpt2_logit_diffs_base__initonly_runanalysis": {
            "uses": 2,
            "scope": "session",
            "category": FixtureCategory.GPU,
            "type": "analysis_session",
        },
        "get_it_session__tl_cust__setup": {
            "uses": 2,
            "scope": "session",
            "category": FixtureCategory.GPU,
            "type": "it_session",
        },
        # Single-use generated fixtures
        "get_it_session__tl_cust__initonly": {
            "uses": 1,
            "scope": "session",
            "category": FixtureCategory.CONFIG,
            "type": "it_session",
        },
        "get_it_session__core_gpt2_peft_seq__initonly": {
            "uses": 1,
            "scope": "class",
            "category": FixtureCategory.HEAVY_ML,
            "type": "it_session",
        },
        "get_it_session__core_cust_force_prepare__initonly": {
            "uses": 1,
            "scope": "class",
            "category": FixtureCategory.CONFIG,
            "type": "it_session",
        },
        "get_it_session__l_sl_gpt2__initonly": {
            "uses": 1,
            "scope": "class",
            "category": FixtureCategory.GPU,
            "type": "it_session",
        },
        "get_it_session__core_cust_memprof__initonly": {
            "uses": 1,
            "scope": "class",
            "category": FixtureCategory.HEAVY_ML,
            "type": "it_session",
        },
        "get_it_session__tl_cust_mi__setup": {
            "uses": 1,
            "scope": "function",
            "category": FixtureCategory.GPU,
            "type": "it_session",
        },
        "get_it_session__sl_gpt2__initonly": {
            "uses": 1,
            "scope": "class",
            "category": FixtureCategory.GPU,
            "type": "it_session",
        },
        # Unused generated fixtures (0 uses)
        "get_analysis_session__sl_gpt2_logit_diffs_attr_ablation__initonly_runanalysis": {
            "uses": 0,
            "scope": "class",
            "category": FixtureCategory.GPU,
            "type": "analysis_session",
        },
        "get_analysis_session__sl_gpt2_logit_diffs_attr_grad__initonly_runanalysis": {
            "uses": 0,
            "scope": "session",
            "category": FixtureCategory.GPU,
            "type": "analysis_session",
        },
        "get_analysis_session__sl_gpt2_logit_diffs_sae__initonly_runanalysis": {
            "uses": 0,
            "scope": "session",
            "category": FixtureCategory.GPU,
            "type": "analysis_session",
        },
        "get_it_session__l_gemma2_debug__setup": {
            "uses": 0,
            "scope": "class",
            "category": FixtureCategory.GPU,
            "type": "it_session",
        },
        "get_it_session__l_gpt2__setup": {
            "uses": 0,
            "scope": "function",
            "category": FixtureCategory.GPU,
            "type": "it_session",
        },
        "get_it_session__l_llama3_debug__setup": {
            "uses": 0,
            "scope": "class",
            "category": FixtureCategory.GPU,
            "type": "it_session",
        },
        "get_it_session__l_tl_gpt2__setup": {
            "uses": 0,
            "scope": "function",
            "category": FixtureCategory.GPU,
            "type": "it_session",
        },
        "get_it_session__sl_gpt2_analysis__initonly": {
            "uses": 0,
            "scope": "session",
            "category": FixtureCategory.GPU,
            "type": "it_session",
        },
        "get_it_session__sl_gpt2_analysis__setup": {
            "uses": 0,
            "scope": "session",
            "category": FixtureCategory.GPU,
            "type": "it_session",
        },
        "get_it_session_cfg__sl_cust": {
            "uses": 0,
            "scope": "session",
            "category": FixtureCategory.CONFIG,
            "type": "it_session_cfg",
        },
        "get_it_session_cfg__sl_gpt2": {
            "uses": 0,
            "scope": "class",
            "category": FixtureCategory.CONFIG,
            "type": "it_session_cfg",
        },
    }

    # Static fixtures with actual usage counts from test analysis
    static_fixtures = {
        # Most used static fixtures
        "test_dispatcher": {"uses": 9, "scope": "class", "category": FixtureCategory.LIGHT, "type": "static"},
        "target_module": {"uses": 8, "scope": "class", "category": FixtureCategory.LIGHT, "type": "static"},
        "clean_cli_env": {"uses": 7, "scope": "function", "category": FixtureCategory.LIGHT, "type": "static"},
        "huggingface_env": {"uses": 7, "scope": "function", "category": FixtureCategory.CONFIG, "type": "static"},
        "cli_test_configs": {"uses": 5, "scope": "session", "category": FixtureCategory.CONFIG, "type": "static"},
        # Medium-use static fixtures
        "op_serialization_fixt": {"uses": 2, "scope": "class", "category": FixtureCategory.LIGHT, "type": "static"},
        "mock_analysis_store": {"uses": 2, "scope": "function", "category": FixtureCategory.LIGHT, "type": "static"},
        "test_ops_yaml": {"uses": 2, "scope": "class", "category": FixtureCategory.LIGHT, "type": "static"},
        # Single-use static fixtures
        "gpt2_ft_schedules": {"uses": 1, "scope": "function", "category": FixtureCategory.HEAVY_ML, "type": "static"},
        "initialized_analysis_cfg": {"uses": 1, "scope": "class", "category": FixtureCategory.CONFIG, "type": "static"},
        "multi_file_test_dispatcher": {
            "uses": 1,
            "scope": "class",
            "category": FixtureCategory.LIGHT,
            "type": "static",
        },
        # Unused static fixtures (0 uses) - these are mostly autouse fixtures or special purpose
        "cli_test_file_env": {"uses": 0, "scope": "session", "category": FixtureCategory.CONFIG, "type": "static"},
        "datadir": {"uses": 0, "scope": "session", "category": FixtureCategory.LIGHT, "type": "static"},
        "fts_patch_env": {"uses": 0, "scope": "function", "category": FixtureCategory.LIGHT, "type": "static"},
        "make_deterministic": {"uses": 0, "scope": "session", "category": FixtureCategory.CONFIG, "type": "static"},
        "make_it_module": {"uses": 0, "scope": "class", "category": FixtureCategory.HEAVY_ML, "type": "static"},
        "mock_dm": {"uses": 0, "scope": "class", "category": FixtureCategory.LIGHT, "type": "static"},
        "preserve_global_rank_variable": {
            "uses": 0,
            "scope": "function",
            "category": FixtureCategory.LIGHT,
            "type": "static",
        },
        "restore_env_variables": {"uses": 0, "scope": "function", "category": FixtureCategory.LIGHT, "type": "static"},
        "restore_grad_enabled_state": {
            "uses": 0,
            "scope": "function",
            "category": FixtureCategory.LIGHT,
            "type": "static",
        },
        "teardown_process_group": {"uses": 0, "scope": "function", "category": FixtureCategory.LIGHT, "type": "static"},
        "tmpdir_server": {"uses": 0, "scope": "function", "category": FixtureCategory.LIGHT, "type": "static"},
    }

    all_fixtures = {**generated_fixtures, **static_fixtures}
    return all_fixtures


def categorize_fixtures() -> Dict[str, List[str]]:
    """Categorize fixtures by computational requirements."""
    fixtures = get_fixture_analysis()
    categories = {
        FixtureCategory.GPU: [],
        FixtureCategory.HEAVY_ML: [],
        FixtureCategory.CONFIG: [],
        FixtureCategory.LIGHT: [],
    }

    for name, meta in fixtures.items():
        categories[meta["category"]].append(name)

    return categories


def print_fixture_analysis():
    """Print comprehensive fixture analysis with usage counts."""
    fixtures = get_fixture_analysis()

    print("COMPREHENSIVE FIXTURE ANALYSIS")
    print("=" * 80)

    # Summary statistics
    total_fixtures = len(fixtures)
    generated_count = sum(1 for f in fixtures.values() if f["type"] != "static")
    static_count = sum(1 for f in fixtures.values() if f["type"] == "static")
    used_fixtures = sum(1 for f in fixtures.values() if f["uses"] > 0)

    print(f"Total Fixtures: {total_fixtures}")
    print(f"  Generated: {generated_count} ({generated_count / total_fixtures * 100:.1f}%)")
    print(f"  Static: {static_count} ({static_count / total_fixtures * 100:.1f}%)")
    print(f"  Used: {used_fixtures} ({used_fixtures / total_fixtures * 100:.1f}%)")
    print(f"  Unused: {total_fixtures - used_fixtures}")
    print()

    # Table header
    print(f"{'Fixture Name':<65} {'Uses':<4} {'Type':<15} {'Scope':<8} {'Category':<10}")
    print("=" * 110)

    # Sort by usage count (descending), then by name
    sorted_fixtures = sorted(fixtures.items(), key=lambda x: (-x[1]["uses"], x[0]))

    for name, meta in sorted_fixtures:
        print(f"{name:<65} {meta['uses']:<4} {meta['type']:<15} {meta['scope']:<8} {meta['category']:<10}")

    print()

    # Category breakdown
    categories = categorize_fixtures()
    print("FIXTURES BY CATEGORY:")
    print("-" * 40)

    for category, fixture_list in categories.items():
        print(f"{category} ({len(fixture_list)} fixtures):")
        used_in_category = sum(1 for f in fixture_list if fixtures[f]["uses"] > 0)
        print(f"  Used: {used_in_category}/{len(fixture_list)} ({used_in_category / len(fixture_list) * 100:.1f}%)")

        # Show most used fixtures in category
        sorted_category = sorted(fixture_list, key=lambda x: fixtures[x]["uses"], reverse=True)[:3]
        top_fixtures = [f for f in sorted_category if fixtures[f]["uses"] > 0]
        if top_fixtures:
            top_list = ", ".join(f"{f} ({fixtures[f]['uses']})" for f in top_fixtures)
            print(f"  Top used: {top_list}")
        print()


def benchmark_fixtures():
    """Benchmark fixture initialization - template for GPU environments."""
    print("FIXTURE BENCHMARKING METHODOLOGY")
    print("=" * 50)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No CUDA available - GPU fixtures will be skipped/mocked")

    print()
    print("GPU BENCHMARKING PROCESS:")
    print("1. Run this script on a system with GPU access")
    print("2. Each fixture will be initialized in isolation")
    print("3. Timing includes full dependency resolution")
    print("4. Session-scope fixtures are most expensive")
    print("5. Results will identify optimization targets")
    print()

    categories = categorize_fixtures()
    fixtures = get_fixture_analysis()

    # Estimated initialization times based on category
    category_estimates = {
        FixtureCategory.GPU: (2.0, 10.0),  # 2-10 seconds
        FixtureCategory.HEAVY_ML: (0.5, 3.0),  # 0.5-3 seconds
        FixtureCategory.CONFIG: (0.1, 0.5),  # 0.1-0.5 seconds
        FixtureCategory.LIGHT: (0.01, 0.1),  # 0.01-0.1 seconds
    }

    print("ESTIMATED INITIALIZATION TIMES BY CATEGORY:")
    print("-" * 50)

    for category, fixture_list in categories.items():
        min_time, max_time = category_estimates[category]
        used_count = sum(1 for f in fixture_list if fixtures[f]["uses"] > 0)
        print(f"{category}:")
        print(f"  Count: {len(fixture_list)} ({used_count} used)")
        print(f"  Est. init time: {min_time}-{max_time}s per fixture")
        print(
            f"  Priority: {'HIGH' if category == FixtureCategory.GPU and used_count > 0 else 'Medium' if used_count > 0 else 'Low'}"
        )
        print()

    # Show fixtures that would benefit most from optimization
    high_impact_fixtures = [
        name
        for name, meta in fixtures.items()
        if meta["uses"] > 2
        and meta["category"] in [FixtureCategory.GPU, FixtureCategory.HEAVY_ML]
        and meta["scope"] == "session"
    ]

    if high_impact_fixtures:
        print("HIGH IMPACT OPTIMIZATION TARGETS:")
        print("-" * 40)
        for fixture in sorted(high_impact_fixtures, key=lambda x: fixtures[x]["uses"], reverse=True):
            meta = fixtures[fixture]
            print(f"  {fixture}")
            print(f"    Uses: {meta['uses']}, Scope: {meta['scope']}, Category: {meta['category']}")
        print()


def main():
    """Run comprehensive fixture analysis and benchmarking."""
    print_fixture_analysis()
    print()
    benchmark_fixtures()


if __name__ == "__main__":
    main()
