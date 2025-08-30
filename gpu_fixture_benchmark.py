#!/usr/bin/env python3
"""GPU Fixture Benchmarking Script for Interpretune.

Run this script on a system with GPU access to benchmark fixture initialization times.
"""

import time
import torch


def benchmark_gpu_fixtures():
    """Benchmark GPU-dependent fixtures."""
    print("GPU Fixture Benchmarking")
    print("=" * 40)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("No CUDA available - some fixtures may be skipped")

    # List of fixtures to benchmark (add GPU-requiring fixtures here)
    gpu_fixtures = [
        "make_deterministic",
        "get_it_session__tl_cust__setup",
        "get_it_session__l_gpt2__setup",
        "get_it_session__sl_gpt2_analysis__setup",
        # Add more GPU fixtures as identified
    ]

    results = {}

    for fixture_name in gpu_fixtures:
        print(f"\nBenchmarking {fixture_name}...")
        start_time = time.time()
        try:
            # This would need to be adapted based on how fixtures are accessed
            # In practice, you'd use pytest's fixture system
            print("  [Placeholder - implement fixture access]")
            init_time = time.time() - start_time
            results[fixture_name] = init_time
            print(f"  Time: {init_time:.3f}s")
        except Exception as e:
            print(f"  Error: {e}")
            results[fixture_name] = float("inf")

    print("\nGPU Fixture Timing Summary:")
    for name, time_taken in sorted(results.items(), key=lambda x: x[1]):
        if time_taken != float("inf"):
            print(f"  {name:<40}: {time_taken:.3f}s")
        else:
            print(f"  {name:<40}: FAILED")


if __name__ == "__main__":
    benchmark_gpu_fixtures()
