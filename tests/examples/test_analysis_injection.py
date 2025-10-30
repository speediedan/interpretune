"""Test analysis injection framework generalization.

This module tests the updated analysis injection framework.
"""

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import pytest

from tests.runif import RunIf


@RunIf(standalone=True)
def test_framework():
    """Test the generalized analysis injection framework."""
    print("=" * 80)
    print("Testing Analysis Injection Framework Generalization")
    print("=" * 80)

    # Step 1: Create a minimal config
    print("\n1. Creating test configuration...")
    tmp_dir = Path(tempfile.gettempdir()) / "test_analysis_injection"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_dir / "test_config.yaml"

    cfg_text = textwrap.dedent(
        """
        settings:
          enabled: true
          log_to_console: true
          log_to_file: false
          log_dir: /tmp

        shared_context:
          target_tokens: []
        enabled_points: []

        file_hooks:
          test_point:
            file_path: attribution/attribute.py
            regex_pattern: '^\\s*def\\s+attribute\\('
            insert_after: false
            description: "Test hook"

        variants: {}
    """
    )
    cfg_path.write_text(cfg_text)
    print(f"   ✓ Config written to: {cfg_path}")

    # Step 2: Import and test the new API
    print("\n2. Testing new API...")
    try:
        from it_examples.utils.analysis_injection import (
            setup_analysis_injection,
            get_module_debug_info,
            HOOK_REGISTRY,
        )

        print("   ✓ Successfully imported analysis_injection module")
    except ImportError as e:
        print(f"   ✗ Failed to import: {e}")
        pytest.fail(f"Failed to import analysis_injection module: {e}")

    # Step 3: Test orchestrator initialization
    print("\n3. Testing orchestrator initialization...")
    try:
        orchestrator = setup_analysis_injection(config_path=cfg_path, target_package="circuit_tracer")
        print("   ✓ Orchestrator created successfully")
        print(f"   ✓ Target package: {orchestrator.target_package_name}")
        print(f"   ✓ Target path: {orchestrator.target_package_path}")
    except Exception as e:
        print(f"   ✗ Failed to create orchestrator: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Failed to create orchestrator: {e}")

    # Step 4: Verify patched modules
    print("\n4. Verifying patched modules...")
    if orchestrator.patched_modules:
        print(f"   ✓ {len(orchestrator.patched_modules)} module(s) patched:")
        for module_name in orchestrator.patched_modules:
            print(f"     - {module_name}")
    else:
        print("   ⚠ No modules were patched (this is expected for minimal config)")

    # Step 5: Test verification report
    print("\n5. Testing verification report...")
    try:
        report = orchestrator.get_verification_report()
        print("   ✓ Verification report generated:")
        print(textwrap.indent(report, "     "))
    except Exception as e:
        print(f"   ✗ Failed to generate report: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Failed to generate verification report: {e}")

    # Step 6: Test module debug info
    print("\n6. Testing module debug info...")
    try:
        from it_examples.utils.analysis_injection import get_module_debug_info

        target_module = "circuit_tracer.attribution.attribute"
        info = get_module_debug_info(target_module)
        print(f"   ✓ Debug info for {target_module}:")
        print(f"     - Loaded: {info['loaded']}")
        print(f"     - File: {info.get('file_path', 'N/A')}")
        print(f"     - Hook calls: {info.get('hook_call_count', 0)}")
        print(f"     - Has HOOK_REGISTRY: {info.get('has_hook_registry', False)}")
    except Exception as e:
        print(f"   ✗ Failed to get debug info: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Failed to get debug info: {e}")

    # Step 7: Verify HOOK_REGISTRY state
    print("\n7. Verifying HOOK_REGISTRY state...")
    from it_examples.utils.analysis_injection import HOOK_REGISTRY

    print(f"   ✓ Enabled: {HOOK_REGISTRY._enabled}")
    print(f"   ✓ Registered hooks: {len(HOOK_REGISTRY._hooks)}")

    # Step 8: Test teardown
    print("\n8. Testing teardown...")
    try:
        orchestrator.teardown()
        print("   ✓ Teardown successful")
        print(f"   ✓ Hooks disabled: {not HOOK_REGISTRY._enabled}")
    except Exception as e:
        print(f"   ✗ Teardown failed: {e}")
        pytest.fail(f"Teardown failed: {e}")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)


@RunIf(standalone=True)
def test_orchestrator_access():
    """Test that orchestrator supports key access for analysis point data."""
    print("=" * 80)
    print("Testing Orchestrator Key Access")
    print("=" * 80)

    tmp_dir = Path(tempfile.gettempdir()) / "test_key_access"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_dir / "test_key_access_config.yaml"

    cfg_text = textwrap.dedent(
        """
        settings:
            enabled: true
            log_to_console: false
            log_to_file: false
            log_dir: /tmp

        shared_context:
          target_tokens: []

        file_hooks:
            test_point_1:
              file_path: dummy/module.py
              regex_pattern: "test_point_1"
              insert_after: true
              enabled: true
            test_point_2:
              file_path: dummy/module.py
              regex_pattern: "test_point_2"
              insert_after: true
              enabled: true
            test_point_3:
              file_path: dummy/module.py
              regex_pattern: "test_point_3"
              insert_after: true
              enabled: true
        """
    )
    cfg_path.write_text(cfg_text)
    print(f"   ✓ Config written to: {cfg_path}")

    # Step 2: Define test analysis functions
    print("\n2. Defining test analysis functions...")

    def test_point_1(local_vars):
        from it_examples.utils.analysis_injection.orchestrator import analysis_log_point

        analysis_log_point("Test point 1 executed", {"value": 1, "status": "executed"})

    def test_point_2(local_vars):
        from it_examples.utils.analysis_injection.orchestrator import analysis_log_point

        analysis_log_point("Test point 2 executed", {"value": 2, "status": "executed"})

    # test_point_3 will not be executed, should remain None

    # Step 3: Create orchestrator and register hooks
    print("\n3. Creating orchestrator and registering hooks...")
    try:
        from it_examples.utils.analysis_injection import orchestrator

        # Clear any existing data
        orchestrator.clear_analysis_data()

        orchestrator_obj = orchestrator.AnalysisInjectionOrchestrator(
            config_path=cfg_path,
            target_package_path="/tmp",  # Dummy path since we're not patching
            target_package_name="dummy",
        )
        orchestrator_obj.load_config()
        orchestrator_obj.register_hooks()

        print("   ✓ Orchestrator created and hooks registered")
    except Exception as e:
        print(f"   ✗ Failed to create orchestrator: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Failed to create orchestrator: {e}")

    # Step 4: Verify ANALYSIS_DATA initialization
    print("\n4. Verifying ANALYSIS_DATA initialization...")
    analysis_data = orchestrator.get_analysis_data()
    print(f"   ✓ ANALYSIS_DATA keys: {list(analysis_data.keys())}")
    assert len(analysis_data) == 3, f"Expected 3 points, got {len(analysis_data)}"
    assert "test_point_1" in analysis_data, "test_point_1 should be initialized"
    assert "test_point_2" in analysis_data, "test_point_2 should be initialized"
    assert "test_point_3" in analysis_data, "test_point_3 should be initialized"
    assert analysis_data["test_point_1"] is None, "test_point_1 should be None initially"
    assert analysis_data["test_point_2"] is None, "test_point_2 should be None initially"
    assert analysis_data["test_point_3"] is None, "test_point_3 should be None initially"
    print("   ✓ All points initialized with None")

    # Step 5: Test key access via orchestrator
    print("\n5. Testing key access via orchestrator...")
    try:
        # Test accessing initialized but not executed points
        result1 = orchestrator_obj["test_point_1"]
        assert result1 is None, f"Expected None for test_point_1, got {result1}"
        print("   ✓ test_point_1 accessible and returns None")

        result2 = orchestrator_obj["test_point_2"]
        assert result2 is None, f"Expected None for test_point_2, got {result2}"
        print("   ✓ test_point_2 accessible and returns None")

        result3 = orchestrator_obj["test_point_3"]
        assert result3 is None, f"Expected None for test_point_3, got {result3}"
        print("   ✓ test_point_3 accessible and returns None")

    except Exception as e:
        print(f"   ✗ Key access failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Key access failed: {e}")

    # Step 6: Simulate executing analysis points
    print("\n6. Simulating analysis point execution...")
    try:
        # Manually execute the functions to simulate hook execution
        test_point_1({})
        test_point_2({})

        print("   ✓ Analysis functions executed")
    except Exception as e:
        print(f"   ✗ Failed to execute analysis functions: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Failed to execute analysis functions: {e}")

    # Step 7: Verify executed points have data
    print("\n7. Verifying executed points have data...")
    analysis_data_after = orchestrator.get_analysis_data()
    print(f"   ✓ ANALYSIS_DATA after execution: {dict(analysis_data_after)}")

    # Check executed points
    expected_1 = {"value": 1, "status": "executed"}
    assert analysis_data_after["test_point_1"] == expected_1, (
        f"test_point_1 data incorrect: {analysis_data_after['test_point_1']}"
    )

    expected_2 = {"value": 2, "status": "executed"}
    assert analysis_data_after["test_point_2"] == expected_2, (
        f"test_point_2 data incorrect: {analysis_data_after['test_point_2']}"
    )

    assert analysis_data_after["test_point_3"] is None, (
        f"test_point_3 should still be None: {analysis_data_after['test_point_3']}"
    )
    print("   ✓ Executed points have correct data")

    # Step 8: Test key access for executed points
    print("\n8. Testing key access for executed points...")
    try:
        result1_exec = orchestrator_obj["test_point_1"]
        assert result1_exec == {"value": 1, "status": "executed"}, f"test_point_1 access failed: {result1_exec}"
        print("   ✓ test_point_1 accessible with execution data")

        result2_exec = orchestrator_obj["test_point_2"]
        assert result2_exec == {"value": 2, "status": "executed"}, f"test_point_2 access failed: {result2_exec}"
        print("   ✓ test_point_2 accessible with execution data")

        result3_none = orchestrator_obj["test_point_3"]
        assert result3_none is None, f"test_point_3 should be None: {result3_none}"
        print("   ✓ test_point_3 accessible and still None")

    except Exception as e:
        print(f"   ✗ Key access for executed points failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Key access for executed points failed: {e}")

    # Step 9: Test for non-existent point
    try:
        _ = orchestrator_obj["non_existent_point"]
        pytest.fail("Should have raised KeyError for non-existent point")
    except KeyError:
        print("   ✓ Correctly raised for non-existent point")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        pytest.fail(f"Unexpected error for non-existent point: {e}")

    # Step 10: Test iteration methods
    print("\n10. Testing iteration methods...")
    try:
        # Test keys()
        keys_list = list(orchestrator_obj.keys())
        expected_keys = ["test_point_1", "test_point_2", "test_point_3"]
        assert set(keys_list) == set(expected_keys), f"Keys mismatch: {keys_list} vs {expected_keys}"
        print("   ✓ keys() returns correct point IDs")

        # Test values()
        values_list = list(orchestrator_obj.values())
        expected_values = [expected_1, expected_2, None]
        assert values_list == expected_values, f"Values mismatch: {values_list}"
        print("   ✓ values() returns correct data values")

        # Test items()
        items_list = list(orchestrator_obj.items())
        expected_items = [("test_point_1", expected_1), ("test_point_2", expected_2), ("test_point_3", None)]
        assert items_list == expected_items, f"Items mismatch: {items_list} vs {expected_items}"
        print("   ✓ items() returns correct (key, value) pairs")

        # Test direct iteration
        iterated_keys = list(orchestrator_obj)
        assert set(iterated_keys) == set(expected_keys), f"Iteration mismatch: {iterated_keys} vs {expected_keys}"
        print("   ✓ Direct iteration works correctly")

        # Test dict comprehension syntax
        comprehension_result = {ap: data for ap, data in orchestrator_obj.items()}
        expected_dict = {"test_point_1": expected_1, "test_point_2": expected_2, "test_point_3": None}
        assert comprehension_result == expected_dict, (
            f"Dict comprehension mismatch: {comprehension_result} vs {expected_dict}"
        )
        print("   ✓ Dict comprehension syntax works")

    except Exception as e:
        print(f"   ✗ Iteration methods failed: {e}")
        import traceback

        traceback.print_exc()
        pytest.fail(f"Iteration methods failed: {e}")

    # Step 11: Clean up
    print("\n11. Cleaning up...")
    orchestrator.clear_analysis_data()
    print("   ✓ Analysis data cleared")

    print("\n" + "=" * 80)
    print("✅ Orchestrator iteration test passed!")
    print("=" * 80)
