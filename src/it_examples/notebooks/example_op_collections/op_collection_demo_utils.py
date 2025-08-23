"""Utility functions for operation collection demonstration notebooks.

This module provides helper functions for setting up local and hub operation collections, including path management and
file copying operations.
"""

import os
import shutil
import sys
import io
import contextlib
from pathlib import Path
from typing import Tuple, Generator

from interpretune.utils import rank_zero_warn
from interpretune.analysis import IT_ANALYSIS_OP_PATHS


def purge_it_modules_from_sys() -> None:
    """Remove interpretune modules from sys.modules to force reimport.

    This function cleans up all interpretune-related modules from the Python module cache, forcing a fresh import when
    interpretune is next imported.
    """
    modules_to_remove = [name for name in sys.modules.keys() if name.startswith("interpretune")]
    for module_name in modules_to_remove:
        del sys.modules[module_name]


def categorize_operations(operation_definitions: dict) -> tuple:
    """Categorize operations into canonical, hub, local, composed, and builtin operations.

    Args:
        operation_definitions: Dictionary of operation definitions from DISPATCHER.registered_ops

    Returns:
        Tuple of (canonical_ops, alias_map, hub_ops, local_ops, composed_ops, builtin_ops)
    """
    # Get canonical names (resolve aliases to their target operations)
    canonical_ops = {}
    alias_map = {}
    for op_name, op_def in operation_definitions.items():
        canonical_name = op_def.name
        if canonical_name not in canonical_ops:
            canonical_ops[canonical_name] = op_def
            alias_map[canonical_name] = []
        if op_name != canonical_name:
            alias_map[canonical_name].append(op_name)

    # Count hub operations (those with dots in their canonical names indicating namespacing)
    hub_ops = {name: op_def for name, op_def in canonical_ops.items() if "." in name}

    # Count local operations (those without dots but not in the built-in list)
    builtin_ops = {
        "labels_to_ids",
        "get_answer_indices",
        "get_alive_latents",
        "model_forward",
        "model_cache_forward",
        "model_ablation",
        "model_gradient",
        "logit_diffs",
        "logit_diffs_cache",
        "sae_correct_acts",
        "gradient_attribution",
        "ablation_attribution",
    }

    local_ops = {
        name: op_def
        for name, op_def in canonical_ops.items()
        if "." not in name and name not in builtin_ops and op_def.composition is None
    }

    # Count composed operations
    composed_ops = {name: op_def for name, op_def in canonical_ops.items() if op_def.composition is not None}

    return canonical_ops, alias_map, hub_ops, local_ops, composed_ops, builtin_ops


def generate_op_summary(operation_definitions: dict, title: str = "Operation Summary") -> None:
    """Generate and print a comprehensive operation summary.

    Args:
        operation_definitions: Dictionary of operation definitions from DISPATCHER.registered_ops
        title: Title for the summary section
    """
    total_ops = len(operation_definitions)
    canonical_ops, alias_map, hub_ops, local_ops, composed_ops, builtin_ops = categorize_operations(
        operation_definitions
    )

    print(f"\n📊 {title}:")
    print(f"  Total registered names: {total_ops}")
    print(f"  Unique operations: {len(canonical_ops)}")
    print(f"  Hub operations: {len(hub_ops)}")
    print(f"  Local operations: {len(local_ops)}")
    print(f"  Composed operations: {len(composed_ops)}")
    print(f"  Built-in operations: {len(builtin_ops)}")

    print("\n🌐 Hub operations found:")
    for op_name, op_def in hub_ops.items():
        aliases = alias_map.get(op_name, [])
        all_names = [op_name] + aliases
        print(f"  - {op_name} (accessible as: {', '.join(all_names)})")

    print("\n🏠 Local operations found:")
    for op_name, op_def in local_ops.items():
        aliases = alias_map.get(op_name, [])
        all_names = [op_name] + aliases
        print(f"  - {op_name} (accessible as: {', '.join(all_names)}) - {op_def.description}")


def verify_cleanup_status(operation_definitions: dict, title: str = "Operation Summary After Cleanup") -> None:
    """Generate operation summary with cleanup verification status.

    Args:
        operation_definitions: Dictionary of operation definitions from DISPATCHER.registered_ops
        title: Title for the summary section
    """
    total_ops = len(operation_definitions)
    canonical_ops, alias_map, hub_ops, local_ops, composed_ops, builtin_ops = categorize_operations(
        operation_definitions
    )

    print(f"\n📊 {title}:")
    print(f"  Total registered names: {total_ops}")
    print(f"  Unique operations: {len(canonical_ops)}")
    print(f"  Hub operations: {len(hub_ops)}")
    print(f"  Local operations: {len(local_ops)}")
    print(f"  Composed operations: {len(composed_ops)}")
    print(f"  Built-in operations: {len(builtin_ops)}")

    if len(hub_ops) == 0:
        print("\n✅ Success: No hub operations found - cleanup successful!")
    else:
        print(f"\n❌ Warning: {len(hub_ops)} hub operations still present:")
        for op_name, op_def in hub_ops.items():
            aliases = alias_map.get(op_name, [])
            all_names = [op_name] + aliases
            print(f"  - {op_name} (accessible as: {', '.join(all_names)})")

    if len(local_ops) > 0:
        print("\n🏠 Local operations still available:")
        for op_name, op_def in local_ops.items():
            aliases = alias_map.get(op_name, [])
            all_names = [op_name] + aliases
            print(f"  - {op_name} (accessible as: {', '.join(all_names)}) - {op_def.description}")
    else:
        print("\n⚠️ No local operations found")

    # Show remaining operations by category
    print("\n📋 Detailed breakdown:")
    print(f"\n  Built-in operations ({len(builtin_ops)}):")
    for name in sorted(builtin_ops):
        if name in canonical_ops:
            aliases = alias_map.get(name, [])
            all_names = [name] + aliases
            print(f"    - {name} (accessible as: {', '.join(all_names)})")

    if composed_ops:
        print(f"\n  Composed operations ({len(composed_ops)}):")
        for name in sorted(composed_ops.keys()):
            aliases = alias_map.get(name, [])
            all_names = [name] + aliases
            print(f"    - {name} (accessible as: {', '.join(all_names)})")


def print_env_summary(
    interpretune_version: str,
    IT_ANALYSIS_CACHE,
    IT_MODULES_CACHE,
    IT_ANALYSIS_HUB_CACHE,
    IT_ANALYSIS_OP_PATHS,
    example_hub_op_collection_dir,
    example_local_op_collection_dir,
) -> None:
    """Print a summary of the current interpretune environment and paths.

    Args:
        interpretune_version: Version string of interpretune
        IT_ANALYSIS_CACHE: Analysis cache location
        IT_MODULES_CACHE: Modules cache location
        IT_ANALYSIS_HUB_CACHE: Hub cache location
        IT_ANALYSIS_OP_PATHS: Current analysis operation paths
        example_hub_op_collection_dir: Path to example hub op collection
        example_local_op_collection_dir: Path to example local op collection
    """
    print(f"Interpretune version: {interpretune_version}")
    print(f"Current analysis cache location: {IT_ANALYSIS_CACHE}")
    print(f"Current modules cache location: {IT_MODULES_CACHE}")
    print(f"Current hub cache location: {IT_ANALYSIS_HUB_CACHE}")
    print(f"Current IT analysis op paths: {IT_ANALYSIS_OP_PATHS}")
    print(f"This notebook's example hub op collection directory: {example_hub_op_collection_dir}")
    print(f"This notebook's example local op collection directory: {example_local_op_collection_dir}")


def setup_local_op_collection(
    source_local_op_collection: Path, tmp_local_op_collection: Path = Path("/tmp/local_op_collection")
) -> Tuple[str, str]:
    """Setup local operation collection by copying to /tmp/ and updating environment variables.

    Args:
        source_local_op_collection: Source path of the local op collection
        tmp_local_op_collection: Destination path for the copied collection

    Returns:
        Tuple of (original_op_paths_env, new_op_paths) for cleanup purposes

    Raises:
        FileNotFoundError: If source local op_collection doesn't exist
    """
    print(f"Source local op_collection: {source_local_op_collection}")
    print(f"Destination: {tmp_local_op_collection}")

    # Check if source exists
    if not source_local_op_collection.exists():
        raise FileNotFoundError(f"Source local op_collection not found at {source_local_op_collection}")

    # Warn if destination already exists
    if tmp_local_op_collection.exists():
        rank_zero_warn(f"Destination folder {tmp_local_op_collection} already exists and will be overwritten!")
        shutil.rmtree(tmp_local_op_collection)

    # Copy the local op collection folder
    shutil.copytree(source_local_op_collection, tmp_local_op_collection)
    print(f"✓ Successfully copied local op_collection to {tmp_local_op_collection}")

    # Store the original IT_ANALYSIS_OP_PATHS environment variable
    original_op_paths_env = os.environ.get("IT_ANALYSIS_OP_PATHS", "")
    print(f"Original IT_ANALYSIS_OP_PATHS environment variable: '{original_op_paths_env}'")

    # Set the IT_ANALYSIS_OP_PATHS environment variable
    # The format is a colon-separated list of paths
    new_op_paths = str(tmp_local_op_collection)
    if original_op_paths_env:
        new_op_paths = f"{original_op_paths_env}:{new_op_paths}"

    os.environ["IT_ANALYSIS_OP_PATHS"] = new_op_paths
    print(f"✓ Set IT_ANALYSIS_OP_PATHS environment variable to: '{new_op_paths}'")

    # Also update the imported list for consistency (this was the old approach)
    if str(tmp_local_op_collection) not in IT_ANALYSIS_OP_PATHS:
        IT_ANALYSIS_OP_PATHS.append(str(tmp_local_op_collection))
        print(f"✓ Also added {tmp_local_op_collection} to imported IT_ANALYSIS_OP_PATHS list")

    print(f"\nUpdated IT_ANALYSIS_OP_PATHS list: {IT_ANALYSIS_OP_PATHS}")
    print(f"Current IT_ANALYSIS_OP_PATHS env var: '{os.environ.get('IT_ANALYSIS_OP_PATHS', '')}'")

    # Verify contents
    print("\nContents of copied local op_collection:")
    for item in tmp_local_op_collection.iterdir():
        print(f"  - {item.name}")

    return original_op_paths_env, new_op_paths


def setup_hub_op_collection(
    source_op_collection: Path, tmp_op_collection: Path = Path("/tmp/hub_op_collection")
) -> None:
    """Setup hub operation collection by copying to /tmp/ for upload to the hub.

    Args:
        source_op_collection: Source path of the hub op collection
        tmp_op_collection: Destination path for the copied collection

    Raises:
        FileNotFoundError: If source op_collection doesn't exist
    """
    print(f"Source hub op_collection: {source_op_collection}")
    print(f"Destination: {tmp_op_collection}")

    # Check if source exists
    if not source_op_collection.exists():
        raise FileNotFoundError(f"Source op_collection not found at {source_op_collection}")

    # Warn if destination already exists
    if tmp_op_collection.exists():
        rank_zero_warn(f"Destination folder {tmp_op_collection} already exists and will be overwritten!")
        shutil.rmtree(tmp_op_collection)

    # Copy the folder
    shutil.copytree(source_op_collection, tmp_op_collection)
    print(f"✓ Successfully copied hub op_collection to {tmp_op_collection}")

    # Verify contents
    print("\nContents of copied hub op_collection:")
    for item in tmp_op_collection.iterdir():
        print(f"  - {item.name}")


def cleanup_op_collections(
    tmp_op_collection: Path = Path("/tmp/hub_op_collection"),
    tmp_local_op_collection: Path = Path("/tmp/local_op_collection"),
    original_op_paths_env: str = "",
) -> None:
    """Clean up temporary operation collection files and restore environment variables.

    Args:
        tmp_op_collection: Path to temporary hub op collection
        tmp_local_op_collection: Path to temporary local op collection
        original_op_paths_env: Original value of IT_ANALYSIS_OP_PATHS environment variable
    """
    print("Cleaning up temporary files...")

    # Remove temporary hub op_collection
    if tmp_op_collection.exists():
        shutil.rmtree(tmp_op_collection)
        print(f"✓ Removed temporary hub op_collection: {tmp_op_collection}")

    # Remove temporary local op_collection
    if tmp_local_op_collection.exists():
        shutil.rmtree(tmp_local_op_collection)
        print(f"✓ Removed temporary local op_collection: {tmp_local_op_collection}")

    # Restore the original IT_ANALYSIS_OP_PATHS environment variable
    if original_op_paths_env:
        os.environ["IT_ANALYSIS_OP_PATHS"] = original_op_paths_env
        print(f"✓ Restored IT_ANALYSIS_OP_PATHS environment variable to: '{original_op_paths_env}'")
    else:
        if "IT_ANALYSIS_OP_PATHS" in os.environ:
            del os.environ["IT_ANALYSIS_OP_PATHS"]
            print("✓ Unset IT_ANALYSIS_OP_PATHS environment variable")

    # Remove from the imported IT_ANALYSIS_OP_PATHS list
    if str(tmp_local_op_collection) in IT_ANALYSIS_OP_PATHS:
        IT_ANALYSIS_OP_PATHS.remove(str(tmp_local_op_collection))
        print(f"✓ Removed {tmp_local_op_collection} from imported IT_ANALYSIS_OP_PATHS list")

    print(f"\nFinal IT_ANALYSIS_OP_PATHS list: {IT_ANALYSIS_OP_PATHS}")
    print(f"Final IT_ANALYSIS_OP_PATHS env var: '{os.environ.get('IT_ANALYSIS_OP_PATHS', '')}'")


def cleanup_hub_repository(download_result) -> None:
    """Remove only the specific repository that was downloaded, not the entire hub cache.

    Args:
        download_result: The result object from hub_manager.download_ops()
    """
    if download_result is not None and hasattr(download_result, "local_path"):
        repo_cache_path = download_result.local_path
        if repo_cache_path.exists():
            # Remove only the specific repository cache
            # The path structure is typically: cache/models--username--repo-name/
            # We want to remove the entire models--username--repo-name directory
            repo_root = repo_cache_path
            # Navigate up to find the repo root (models--username--repo-name)
            while repo_root.parent != repo_root and not repo_root.name.startswith("models--"):
                repo_root = repo_root.parent

            if repo_root.name.startswith("models--"):
                shutil.rmtree(repo_root)
                print(f"✓ Removed specific hub repository cache: {repo_root}")
            else:
                print(f"⚠️ Could not determine repo root from path: {repo_cache_path}")
        else:
            print(f"Hub repository cache path not found: {repo_cache_path}")
    else:
        print("⚠️ No download_result available - cannot determine what to clean up")


def reimport_interpretune_with_capture() -> Tuple[str, str]:
    """Re-import interpretune with stdout and stderr capture to check for expected warnings.

    Returns:
        Tuple of (stdout_output, stderr_output) from the import process
    """
    # Capture stdout and stderr during import to check for the expected warning
    f_stdout = io.StringIO()
    f_stderr = io.StringIO()

    with contextlib.redirect_stdout(f_stdout), contextlib.redirect_stderr(f_stderr):
        # Remove interpretune modules again to force reimport and warning emission
        purge_it_modules_from_sys()

        # Re-import interpretune
        from interpretune import DISPATCHER

    stdout_output = f_stdout.getvalue()
    stderr_output = f_stderr.getvalue()

    return stdout_output, stderr_output, DISPATCHER


def inspect_err_for_composite_op_warning(stderr_output: str) -> None:
    if stderr_output and "Failed to compile operation 'composite_trivial_test_op'" in stderr_output:
        # note that this warning won't be issued if we've already executed this cell since the latest cache will be used
        print(stderr_output)
        print(
            "Note the above \"Failed to compile operation 'composite_trivial_test_op'\" error on re-import of "
            "interpretune after our cleanup.\n"
            "This is expected: we have removed our hub op definitions (trivial_test_op), but not our local op "
            "definitions (trivial_local_test_op, composite_trivial_test_op).\n"
            "As a result, the locally defined composite operation 'composite_trivial_test_op' could not be "
            "constructed since it depended on the now-missing hub op.\nAll other available operations (local and "
            "built-in) should still be present as we will see."
        )
    elif stderr_output:
        print("Unexpected stderr output during import:")
        print(stderr_output)


def generate_test_batches(num_batches: int = 2) -> Generator[Tuple[str, object, object], None, None]:
    """Generator that yields test analysis_batch objects with random orig_labels.

    Args:
        num_batches: Number of test batches to generate

    Yields:
        Tuple of (batch_name, separate_op_input_batch, pipeline_op_input_batch)
    """
    import torch
    from interpretune.analysis.ops.base import AnalysisBatch

    for i in range(num_batches):
        separate_op_input_batch, pipeline_op_input_batch = AnalysisBatch(), AnalysisBatch()
        orig_labels = torch.randint(0, 5, (4,))
        for batch in (separate_op_input_batch, pipeline_op_input_batch):
            batch.update(orig_labels=orig_labels.clone())
        yield f"Batch {i + 1} (random orig_labels)", separate_op_input_batch, pipeline_op_input_batch


def maybe_print_output(output, verbose: bool = False) -> None:
    """Log operation output if verbose logging is enabled.

    Args:
        output: The output to potentially print
        verbose: Whether to print the output
    """
    if verbose:
        print(output)


def compare_operation_outputs(individual_outputs: list, composite_outputs: list) -> bool:
    """Compare outputs from individual component operations and composite operations.

    Args:
        individual_outputs: List of outputs from individual component operations
        composite_outputs: List of outputs from composite operations

    Returns:
        True if all outputs match, False otherwise
    """
    print("\n🔍 Validating that composite and individual component op outputs are identical...")
    all_match = True

    for idx, (sep_batch, composite_batch) in enumerate(zip(individual_outputs, composite_outputs)):
        if sep_batch == composite_batch:
            print(f"  ✓ Batch {idx + 1}: Outputs match.")
        else:
            print(f"  ❌ Batch {idx + 1}: Outputs do NOT match!")
            print(f"    Individual op output: {sep_batch}")
            print(f"    Composite op output: {composite_batch}")
            all_match = False

    if all_match:
        print("\n🎉 All batches match: individual and composite operation outputs are identical!")
    else:
        print("\n⚠️ Some batches did not match: please check the operation implementations.")

    return all_match


def demo_lazy_op_instantiation(it_module, hub_ops: dict, local_ops: dict) -> None:
    """Test lazy operation instantiation for different operation types.

    Args:
        it_module: The interpretune module (imported as 'it')
        hub_ops: Dictionary of hub operations
        local_ops: Dictionary of local operations
    """
    from interpretune.analysis.ops.base import OpWrapper

    print("\n🔧 Testing operation instantiation:")
    print(f"labels_to_ids op reference type: {type(it_module.labels_to_ids)}")

    print(f"get_answer_indices op reference type: {type(it_module.get_answer_indices)}")
    print(f"trivial_test_op op reference type: {type(it_module.trivial_test_op)}")

    print(
        f"Get non-direct access attribute of labels_to_ids (description of the underlying AnalysisOp): "
        f"{it_module.labels_to_ids.description}"
    )
    print(f"Type of labels_to_ids now: {type(it_module.labels_to_ids)}")
    print(
        f"Type of get_answer_indices is still: {type(it_module.get_answer_indices)} and its instantiated status is "
        f"{it_module.get_answer_indices._is_instantiated}"
    )
    print(
        f"Non-direct access attribute of get_answer_indices (name of the underlying AnalysisOp): "
        f"{it_module.get_answer_indices.name}"
    )
    print(f"Type of get_answer_indices is now: {type(it_module.get_answer_indices)}")
    print(
        f"Type of trivial_test_op is: {type(it_module.trivial_test_op)} and its instantiated status is "
        f"{it_module.trivial_test_op._is_instantiated}"
    )

    try:
        print(
            f"Non-direct access attribute of trivial_test_op (name of the underlying AnalysisOp): "
            f"{it_module.trivial_test_op.name}"
        )
        print(
            f"Type of trivial_test_op is now: {type(it_module.trivial_test_op)} as it has been successfully "
            f"instantiated"
        )
    except Exception as e:
        print(f"❌ Error instantiating trivial_test_op: {e}")
        if isinstance(it_module.trivial_test_op, OpWrapper):
            print(
                f"Type of trivial_test_op is still: {type(it_module.trivial_test_op)} and its instantiated status is "
                f"{it_module.trivial_test_op._is_instantiated} likely because of a dynamic import failure."
            )

    try:
        print(
            f"Non-direct access attribute of trivial_local_test_op (name of the underlying AnalysisOp): "
            f"{it_module.trivial_local_test_op.name}"
        )
        print(
            f"Type of trivial_local_test_op is now: {type(it_module.trivial_local_test_op)} as it has been "
            f"successfully instantiated"
        )
    except Exception as e:
        print(f"❌ Error instantiating trivial_local_test_op: {e}")
        if isinstance(it_module.trivial_local_test_op, OpWrapper):
            print(
                f"Type of trivial_local_test_op is still: {type(it_module.trivial_local_test_op)} and its "
                f"instantiated status is {it_module.trivial_local_test_op._is_instantiated} likely because of a "
                f"dynamic import failure."
            )

    # Test accessing operations by type
    if hub_ops:
        hub_op_name = next(iter(hub_ops.keys()))
        print(f"{hub_op_name} op reference type: {type(getattr(it_module, hub_op_name, 'Not found'))}")
    if local_ops:
        local_op_name = next(iter(local_ops.keys()))
        print(f"{local_op_name} op reference type: {type(getattr(it_module, local_op_name, 'Not found'))}")
