import pytest
import tempfile
from pathlib import Path
from datasets import Dataset
from datetime import datetime
from typing import Any

import interpretune as it
from tests.configuration import get_deepcopied_session
from interpretune.config import AnalysisCfg
from interpretune.runners.analysis import dataset_features_and_format, maybe_init_analysis_cfg
from interpretune.analysis.core import get_module_dims


@pytest.fixture
def op_serialization_fixt():
    """Create a test utility for serializing and loading analysis results."""
    def _op_serialization_fixt(
        it_session,
        result_batch,
        batch,
        request=None,
    ):
        """Test serialization and loading of analysis results.

        Args:
            it_session: The interpretune session
            result_batch: The analysis batch result(s) to serialize (single or list)
            batch: The original input batch(es) (single or list)
            request: Optional pytest request object for test identification

        Returns:
            loaded_dataset: The loaded dataset with the serialized batch(es)
        """
        module = it_session.module

        # Generate a timestamp for the dataset directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Use test name if request is provided, otherwise use a generic name
        if request:
            test_name = request.node.name.replace("[", "_").replace("]", "_").replace(" ", "_")
            dataset_name = f"{test_name}_{timestamp}"
        else:
            dataset_name = f"test_dataset_{timestamp}"

        # Determine the save directory
        if hasattr(module.analysis_cfg.output_store, 'save_dir'):
            base_dir = module.analysis_cfg.output_store.save_dir
            save_dir = base_dir / "test_datasets" / dataset_name
        else:
            # Fallback to temporary directory if no save_dir configured
            tmp_dir = tempfile.mkdtemp()
            base_dir = Path(tmp_dir)
            save_dir = base_dir / dataset_name

        # Store the original save_dir and restore it after the test
        original_save_dir = None
        if hasattr(module.analysis_cfg.output_store, 'save_dir'):
            original_save_dir = module.analysis_cfg.output_store.save_dir

        try:
            # Create the directory structure
            save_dir.parent.mkdir(exist_ok=True, parents=True)

            # Set the temporary save_dir
            module.analysis_cfg.output_store.save_dir = save_dir

            # Generate features and format parameters
            features, it_format_kwargs, _ = dataset_features_and_format(module, {})

            # Check if we're dealing with multiple batches
            is_multi_batch = isinstance(result_batch, list)

            # Ensure batch is also a list if result_batch is a list
            if is_multi_batch and not isinstance(batch, list):
                raise ValueError("If result_batch is a list, batch must also be a list")

            # If single batch, convert to list for uniform processing
            if not is_multi_batch:
                result_batch = [result_batch]
                batch = [batch]

            # Create a generator that yields all processed batches
            def multi_batch_generator():
                for i, (res_batch, input_batch) in enumerate(zip(result_batch, batch)):
                    # Process and yield the batch
                    processed_batch = module.analysis_cfg.op.save_batch(
                        res_batch,
                        input_batch,
                        tokenizer=it_session.datamodule.tokenizer,
                        save_prompts=module.analysis_cfg.save_prompts,
                        save_tokens=module.analysis_cfg.save_tokens,
                        decode_kwargs=module.analysis_cfg.decode_kwargs
                    )
                    yield processed_batch

            # Create dataset from the generator
            dataset = Dataset.from_generator(
                generator=multi_batch_generator,
                features=features,
                cache_dir=module.analysis_cfg.output_store.cache_dir,
                split='test',
            ).with_format("interpretune", **it_format_kwargs)

            # Save the dataset
            dataset.save_to_disk(str(save_dir))

            # Load the dataset with our custom formatter
            loaded_dataset = Dataset.load_from_disk(str(save_dir))
            loaded_dataset = loaded_dataset.with_format("interpretune", **it_format_kwargs)

            return loaded_dataset

        finally:
            # Restore original save_dir if needed
            if original_save_dir is not None:
                module.analysis_cfg.output_store.save_dir = original_save_dir

    return _op_serialization_fixt


@pytest.fixture
def initialized_analysis_cfg():
    def _initialized_analysis_cfg(fixture, target_op: Any = it.logit_diffs_attr_ablation):
        it_session = get_deepcopied_session(fixture.it_session)
        # Configure the analysis
        analysis_cfg = AnalysisCfg(target_op=target_op, ignore_manual=True, save_tokens=False,
                                   sae_analysis_targets=fixture.test_cfg().sae_analysis_targets)
        # Initialize analysis config on the module
        maybe_init_analysis_cfg(it_session.module, analysis_cfg)

        batch_size, max_answer_tokens, num_classes, vocab_size, max_seq_len = get_module_dims(it_session.module)
        dim_vars = {
            'batch_size': batch_size,
            'max_answer_tokens': max_answer_tokens,
            'num_classes': num_classes,
            'vocab_size': vocab_size,
            'max_seq_len': max_seq_len,
        }
        return it_session, dim_vars

    return _initialized_analysis_cfg
