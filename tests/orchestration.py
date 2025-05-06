# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Initially based on https://bit.ly/3oQ8Vqf
from pathlib import Path
from unittest import mock
from typing import Dict

from interpretune.protocol import Adapter
from interpretune.session import ITSessionConfig, ITSession
from interpretune.runners import SessionRunner, AnalysisRunner, core_analysis_loop
from interpretune.config.runner import SessionRunnerCfg, AnalysisRunnerCfg
from interpretune.config.analysis import AnalysisStoreProtocol
from interpretune.config import init_analysis_cfgs
from interpretune.base import _call_itmodule_hook
from tests import Trainer, ModelCheckpoint
from tests.configuration import config_modules, cfg_op_env
from tests.base_defaults import AnalysisBaseCfg, BaseCfg, OpTestConfig
from tests.utils import kwargs_from_cfg_obj
import torch
from interpretune.analysis.ops.base import AnalysisBatch


########################################################################################################################
# Core Train/Test Orchestration
########################################################################################################################

########################################################################################################################
# NOTE: [Parity Testing Approach]
# - We use a single set of results but separate tests for adapter parity tests since we want to minimize adapter
#   dependencies for Interpretune and we want to mark at the test-level for greater clarity and flexibility (we want to
#   signal clearly when either diverges from the expected benchmark so aren't testing relative values only)
# - The configuration space for parity tests is sampled rather than exhaustively testing all adapter configuration
#   combinations due to resource constraints
# - Note that while we could access test_alias using the request fixture (`request.node.callspec.id`), this approach
#   using dataclass encapsulation allows us to flexibly define test ids, configurations, marks and expected outputs
#   together
# - We always check for basic exact match on device type, precision and dataset state
# - Our result mapping function uses these shared results for all supported parity test suffixes (e.g. '_l')
# - Set `state_log_mode=True` by setting the environmental variable `IT_GLOBAL_STATE_LOG_MODE` to `1` during development
#   to generate/dump state logs for tests rather than testing the relevant assertions (this can be manually overridden
#   on a test-by-test basis as well)
########################################################################################################################

def parity_test(test_cfg, test_alias, expected_results, tmp_path, state_log_mode: bool = False):
    it_session = config_modules(test_cfg, test_alias, expected_results, tmp_path, state_log_mode=state_log_mode)
    if Adapter.lightning in test_cfg.adapter_ctx:
        _ = run_lightning(it_session, test_cfg, tmp_path)
    else:
        run_it(it_session, test_cfg)

def init_it_runner(it_session: ITSession, test_cfg: BaseCfg, *args, **kwargs):
    # we allow passing unused args for parity testing, e.g. the IT module configured tmp_path is needed by some trainers
    test_cfg_overrides = {k: v for k,v in test_cfg.__dict__.items() if k in SessionRunnerCfg.__dict__.keys()}
    runner_config = SessionRunnerCfg(it_session=it_session, **test_cfg_overrides)
    runner = SessionRunner(run_cfg=runner_config)
    return runner

def run_it(it_session: ITSession, test_cfg: BaseCfg, init_only: bool = False) -> \
    SessionRunner | AnalysisStoreProtocol | Dict[str, AnalysisStoreProtocol] | None:
    # Check if test_cfg is an AnalysisBaseCfg and use the appropriate runner initialization
    if isinstance(test_cfg, AnalysisBaseCfg):
        runner = init_analysis_runner(it_session, test_cfg)
    else:
        runner = init_it_runner(it_session, test_cfg)

    if init_only:
        # If only initializing the runner, return it without executing any phase
        return runner

    # Execute the appropriate phase
    if test_cfg.phase == "test":
        runner.test()
    elif test_cfg.phase == "train":
        runner.train()
    elif test_cfg.phase == "analysis":
        # If the phase is analysis, multiple analysis configurations can be passed so we orchestrate the sequence of
        # analysis phase steps using the AnalysisRunner
        return runner.run_analysis()
    else:
        raise ValueError("Unsupported phase type, phase must be 'test', 'train' or 'analysis'")


################################################################################
# Analysis Orchestration
################################################################################

def init_analysis_runner(it_session: ITSession, test_cfg: BaseCfg, *args, **kwargs):
    """Initialize an AnalysisRunner with the appropriate configuration.

    Args:
        it_session: The ITSession instance to use for analysis
        test_cfg: Test configuration dataclass with optional analysis-specific properties
        *args, **kwargs: Additional arguments to pass to the AnalysisRunner

    Returns:
        An initialized AnalysisRunner instance
    """
    run_cfg_kwargs = kwargs_from_cfg_obj(AnalysisRunnerCfg, test_cfg, base_kwargs={"it_session": it_session})
    analysis_cfg = AnalysisRunnerCfg(**run_cfg_kwargs)
    runner = AnalysisRunner(run_cfg=analysis_cfg)
    return runner

def run_analysis_operation(it_session, use_run_cfg=True, test_cfg: BaseCfg | None = None, **kwargs):
    """Run analysis operations either through a runner with run_config or directly via core_analysis_loop.

    Args:
        it_session: The IT session instance containing module and datamodule
        analysis_cfgs: Analysis configuration(s) to use
        use_run_cfg: Whether to use AnalysisRunner with run_config (True) or core_analysis_loop (False)
        **kwargs: Additional configuration parameters

    Returns:
        Analysis results
    """

    if use_run_cfg:
        runner = init_analysis_runner(it_session, test_cfg)
        return runner.run_analysis()
    else:
        # Direct core_analysis_loop approach with a single analysis config
        if not test_cfg.analysis_cfgs or len(test_cfg.analysis_cfgs) == 0:
            raise ValueError("For direct core_analysis_loop, at least one analysis_cfg must be provided")

        target_analysis_cfg = test_cfg.analysis_cfgs[0]

        run_cfg_kwargs = kwargs_from_cfg_obj(init_analysis_cfgs, test_cfg, base_kwargs={"it_session": it_session})
        kwargs.update(run_cfg_kwargs)
        # Use kwargs for core_analysis_loop with the single analysis config
        analysis_cfg_kwargs = {
            "analysis_cfg": target_analysis_cfg,
            "module": it_session.module,
            "datamodule": it_session.datamodule,
            "step_fn": target_analysis_cfg.step_fn,
            **kwargs
        }

        return core_analysis_loop(**analysis_cfg_kwargs)

def run_op_with_config(request, op_cfg: OpTestConfig):
    """Run operation and return results.

    Args:
        request: pytest request fixture
        op_cfg: Configuration for the operation test

    Returns:
        Tuple of (it_session, batches, result_batches, pre_serialization_shapes)
    """

    # Set up the test environment
    it_session, batches, analysis_batches = cfg_op_env(
        request,
        op_cfg.session_fixt,
        op_cfg.target_op,
        batches=op_cfg.batch_size,
        generate_required_only=op_cfg.generate_required_only,
        override_req_cols=op_cfg.override_req_cols,
        deepcopy_session_fixt=op_cfg.deepcopy_session_fixt,
    )

    # update op_cfg.resolved_op with resolved operation, this allows us to test both OpWrapper and our resolved op
    op_cfg.resolved_op = it_session.module.analysis_cfg.op
    # Convert to lists for consistent handling
    batches = [batches] if op_cfg.batch_size == 1 else batches
    analysis_batches = [analysis_batches] if op_cfg.batch_size == 1 else analysis_batches

    # Results container
    result_batches = []
    pre_serialization_shapes = {}

    _call_itmodule_hook(it_session.module, hook_name="on_analysis_start", hook_msg="Running analysis start hooks")

    # Execute the operation for each batch
    for i, (batch, analysis_batch) in enumerate(zip(batches, analysis_batches)):
        # Run the operation
        result = op_cfg.resolved_op(it_session.module, analysis_batch, batch, i)

        # Verify result is a proper AnalysisBatch
        assert isinstance(result, AnalysisBatch)

        # Record pre-serialization shapes for all important fields
        if i == 0:  # Only need to record shapes once
            # Get fields from output_schema
            for column_name, col_cfg in it_session.module.analysis_cfg.op.output_schema.items():
                # Skip intermediate-only fields
                if col_cfg.intermediate_only:
                    continue

                # Skip fields not present in result
                if not hasattr(result, column_name):
                    if col_cfg.required:
                        raise ValueError(f"Required field {column_name} not found in result")
                    continue

                value = getattr(result, column_name)
                if isinstance(value, torch.Tensor):
                    pre_serialization_shapes[column_name] = value.shape
                elif isinstance(value, dict) and value:
                    # For dictionary columns - store shape for each key with tensor values
                    if all(isinstance(v, torch.Tensor) for v in value.values()):
                        # Simple dict of tensors
                        pre_serialization_shapes[column_name] = {k: v.shape for k, v in value.items()}
                    elif any(isinstance(v, dict) for v in value.values()):
                        # Nested dicts (like per_latent structure)
                        nested_shapes = {}
                        for k, v in value.items():
                            if isinstance(v, dict):
                                nested_shapes[k] = {inner_k: inner_v.shape for inner_k, inner_v in v.items()
                                                  if isinstance(inner_v, torch.Tensor)}
                            elif isinstance(v, torch.Tensor):
                                nested_shapes[k] = v.shape
                        pre_serialization_shapes[column_name] = nested_shapes

        result_batches.append(result)

    _call_itmodule_hook(it_session.module, hook_name="on_analysis_end", hook_msg="Running analysis start hooks")

    return it_session, batches, result_batches, pre_serialization_shapes

################################################################################
# Lightning Adapter Train/Test Orchestration
################################################################################

def init_lightning_trainer(it_session: ITSession, test_cfg: tuple, tmp_path: Path) -> Trainer:
    accelerator = "cpu" if test_cfg.device_type == "cpu" else "gpu"
    callbacks = instantiate_callbacks(getattr(test_cfg, 'callback_cfgs', {}))
    trainer_steps = {"limit_train_batches": test_cfg.limit_train_batches,
                     "limit_val_batches": test_cfg.limit_val_batches, "limit_test_batches": test_cfg.limit_test_batches,
                     "limit_predict_batches": 1, "max_steps": test_cfg.max_steps}
    trainer = Trainer(default_root_dir=tmp_path, devices=1, deterministic=True, accelerator=accelerator,
                      max_epochs=test_cfg.max_epochs, precision=lightning_prec_alias(test_cfg.precision),
                      num_sanity_val_steps=0, callbacks=callbacks, **trainer_steps)
    return trainer

def run_lightning(it_session: ITSessionConfig, test_cfg: tuple, tmp_path: Path) -> Trainer:
    trainer = init_lightning_trainer(it_session, test_cfg, tmp_path)
    match test_cfg.phase:
        case "train":
            lightning_func = trainer.fit
        case "test":
            lightning_func = trainer.test
        case "predict":
            lightning_func = trainer.predict
        case _:
            raise ValueError("Unsupported phase type, phase must be 'train', 'test' or 'predict'")
    if test_cfg.save_checkpoints:
        lightning_func(**it_session)
    else:
        with mock.patch.object(ModelCheckpoint, "_save_checkpoint"):  # by default, don't save checkpoints for lightning
            lightning_func(**it_session)
    return trainer

def lightning_prec_alias(precision: str):
    # TODO: update this and get_model_input_dtype() to use a shared set of supported alias mappings once more
    # types are tested
    return "bf16-true" if precision == "bf16" else "32-true"

def instantiate_callbacks(callbacks_cfg: dict):
    return [callback_cls(**kwargs) for callback_cls, kwargs in callbacks_cfg.items()]
