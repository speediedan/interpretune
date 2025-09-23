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
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch, Mock
import operator
import os
import sys
from typing import Any
import json
import tempfile

import pytest
import torch

from tests.runif import RunIf
from tests.utils import ablate_cls_attrs
from tests.warns import CORE_CTX_WARNS, unexpected_warns, unmatched_warns
from interpretune.session import ITSession
from interpretune.utils import (
    resolve_funcs,
    _get_rank,
    rank_zero_only,
    rank_zero_deprecation,
    instantiate_class,
    _resolve_dtype,
    package_available,
    move_data_to_device,
    to_device,
    module_available,
    compare_version,
)
from interpretune.config import ITExtension, SessionRunnerCfg
from interpretune.extensions import MemProfilerHooks, DefaultMemHooks
from interpretune.utils.exceptions import (
    handle_exception_with_debug_dump,
    _introspect_variable,
    _json_serializer,
    MisconfigurationException,
    IT_ANALYSIS_DUMP_DIR_NAME,
)


class TestClassUtils:
    @RunIf(min_cuda_gpus=1)
    @pytest.mark.parametrize(
        "w_expected",
        ["Unable to patch `get_cud.*", None],
        ids=["cuda_loading_warn_unpatched", "cuda_loading_patched"],
    )
    def test_get_cuda_loading_patch(self, recwarn, get_it_session_cfg__core_cust, w_expected):
        sess_cfg = deepcopy(get_it_session_cfg__core_cust)
        orig_patch_mgr = None
        if w_expected:
            orig_patch_mgr = sys.modules["interpretune.utils.logging"].__dict__.pop("patch_torch_env_logging_fn")
        it_session = ITSession(sess_cfg)
        if w_expected:
            sys.modules["interpretune.utils.logging"].__dict__["patch_torch_env_logging_fn"] = orig_patch_mgr
        collected_cuda_loading_config = it_session.module._it_state._init_hparams["env_info"]["cuda_module_loading"]
        assert isinstance(collected_cuda_loading_config, str)
        if w_expected:
            assert collected_cuda_loading_config != "not inspected"
            unmatched = unmatched_warns(rec_warns=recwarn.list, expected_warns=w_expected)
            assert not unmatched
        else:
            assert collected_cuda_loading_config == "not inspected"
            unexpected = unexpected_warns(rec_warns=recwarn.list, expected_warns=CORE_CTX_WARNS)
            assert not unexpected, tuple(w.message.args[0] + ":" + w.filename + ":" + str(w.lineno) for w in unexpected)

    def test_rank_zero_utils(self):
        with patch.dict(os.environ, {"RANK": "42"}):
            test_rank = _get_rank()
            assert test_rank == 42
        with patch.object(rank_zero_only, "rank", "13"):

            def rank_zero_default(*args: Any, stacklevel: int = 4, **kwargs: Any) -> None:
                pass

            rank_zero_default = rank_zero_only(rank_zero_default, default="test success")
            assert rank_zero_default() == "test success"
        with patch.object(rank_zero_only, "rank", None):

            @rank_zero_only
            def rank_zero_errtest(*args: Any, stacklevel: int = 4, **kwargs: Any) -> None:
                pass

            with pytest.raises(RuntimeError, match="needs to be set before use"):
                rank_zero_errtest()
        with pytest.warns(DeprecationWarning, match="Test deprecation msg"):
            rank_zero_deprecation("Test deprecation msg.")

    def test_fn_instantiate_class(self):
        short_circuit_path = "ITExtension"
        ext_init = {
            "class_path": short_circuit_path,
            "init_args": {"ext_attr": "test_ext", "ext_cls_fqn": "some.loc", "ext_cfg_fqn": "another.loc"},
        }
        sys.modules["interpretune.utils.import_utils"].instantiate_class.__globals__[short_circuit_path] = ITExtension
        test_ext = instantiate_class(init=ext_init)
        sys.modules["interpretune.utils.import_utils"].instantiate_class.__globals__.pop(short_circuit_path)
        assert test_ext.ext_cls_fqn == "some.loc"
        del ext_init["class_path"]
        with pytest.raises(MisconfigurationException, match="A class_path was not included"):
            test_ext = instantiate_class(init=ext_init)

    def test_fn_resolve_funcs(self):
        memory_hooks_cfg = MemProfilerHooks(
            pre_forward_hooks=DefaultMemHooks.pre_forward.value, post_forward_hooks=[], reset_state_hooks=[]
        )
        resolved_single_hook = resolve_funcs(cfg_obj=memory_hooks_cfg, func_type="pre_forward_hooks")
        assert callable(resolved_single_hook[0])
        memory_hooks_cfg = MemProfilerHooks(
            pre_forward_hooks="interpretune.utils.warnings.unexpected_state_msg_suffix",
            post_forward_hooks=[],
            reset_state_hooks=[],
        )
        with pytest.raises(MisconfigurationException, match="is not callable"):
            resolved_single_hook = resolve_funcs(cfg_obj=memory_hooks_cfg, func_type="pre_forward_hooks")
        memory_hooks_cfg = MemProfilerHooks(
            pre_forward_hooks="notfound.analysis.memprofiler._hook_npp_pre_forward",
            post_forward_hooks=[],
            reset_state_hooks=[],
        )
        with pytest.raises(MisconfigurationException, match="Unable to import and resolve specified function"):
            resolved_single_hook = resolve_funcs(cfg_obj=memory_hooks_cfg, func_type="pre_forward_hooks")

    def test_fn_resolve_dtype(self):
        resolved_dtype = _resolve_dtype(dtype="torch.float32")
        assert isinstance(resolved_dtype, torch.dtype)

    def test_package_module_available(self):
        assert not package_available("notgoingtofind.this.package")
        assert not module_available("missingtoplevelpackage.analysis.memprofiler")
        assert not module_available("interpretune.extensions.missingmodule")

    def test_compare_version(self):
        assert not compare_version("torchnotfound", operator.ge, "2.2.0", use_base_version=True)
        with ablate_cls_attrs(torch, "__version__"):
            assert compare_version("torch", operator.ge, "2.0.0", use_base_version=True)
        with patch.object(torch, "__version__", lambda x: x + 1):  # allow TypeError for Sphinx mocks
            assert compare_version("torch", operator.ge, "2.0.0", use_base_version=True)

    @RunIf(min_cuda_gpus=1)
    def test_to_device(self):
        mod = torch.nn.Linear(4, 8)
        mod = to_device("cuda", mod)
        assert mod.bias.device.type == "cuda"

    @RunIf(min_cuda_gpus=1)
    def test_move_data_to_device(self):
        batch = torch.ones([2, 3])
        batch = move_data_to_device(batch, "cuda")
        assert batch.device.type == "cuda"
        orig_to = torch._tensor.Tensor.to

        def degen_to(aten, *args, **kwargs):
            aten = orig_to(aten, *args, **kwargs)
            assert aten.device.type == "cpu"  # forget to return self

        with patch("torch._tensor.Tensor.to", degen_to):
            batch = move_data_to_device(batch, "cpu")
        assert batch.device.type == "cuda"  # still on cuda because of improperly implemented `to`

    def test_basic_trainer_warns(self, get_it_session__core_cust__setup):
        fixture = get_it_session__core_cust__setup
        it_session, test_cfg = fixture.it_session, fixture.test_cfg()
        test_cfg_overrides = {k: v for k, v in test_cfg.__dict__.items() if k in SessionRunnerCfg.__dict__.keys()}
        with pytest.raises(MisconfigurationException, match="If not providing `it_session`"):
            _ = SessionRunnerCfg(module=it_session.module, datamodule=None, **test_cfg_overrides)
        assert test_cfg
        with pytest.warns(UserWarning, match="should only be specified if not providing `it_session`"):
            trainer_config = SessionRunnerCfg(module=it_session.module, it_session=it_session, **test_cfg_overrides)
        assert trainer_config

    def test_handle_exception_with_debug_dump(self, tmp_path):
        """Test that handle_exception_with_debug_dump creates a debug dump file with correct content."""
        # Setup
        test_exception = ValueError("Test exception")
        test_context = {"test_key": "test_value", "nested": {"a": 1, "b": 2}}
        operation = "test_operation"

        # Create a mock datetime that will be returned by datetime.now()
        mock_datetime = Mock()
        mock_datetime.strftime.return_value = "20230101_120000"

        # Execute with a temp directory to avoid cluttering the project
        with (
            patch("json.dump") as mock_dump,
            patch("os.makedirs") as mock_makedirs,
            patch("pathlib.Path.open"),
            patch("datetime.datetime", autospec=True) as mock_dt_class,
            pytest.raises(ValueError) as exc_info,
        ):
            # Set the datetime.now to return our mock
            mock_dt_class.now.return_value = mock_datetime
            handle_exception_with_debug_dump(test_exception, test_context, operation, debug_dir_override=tmp_path)

        # Verify
        assert exc_info.value == test_exception  # The original exception is re-raised
        mock_makedirs.assert_called_once_with(tmp_path, exist_ok=True)

        # Check that the dump contains the expected data
        expected_debug_info = {
            "error": "Test exception",
            "traceback": mock_dump.call_args[0][0]["traceback"],  # Just compare structure, not actual traceback
            "test_key": "test_value",
            "nested": {"a": 1, "b": 2},
        }

        # Verify that json.dump was called with the expected data structure
        actual_debug_info = mock_dump.call_args[0][0]
        assert "error" in actual_debug_info
        assert "traceback" in actual_debug_info
        assert actual_debug_info["test_key"] == expected_debug_info["test_key"]
        assert actual_debug_info["nested"] == expected_debug_info["nested"]

    def test_handle_exception_with_debug_dump_sequence(self, tmp_path):
        """Test handle_exception_with_debug_dump with a sequence of values."""
        test_exception = RuntimeError("Test sequence exception")
        sequence_context = ["string_value", 123, {"dict_key": "dict_value"}, [1, 2, 3]]

        with (
            patch("json.dump") as mock_dump,
            patch("os.makedirs"),
            patch("pathlib.Path.open"),
            patch("inspect.currentframe") as mock_frame,
            pytest.raises(RuntimeError),
        ):
            # Mock the inspect frame to simulate caller context
            mock_frame.return_value = None  # Simulate no frame info available
            handle_exception_with_debug_dump(test_exception, sequence_context, debug_dir_override=tmp_path)

        # Check that each item in the sequence was introspected
        debug_info = mock_dump.call_args[0][0]
        assert "var_0" in debug_info
        assert "var_1" in debug_info
        assert "var_2" in debug_info
        assert "var_3" in debug_info

        assert debug_info["var_0"]["type"] == "str"
        assert debug_info["var_0"]["value"] == "string_value"

        assert debug_info["var_1"]["type"] == "int"
        assert debug_info["var_1"]["value"] == 123

    def test_handle_exception_with_debug_dump_single_item(self, tmp_path):
        """Test handle_exception_with_debug_dump with a single item context."""
        test_exception = RuntimeError("Test single item context exception")
        single_context = "single_value"

        with (
            patch("json.dump") as mock_dump,
            patch("os.makedirs"),
            patch("pathlib.Path.open"),
            patch("inspect.currentframe") as mock_frame,
            pytest.raises(RuntimeError),
        ):
            # Mock the inspect frame to simulate caller context
            mock_frame.return_value = None  # Simulate no frame info available
            handle_exception_with_debug_dump(test_exception, single_context, debug_dir_override=tmp_path)

        # Check that the single item was introspected
        debug_info = mock_dump.call_args[0][0]
        assert "context" in debug_info
        assert debug_info["context"]["type"] == "str"
        assert debug_info["context"]["value"] == "single_value"

    def test_handle_exception_with_debug_dump_with_frame(self, tmp_path):
        """Test handle_exception_with_debug_dump with frame information."""
        test_exception = Exception("Test frame exception")

        with (
            patch("json.dump") as mock_dump,
            patch("os.makedirs"),
            patch("pathlib.Path.open"),
            patch("inspect.currentframe") as mock_frame,
            patch("inspect.getframeinfo") as mock_frameinfo,
            pytest.raises(Exception),
        ):
            # Create mock frame objects to simulate call stack
            frame_mock = Mock()
            back_frame_mock = Mock()
            frame_mock.f_back = back_frame_mock
            mock_frame.return_value = frame_mock

            # Mock the frame info to provide code context
            frameinfo_mock = Mock()
            frameinfo_mock.code_context = [
                "handle_exception_with_debug_dump(e, context_data=(var1, var2), debug_dir_override=tmp_path)"
            ]
            mock_frameinfo.return_value = frameinfo_mock

            # Test variables to be introspected
            var1 = "test_var1"
            var2 = {"key": "value"}

            handle_exception_with_debug_dump(test_exception, (var1, var2), debug_dir_override=tmp_path)

        # Check that variable names were extracted from frame info
        debug_info = mock_dump.call_args[0][0]
        assert "var_0_var1" in debug_info or "var_0" in debug_info
        assert "var_1_var2" in debug_info or "var_1" in debug_info

    def test_introspect_variable(self):
        """Test the _introspect_variable function with different types."""
        # Test with None
        result = _introspect_variable(None)
        assert result["type"] == "NoneType"
        assert result["value"] is None

        # Test with simple scalar types
        assert _introspect_variable("test")["value"] == "test"
        assert _introspect_variable(42)["value"] == 42
        assert _introspect_variable(3.14)["value"] == 3.14
        assert _introspect_variable(True)["value"] is True

        # Test with sequence types
        list_result = _introspect_variable([1, 2, 3, 4, 5])
        assert list_result["type"] == "list"
        assert list_result["length"] == 5
        assert list_result["sample"] == [1, 2, 3, 4, 5]

        # Test with large sequence (sample limiting)
        large_list = list(range(20))
        large_list_result = _introspect_variable(large_list)
        assert large_list_result["length"] == 20
        assert len(large_list_result["sample"]) == 10

        # Test with dictionary
        dict_result = _introspect_variable({"a": 1, "b": 2, "c": 3})
        assert dict_result["type"] == "dict"
        assert set(dict_result["keys"]) == {"a", "b", "c"}
        assert dict_result["content"] == {"a": 1, "b": 2, "c": 3}

        # Test with large dictionary
        large_dict = {f"key{i}": i for i in range(20)}
        large_dict_result = _introspect_variable(large_dict)
        assert set(large_dict_result["keys"]) == set(large_dict.keys())
        assert "sample" in large_dict_result
        assert len(large_dict_result["sample"]) == 10

        # Test with object having attributes
        class TestObj:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = 42
                self._private = "hidden"

            def method(self):
                return "test method"

        obj = TestObj()
        obj_result = _introspect_variable(obj)

        assert obj_result["type"] == "TestObj"
        assert obj_result["class"] == "TestObj"
        # Only check that module name exists, not its exact value as it depends on how tests are run
        assert "module" in obj_result
        assert obj_result["module"].endswith("test_utils")

        # Test with object that raises error when accessing attributes
        # We need to patch the dir function to avoid the getattr call
        # directly in _introspect_variable that's causing the test failure
        class ProblemObj:
            @property
            def problematic(self):
                raise ValueError("Can't access this")

        prob_obj = ProblemObj()

        # Patch dir to return a list that includes problematic, so we'll try to get it
        with patch("builtins.dir", return_value=["problematic"]):
            prob_result = _introspect_variable(prob_obj)
            assert "problematic" in prob_result["attributes"]
            assert prob_result["attributes"]["problematic"] == "<error getting attribute>"

        # Test with a type that triggers the fallback branch
        test_set = {1, 2, 3}
        set_result = _introspect_variable(test_set)
        assert set_result["type"] == "set"
        assert "repr" in set_result
        assert set_result["repr"] == repr(test_set)

        # Also test with another type that has no __dict__ but complex representation
        test_complex = complex(3, 4)
        complex_result = _introspect_variable(test_complex)
        assert complex_result["type"] == "complex"
        assert "repr" in complex_result
        assert complex_result["repr"] == repr(test_complex)

    def test_json_serializer(self):
        """Test the _json_serializer function used for JSON encoding."""
        # Test with a serializable object
        assert _json_serializer("test") == "test"

        # Test with something that can be converted to string
        class StringableObj:
            def __str__(self):
                return "string representation"

        obj = StringableObj()
        assert _json_serializer(obj) == "string representation"

        # Test with an object that raises an error during serialization
        class UnserializableObj:
            def __str__(self):
                raise ValueError("Can't stringify this")

        unserializable = UnserializableObj()
        assert _json_serializer(unserializable) == "<non-serializable>"

    def test_integration_handle_exception_with_real_file(self, tmp_path):
        """Integration test that verifies a real debug file is created."""
        debug_path = tmp_path / "debug_test"
        test_exception = ValueError("Integration test exception")
        test_context = {"test_key": "integration_value"}

        # Create a real debug file - use try/except pattern instead of context manager
        try:
            raise test_exception  # First raise the exception
        except ValueError as e:
            with pytest.raises(ValueError):
                # Then handle it in the context of an active exception
                handle_exception_with_debug_dump(e, test_context, "integration_test", debug_dir_override=debug_path)

        # Verify debug directory was created
        assert debug_path.exists()
        assert debug_path.is_dir()

        # Find created debug file
        debug_files = list(debug_path.glob("integration_test_error_*.json"))
        assert len(debug_files) == 1

        # Check file contents
        with open(debug_files[0], "r") as f:
            debug_data = json.load(f)

        assert debug_data["error"] == "Integration test exception"
        assert debug_data["test_key"] == "integration_value"
        assert debug_data["traceback"]

    def test_handle_exception_with_debug_dump_default_dir(self, tmp_path, monkeypatch):
        """Test handle_exception_with_debug_dump using the default directory path logic."""
        import datetime

        test_exception = ValueError("Default dir test")
        test_context = {"test_key": "test_value"}

        # Patch IT_ANALYSIS_DUMP_DIR_NAME to include a datetime stamp for test isolation
        dt_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        patched_dir_name = f"{IT_ANALYSIS_DUMP_DIR_NAME}_{dt_stamp}"
        monkeypatch.setattr("interpretune.utils.exceptions.IT_ANALYSIS_DUMP_DIR_NAME", patched_dir_name)
        debug_dir = Path(tempfile.gettempdir()) / patched_dir_name

        # run under patches (no os.makedirs stub so real dir is created)
        with (
            patch("json.dump") as mock_dump,
            patch("pathlib.Path.open", side_effect=open),
            patch("datetime.datetime") as mock_datetime,
            pytest.raises(ValueError),
        ):
            # setup mock datetime for consistent timestamp
            mock_dt = Mock()
            mock_dt.strftime.return_value = "20250519_142500"
            mock_datetime.now.return_value = mock_dt

            handle_exception_with_debug_dump(test_exception, test_context, "default_dir_test")

        # verify debug_dir was created
        assert debug_dir.exists() and debug_dir.is_dir()

        # glob for the single created JSON file (to enforce test isolation)
        files = list(debug_dir.glob("default_dir_test_error_*.json"))
        assert len(files) == 1, f"expected one debug file, found {files}"

        # inspect the dumped data via mock_dump
        debug_data = mock_dump.call_args[0][0]
        assert debug_data["error"] == "Default dir test"
        assert debug_data["test_key"] == "test_value"

        # clean up debug files and directory
        for f in debug_dir.glob("*.json"):
            f.unlink()
        try:
            debug_dir.rmdir()
        except OSError:
            pass
