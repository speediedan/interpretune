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
import pytest
from unittest.mock import MagicMock

from it_examples.example_module_registry import ModuleExampleRegistry, RegisteredExampleCfg
from interpretune.adapters.registration import Adapter
from tests.base_defaults import default_test_task

class TestClassExampleRegistration:


    def test_register_example_module(self):
        TEST_EXAMPLE_MODULE_REGISTRY = ModuleExampleRegistry()
        TEST_EXAMPLE_MODULE_REGISTRY.register(reg_key="cust.test_example", phase="train", model_src_key="cust",
                                              task_name=default_test_task, adapter_combinations=(Adapter.core,),
                                              registered_example_cfg=MagicMock(spec=RegisteredExampleCfg),
                                              description="Testing example registration by both key and composition")
        TEST_EXAMPLE_MODULE_REGISTRY.register(reg_key="cust.test_example.transformer_lens", phase="test",
                                              model_src_key="cust", task_name=default_test_task,
                                              adapter_combinations=(Adapter.core, Adapter.transformer_lens),
                                              registered_example_cfg=MagicMock(spec=RegisteredExampleCfg),
                                              cfg_dict={"orig": "cfg"}, description="Testing example registration")
        TEST_EXAMPLE_MODULE_REGISTRY.get(("cust", default_test_task, "train", (Adapter.core,)))
        TEST_EXAMPLE_MODULE_REGISTRY.get("cust.test_example.transformer_lens")

        available_set = TEST_EXAMPLE_MODULE_REGISTRY.available_compositions(adapter_filter="core")
        assert available_set == {('cust', 'rte', 'train', (Adapter.core,)), ('cust', 'rte', 'test', (Adapter.core,))}
        assert TEST_EXAMPLE_MODULE_REGISTRY['cust.test_example.transformer_lens']['cfg_dict'] == {"orig": "cfg"}

        with pytest.raises(KeyError, match="was not found in the registry"):
            TEST_EXAMPLE_MODULE_REGISTRY.get(("cust", default_test_task, "train",
                                              (Adapter.lightning, Adapter.sae_lens)))

        with pytest.raises(KeyError, match="was not found in the registry"):
            TEST_EXAMPLE_MODULE_REGISTRY.get("cust.oops_dont_exist.transformer_lens")
