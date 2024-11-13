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


class TestClassExampleRegistration:


    def test_register_example_module(self):
        TEST_EXAMPLE_MODULE_REGISTRY = ModuleExampleRegistry()
        TEST_EXAMPLE_MODULE_REGISTRY.register(phase="train", model_src_key="cust", task_name="rte",
                                 adapter_combinations=(Adapter.core,),
                                 registered_example_cfg=MagicMock(spec=RegisteredExampleCfg),
                                 description="Basic example, Custom Transformer with supported adapter compositions")
        TEST_EXAMPLE_MODULE_REGISTRY.register(phase="test", model_src_key="cust", task_name="rte",
                                 adapter_combinations=(Adapter.core, Adapter.transformer_lens),
                                 registered_example_cfg=MagicMock(spec=RegisteredExampleCfg),
                                 description="Basic example, Custom Transformer with supported adapter compositions")

        with pytest.raises(KeyError, match="not found in the registry. Available valid module example compositions"):
            TEST_EXAMPLE_MODULE_REGISTRY.get(("cust", "rte", "train", (Adapter.lightning, Adapter.sae_lens)))
