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

from interpretune.adapters.registration import CompositionRegistry
from interpretune.base.datamodules import ITDataModule
from interpretune.base.modules import BaseITModule
from interpretune.base.config.shared import Adapter
from interpretune.adapters.core import ITModule
from interpretune.adapters.lightning import LightningAdapter, LightningModule


class TestClassRegistration:


    def test_canonicalize_composition(self):
        TEST_ADAPTER_REGISTRY = CompositionRegistry()
        adapter_tuple_cc_result = TEST_ADAPTER_REGISTRY.canonicalize_composition(("transformer_lens", Adapter.core,))
        str_list_cc_result = TEST_ADAPTER_REGISTRY.canonicalize_composition(["core", "transformer_lens", Adapter.core])
        assert adapter_tuple_cc_result == str_list_cc_result == (Adapter.core, Adapter.transformer_lens)

    def test_register_adapter(self):
        TEST_ADAPTER_REGISTRY = CompositionRegistry()
        TEST_ADAPTER_REGISTRY.register(Adapter.core, component_key = "module",
                adapter_combination=(Adapter.core,),
                composition_classes=(ITModule,),
                description="core adapter to be used with native PyTorch",
            )
        with pytest.raises(KeyError, match="was not found in the registry. Available valid compositions"):
            TEST_ADAPTER_REGISTRY.get((Adapter.lightning,))
        TEST_ADAPTER_REGISTRY.register(Adapter.core, component_key = "datamodule",
                adapter_combination=(Adapter.core,),
                composition_classes=(ITDataModule,),
                description="core adapter to be used with native PyTorch",
            )
        TEST_ADAPTER_REGISTRY.register(Adapter.core, component_key = "module",
                adapter_combination=(Adapter.lightning,),
                composition_classes=(LightningAdapter, BaseITModule, LightningModule),
                description="lighting adapter",
            )
        available_set = TEST_ADAPTER_REGISTRY.available_compositions(adapter_filter="core")
        assert available_set == {('module', Adapter.core), ('datamodule', Adapter.core)}
        with pytest.warns(UserWarning, match="The following adapter names"):
            _ = TEST_ADAPTER_REGISTRY.available_compositions(adapter_filter=[Adapter.core, "oops"])
        TEST_ADAPTER_REGISTRY.remove(("datamodule", Adapter.core))
        TEST_ADAPTER_REGISTRY.remove(("module", Adapter.lightning))
        assert TEST_ADAPTER_REGISTRY.available_compositions() == {('module', Adapter.core)}
        assert str(TEST_ADAPTER_REGISTRY)
