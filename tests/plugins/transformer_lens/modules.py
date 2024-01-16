from typing import Dict

from transformer_lens.utilities.devices import get_device_for_block_index

from interpretune.utils.import_utils import _LIGHTNING_AVAILABLE
from interpretune.plugins.transformer_lens import ITLensModule, ITLensLightningModule
from it_examples.experiments.rte_boolq.core import RTEBoolqModuleMixin
from tests.base.modules import BaseTestModule, BaseTestITLightningModule


class TestITLensModule(BaseTestModule, RTEBoolqModuleMixin, ITLensModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO: consider setting output_device as property of the tlens module instead of just in test module
        self.output_device = None

    def setup(self, *args, **kwargs) -> None:
        super().setup(*args, **kwargs)
        # since there may be multiple devices for TransformerLens (once fully supported by Interpretune) we test
        # the type of the output device
        self.output_device = get_device_for_block_index(self.model.cfg.n_layers - 1, self.model.cfg)

    def _get_current_exact(self) -> Dict:
        return {'device_type': self.output_device.type, 'precision': self.tl_cfg.dtype, **self._get_dataset_state()}


if _LIGHTNING_AVAILABLE:
    class TestITLensLightningModule(BaseTestITLightningModule, ITLensLightningModule):
        ...
else:
    TestITLensLightningModule = object
