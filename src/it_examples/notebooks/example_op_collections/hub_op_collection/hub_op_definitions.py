"""Trivial example of a hub-based analysis operation for interpretune framework testing."""
from __future__ import annotations  # see PEP 749, no longer needed when 3.13 reaches EOL
from typing import Optional

import torch
from transformers import BatchEncoding

from interpretune.protocol import BaseAnalysisBatchProtocol, DefaultAnalysisBatchProtocol

class SomeDifferentBatchDef(BaseAnalysisBatchProtocol):
    """Example of batch definition for a trivial demo op."""
    # Define any additional attributes or methods specific to this batch definition
    preds: Optional[torch.Tensor]
    pred_sum: Optional[torch.Tensor]

def trivial_test_op_impl(module: Optional[torch.nn.Module], analysis_batch: DefaultAnalysisBatchProtocol,
                         batch: Optional[BatchEncoding], batch_idx: Optional[int]) -> SomeDifferentBatchDef:
    """Implementation that takes preds as input and returns the sum of the preds (pred_sum)."""
    if hasattr(analysis_batch, 'preds') and analysis_batch.preds is not None:
        # Count zeros in the preds tensor
        pred_sum = torch.sum(analysis_batch.preds)
        analysis_batch.update(pred_sum=pred_sum)
        print(f"Hub op: Calculated pred_sum: {pred_sum}")
    else:
        # Set preds to 0 for this batch if not available
        print("Hub op: No preds found, setting pred_sum to 0")
        analysis_batch.update(pred_sum=torch.tensor(0))

    return analysis_batch
