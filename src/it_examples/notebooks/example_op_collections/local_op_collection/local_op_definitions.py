"""Trivial example of a local analysis operation for interpretune framework testing."""
from interpretune.protocol import DefaultAnalysisBatchProtocol

def trivial_local_test_op_impl(analysis_batch: DefaultAnalysisBatchProtocol) -> DefaultAnalysisBatchProtocol:
    """Implementation that takes orig_labels as input and produces preds as output."""
    if hasattr(analysis_batch, 'orig_labels') and analysis_batch.orig_labels is not None:
        # Simple operation: add 1 to orig_labels to create preds
        preds = analysis_batch.orig_labels + 1
        analysis_batch.update(preds=preds)
        print(f"Local op: Converted orig_labels {analysis_batch.orig_labels} to preds {preds}")
    else:
        print("Local op: No orig_labels found in analysis_batch")

    return analysis_batch
