"""Run one serialized nnsight session request against this process's own copy of a model, as a server would.

Usage: python remote_session_worker.py <model dir> <request blob> <result path>

The test-only stand-in for an NDIF worker: it loads the model itself, deserializes the request against that model's
persistent objects, runs the session body in one outermost trace scope, and writes back only what the body saved.
"""

import sys

import torch
from nnsight import TransformersModel
from nnsight.schema.request import RequestModel
from nnsight.tracing import tracer as tracing


def main(model_dir: str, blob_path: str, result_path: str) -> None:
    lm = TransformersModel(model_dir, task="text-generation", device="cpu", dispatch=True)
    with open(blob_path, "rb") as f:
        session = RequestModel.deserialize(f.read(), lm._remoteable_persistent_objects())
    saved: dict = {}
    push_result = tracing.push_result

    def capture(frame, variables):
        if getattr(tracing._local, "depth", 0) == 1:  # the outermost scope: what the body saved
            keep = tracing._saves()
            saved.update({k: v for k, v in variables.items() if id(v) in keep})
        push_result(frame, variables)

    tracing.push_result = capture
    tracing.inc()
    try:
        session.execute(session.info.code)
    finally:
        tracing.dec()
        tracing.push_result = push_result
    torch.save(saved, result_path)


if __name__ == "__main__":
    main(*sys.argv[1:4])
