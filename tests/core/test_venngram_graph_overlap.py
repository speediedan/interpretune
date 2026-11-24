"""#966: a graph subset compared with another prompt's attribution graph, to measure how much of it transferred."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("venngram.graphs")
pytest.importorskip("circuit_tracer")

from circuit_tracer.attribution.targets import LogitTarget  # noqa: E402
from circuit_tracer.graph import Graph  # noqa: E402
from circuit_tracer.utils.tl_nnsight_mapping import UnifiedConfig  # noqa: E402

from interpretune.analysis.venngram import export_graph_subset, graph_subset_overlap  # noqa: E402
from interpretune.utils import MisconfigurationException  # noqa: E402

MODEL = {"id": "m", "revision": "r1"}
SPACE = {"id": "dict/transcoders", "revision": "p", "kind": "transcoder"}
TOKENIZER = {"id": "m-tokenizer", "revision": "t1"}
#: The source prompt, and a destination with one token prepended, where from-end nodes land one position later.
SOURCE = [5, 22, 5]
DESTINATION = [1] + SOURCE


def _graph(tokens, active, activations):
    """A circuit-tracer graph over ``tokens`` on a two-layer model selecting active rows 0, 1 and 3 (row 2 is
    active and unselected, so a match must follow ``selected_features`` rather than row order)."""
    cfg = UnifiedConfig(
        n_layers=2,
        d_model=4,
        d_head=2,
        n_heads=2,
        d_mlp=8,
        d_vocab=32,
        tokenizer_name="fake-tokenizer",
        model_name="fake-model",
        original_architecture="FakeForCausalLM",
    )
    n = 3 + 3 * len(tokens) + 1
    return Graph(
        input_string="",
        input_tokens=torch.tensor(tokens),
        active_features=torch.tensor(active),
        adjacency_matrix=torch.zeros(n, n),
        cfg=cfg,
        selected_features=torch.tensor([0, 1, 3]),
        activation_values=torch.tensor(activations),
        logit_targets=[LogitTarget("x", 7)],
        logit_probabilities=torch.tensor([1.0]),
        vocab_size=32,
    )


def _source():
    return _graph(SOURCE, [[0, 0, 10], [0, 1, 11], [1, 1, 99], [1, 2, 12]], [3.0, 2.0, 1.5])


def _destination():
    """The destination's graph selects the subset's first two features where they land, and another feature where
    the third does."""
    return _graph(DESTINATION, [[0, 1, 10], [0, 2, 11], [0, 0, 55], [1, 3, 77]], [4.0, 1.0, 0.5])


def _subset(**kwargs):
    return export_graph_subset(_source(), name="reasoning", model=MODEL, space=SPACE, **kwargs)


class TestGraphSubsetOverlap:
    def test_the_placements_the_graph_selects_are_matched_with_their_activations_there(self):
        overlap = graph_subset_overlap(_subset(), _destination(), influence=[0.5, 0.25, 0.25])
        assert overlap.placed == (("f0", 0, 1, 10), ("f1", 0, 2, 11), ("f2", 1, 3, 12))
        assert overlap.matched == {("f0", 0, 1, 10): 4.0, ("f1", 0, 2, 11): 1.0}
        assert overlap.node_overlap == pytest.approx(2 / 3)
        assert overlap.recovered_influence == pytest.approx(0.75)
        assert graph_subset_overlap(_subset(), _destination()).recovered_influence is None

    def test_the_source_graph_contains_its_own_subset(self):
        """The positive control: a subset compared with the graph it came from matches every node."""
        overlap = graph_subset_overlap(_subset(), _source(), influence=[1.0, 1.0, 1.0])
        assert overlap.node_overlap == 1.0 and overlap.recovered_influence == 1.0

    def test_a_document_is_read_as_its_subset(self):
        document = _subset().to_mapping()
        assert (
            graph_subset_overlap(document, _destination()).matched
            == graph_subset_overlap(_subset(), _destination()).matched
        )

    def test_explicit_token_ids_override_the_graph_prompt(self):
        # resolved in the source prompt, the nodes land where the destination graph selects none of their features
        overlap = graph_subset_overlap(_subset(), _destination(), token_ids=SOURCE)
        assert overlap.matched == {} and overlap.node_overlap == 0.0

    def test_token_matched_nodes_follow_their_tokens(self):
        overlap = graph_subset_overlap(
            _subset(anchor="token_match", tokenizer=TOKENIZER), _destination(), tokenizer=TOKENIZER
        )
        assert [(layer, position) for _, layer, position, _ in overlap.placed] == [(0, 1), (0, 2), (1, 3)]

    @pytest.mark.parametrize(
        "subset_kwargs, kwargs, match",
        [
            pytest.param({}, {"influence": [1.0]}, "one per selected feature", id="short-influence"),
            pytest.param(
                {"anchor": "token_match", "tokenizer": TOKENIZER},
                {"tokenizer": TOKENIZER, "token_ids": [7, 8]},
                "token span",
                id="unresolved-anchor",
            ),
            pytest.param(
                {"anchor": "token_match", "tokenizer": TOKENIZER}, {}, "tokenizer", id="token-match-without-tokenizer"
            ),
        ],
    )
    def test_what_cannot_be_compared_is_refused(self, subset_kwargs, kwargs, match):
        with pytest.raises(MisconfigurationException, match=match):
            graph_subset_overlap(_subset(**subset_kwargs), _destination(), **kwargs)

    def test_a_space_with_vector_values_is_refused(self):
        document = _subset(edges=False).to_mapping()
        document["spaces"]["tc"] = {**SPACE, "kind": "block_sparse", "value_shape": [2]}
        for node in document["nodes"]:
            node["activation"] = [node["activation"], 0.0]
        with pytest.raises(MisconfigurationException, match="vector value"):
            graph_subset_overlap(document, _destination())
