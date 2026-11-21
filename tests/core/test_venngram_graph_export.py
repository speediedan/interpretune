"""#966: an attribution graph exported as a venngram graph subset, and applied again in another prompt."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

pytest.importorskip("venngram.graphs")

from interpretune.analysis.venngram import export_graph_subset, recipe_feature_interventions  # noqa: E402
from interpretune.utils import MisconfigurationException  # noqa: E402
from tests.core.test_venngram_features import TRANSCODER  # noqa: E402

MODEL = {"id": "m", "revision": "r1"}
SPACE = {"id": "dict/transcoders", "revision": "p", "kind": "transcoder"}
TOKENIZER = {"id": "m-tokenizer", "revision": "t1"}
SITE = "transcoder.features"


class _Target(NamedTuple):
    token_str: str
    vocab_idx: int


def _graph():
    """A graph over the prompt ``[11, 22, 33, 22]`` selecting three of four active features, with one error node,
    the four embedding nodes and two logit nodes after them (a vocabulary logit and a custom target)."""
    active = torch.tensor([[0, 0, 5], [1, 1, 7], [1, 3, 8], [2, 3, 9]])
    n = 3 + 1 + 4 + 2
    adjacency = torch.zeros(n, n)
    adjacency[2, 0] = 0.25  # f0 -> f2
    adjacency[1, 0] = -0.5  # f0 -> f1
    adjacency[2, 5] = 0.75  # an embedding into f2: not an exported edge
    return SimpleNamespace(
        input_tokens=torch.tensor([11, 22, 33, 22]),
        active_features=active,
        selected_features=torch.tensor([1, 2, 3]),
        activation_values=torch.tensor([3.0, 2.0, 0.5]),
        adjacency_matrix=adjacency,
        logit_targets=[_Target("x", 44), _Target("logit(a)-logit(b)", 100)],
        vocab_size=100,
    )


def _export(**kwargs):
    return export_graph_subset(_graph(), name="reasoning", model=MODEL, space=SPACE, **kwargs)


class TestExport:
    def test_each_selected_feature_is_a_node_at_its_source_coordinate_with_its_activation(self):
        subset = _export()
        rows = [(n.layer, n.position, n.index, n.values["activation"], n.site, n.space) for n in subset.nodes]
        assert rows == [
            (1, 1, (7,), (3.0,), SITE, "tc"),
            (1, 3, (8,), (2.0,), SITE, "tc"),
            (2, 3, (9,), (0.5,), SITE, "tc"),
        ]

    def test_the_source_prompt_and_attribution_targets_are_its_provenance(self):
        source = _export(tokenizer=TOKENIZER, pruning={"node_threshold": 0.8, "edge_threshold": 0.98}).provenance
        assert source["token_ids"] == [11, 22, 33, 22]
        assert source["tokenizer"] == TOKENIZER
        assert source["pruning"] == {"node_threshold": 0.8, "edge_threshold": 0.98}
        assert source["attribution_targets"] == [
            {"kind": "logit", "token_id": 44, "token": "x", "position": 3},
            {"kind": "custom", "label": "logit(a)-logit(b)", "position": 3},
        ]

    def test_edges_are_the_attributions_between_exported_features(self):
        assert {(e.source, e.target, e.attribution) for e in _export().edges} == {
            ("f0", "f2", 0.25),
            ("f0", "f1", -0.5),
        }
        assert _export(edges=False).edges == ()

    @pytest.mark.parametrize(
        "anchor, expected",
        [
            pytest.param("from_end", [{"kind": "from_end", "offset": 2}] + [{"kind": "from_end", "offset": 0}] * 2),
            pytest.param("absolute", [{"kind": "absolute", "position": p} for p in (1, 3, 3)]),
            # token 22 is the node's own token: its first occurrence at position 1, its second at 3
            pytest.param(
                "token_match",
                [{"kind": "token_match", "token_ids": [22], "occurrence": o} for o in (0, 1, 1)],
            ),
            pytest.param("all", [{"kind": "all"}] * 3),
        ],
    )
    def test_every_node_takes_the_anchor_asked_for(self, anchor, expected):
        subset = _export(anchor=anchor, tokenizer=TOKENIZER)
        assert [n.anchor.to_mapping() for n in subset.nodes] == expected

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            pytest.param({"anchor": "nearest"}, "is none of", id="unknown-anchor"),
            pytest.param({"anchor": "token_match"}, "needs the source `tokenizer`", id="token-match-no-tokenizer"),
        ],
    )
    def test_an_anchor_it_cannot_write_is_refused(self, kwargs, match):
        with pytest.raises(MisconfigurationException, match=match):
            _export(**kwargs)

    def test_a_graph_with_nothing_selected_or_misaligned_activations_is_refused(self):
        empty = _graph()
        empty.selected_features, empty.activation_values = torch.tensor([], dtype=torch.long), torch.tensor([])
        with pytest.raises(MisconfigurationException, match="no selected features"):
            export_graph_subset(empty, name="g", model=MODEL, space=SPACE)
        short = _graph()
        short.activation_values = torch.tensor([1.0])
        with pytest.raises(MisconfigurationException, match="1 activation values for 3 selected features"):
            export_graph_subset(short, name="g", model=MODEL, space=SPACE)

    def test_venngram_refuses_what_its_reader_would(self):
        from venngram.errors import AddressError

        with pytest.raises(AddressError, match="revision"):
            export_graph_subset(_graph(), name="g", model={"id": "m"}, space=SPACE)


class TestRoundTrip:
    def test_an_exported_graph_applies_in_another_prompt_through_its_anchors(self, tmp_path):
        """Written to disk, named by a book and applied with ``replace``: each node lands on its anchored position
        in a longer prompt with its source activation, as circuit-tracer's feature tuples."""
        from venngram.book import AddressBook

        path = _export(tokenizer=TOKENIZER).write(tmp_path / "reasoning.yaml")
        book = {
            "schema_version": 5,
            "name": "transfer",
            "sets": {},
            "graphs": {"reasoning": {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
            "treatments": {
                "transfer": {
                    "model": MODEL,
                    "stages": [
                        {
                            "site": SITE,
                            "ops": [
                                {
                                    "op": "replace",
                                    "target": {"graph": "reasoning"},
                                    "value": {"from": {"graph": "reasoning", "column": "activation"}},
                                }
                            ],
                        }
                    ],
                }
            },
        }
        recipe = AddressBook.from_mapping(book, base_dir=tmp_path).treatments["transfer"]
        tuples = recipe_feature_interventions(
            recipe, support=TRANSCODER, token_ids=[5, 6, 7, 8, 9], tokenizer=TOKENIZER
        )
        # from_end offsets 2, 0 and 0 in a prompt of five tokens: positions 2, 4 and 4
        assert sorted(tuples) == [(1, 2, 7, 3.0), (1, 4, 8, 2.0), (2, 4, 9, 0.5)]

    def test_a_large_graph_is_written_with_its_table_in_a_sidecar(self, tmp_path):
        from venngram.graphs import GraphSubset

        subset = _export()
        path = subset.write(tmp_path / "reasoning.yaml", threshold=2)
        assert (tmp_path / "reasoning.table.safetensors").exists()
        again = GraphSubset.from_file(path)
        assert [n.values["activation"] for n in again.nodes] == [n.values["activation"] for n in subset.nodes]


def test_the_stand_in_graph_reads_only_what_circuit_tracers_graph_declares():
    """The export reads a graph by these attributes; circuit-tracer's ``Graph`` declares every one."""
    graph_module = pytest.importorskip("circuit_tracer.graph")
    read = {"input_tokens", "active_features", "selected_features", "activation_values", "adjacency_matrix"}
    read |= {"logit_targets", "vocab_size"}
    assert read <= set(graph_module.Graph.__annotations__)
