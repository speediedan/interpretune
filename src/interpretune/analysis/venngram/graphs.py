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
r"""
Exporting an Attribution Graph as a venngram Graph Subset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The inverse of applying a graph subset (:func:`~interpretune.analysis.venngram.apply.recipe_feature_interventions` with
``target: {graph: <name>}``): a circuit-tracer attribution graph, usually pruned, written as venngram's ``graph_subset``
document, so the reasoning it describes in one prompt can be applied in another, or read by any consumer without
interpretune installed.

Each selected feature of the graph is one feature node: its layer, source position and feature index, its activation,
and an anchor that places it in a new prompt. Edges between the exported features carry their attributions, as
evidence. The source prompt's token ids, the attribution targets and the pruning settings are the document's provenance.

**Anchors** (venngram's :class:`~venngram.graphs.Anchor`) are chosen once for the whole graph:

- ``from_end`` (the default): the node's offset from the source prompt's last token, so a node at the final position
  lands on the destination's final position whatever its length;
- ``absolute``: the node's source position;
- ``token_match``: the source token at the node's position, and which occurrence of it that position is, so the node
  follows that token in the destination. It needs the ``tokenizer`` the source was encoded with;
- ``all``: every destination position.

Only feature nodes are written: error, embedding and logit nodes carry no activation a graph keeps, and a treatment
never applies them. A graph whose features belong to no transcoder or cross-layer transcoder space is still written, but
:func:`~interpretune.analysis.venngram.apply.recipe_feature_interventions` applies only those two kinds.

.. warning:: This is an experimental API and may change in a future release.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from interpretune.analysis.venngram.fetch import _require
from interpretune.utils import MisconfigurationException

#: The anchor kinds an export can give every node.
EXPORT_ANCHORS = ("from_end", "absolute", "token_match", "all")
#: The site feature nodes are written at by default: the one a graph-targeted stage names.
DEFAULT_GRAPH_SITE = "transcoder.features"


def _anchor(kind: str, position: int, token_ids: list[int]) -> dict[str, Any]:
    if kind == "from_end":
        return {"kind": "from_end", "offset": len(token_ids) - 1 - position}
    if kind == "absolute":
        return {"kind": "absolute", "position": position}
    if kind == "token_match":
        token = token_ids[position]
        return {"kind": "token_match", "token_ids": [token], "occurrence": token_ids[:position].count(token)}
    return {"kind": "all"}


def _attribution_targets(graph: Any, last: int) -> list[dict[str, Any]]:
    """Each attribution target: a vocabulary logit by its token id, else a custom target by its label."""
    vocab = getattr(graph, "vocab_size", None)
    out = []
    for target in getattr(graph, "logit_targets", None) or ():
        idx, label = int(target.vocab_idx), str(target.token_str)
        if vocab is None or idx < vocab:
            out.append({"kind": "logit", "token_id": idx, "token": label, "position": last})
        else:
            out.append({"kind": "custom", "label": label, "position": last})
    return out


def export_graph_subset(
    graph: Any,
    *,
    name: str,
    model: Mapping[str, str],
    space: Mapping[str, Any],
    space_name: str = "tc",
    site: str = DEFAULT_GRAPH_SITE,
    anchor: str = "from_end",
    edges: bool = True,
    tokenizer: Mapping[str, str] | None = None,
    pruning: Mapping[str, Any] | None = None,
    description: str | None = None,
) -> Any:
    """Write a circuit-tracer attribution graph's selected features as a venngram graph subset.

    Pass a pruned graph (``build_pruned_graph``) to export the subset pruning kept; a full graph exports every
    selected feature.

    Args:
        graph: A ``circuit_tracer.graph.Graph``, or anything with its ``input_tokens``, ``active_features``,
            ``selected_features``, ``activation_values`` and ``adjacency_matrix``.
        name: The subset's name, the one a book's ``graphs:`` entry and a treatment's ``target: {graph: <name>}`` use.
        model: ``{id, revision}`` of the model the graph was attributed on: its coordinates mean something in that
            revision only.
        space: ``{id, revision, kind}`` of the transcoders (``kind`` is ``transcoder`` or ``clt`` for the graph to
            apply through circuit-tracer).
        space_name: The space's name in the document.
        site: The site every feature node is written at; a graph-targeted stage names the same site.
        anchor: How each node lands in a new prompt: ``from_end``, ``absolute``, ``token_match`` or ``all`` (see the
            module docs).
        edges: Write the attributions between the exported features as edges.
        tokenizer: ``{id, revision}`` of the tokenizer the source prompt was encoded with. A ``token_match`` anchor
            needs it, since a token id means something in one tokenizer only.
        pruning: How the graph was pruned (``{node_threshold, edge_threshold}``, say), recorded as provenance.
        description: The document's description.

    Returns:
        A ``venngram.graphs.GraphSubset``, checked by venngram as it would read it. ``.write(path)`` writes it, with
        its table in a sidecar when it is large.

    Raises:
        MisconfigurationException: For an anchor kind outside :data:`EXPORT_ANCHORS`, a ``token_match`` anchor with no
            ``tokenizer``, a graph with no selected features, or activation values that do not line up with them.
        venngram.errors.AddressError: For anything venngram refuses in the document (a model or space without its
            revision, a repeated source coordinate, a value that is not finite).
    """
    GraphSubset = _require("venngram.graphs").GraphSubset

    if anchor not in EXPORT_ANCHORS:
        raise MisconfigurationException(f"anchor {anchor!r} is none of {list(EXPORT_ANCHORS)}")
    if anchor == "token_match" and not tokenizer:
        raise MisconfigurationException(
            "a `token_match` anchor needs the source `tokenizer` ({id, revision}): a token id means something in one"
            " tokenizer only"
        )
    selected = [int(i) for i in graph.selected_features.tolist()]
    activations = [float(v) for v in graph.activation_values.tolist()]
    if not selected:
        raise MisconfigurationException(f"graph subset {name!r}: the graph has no selected features to export")
    if len(activations) != len(selected):
        raise MisconfigurationException(
            f"graph subset {name!r}: {len(activations)} activation values for {len(selected)} selected features"
        )
    token_ids = [int(t) for t in graph.input_tokens.tolist()]
    rows = graph.active_features.tolist()
    nodes = []
    for k, (i, activation) in enumerate(zip(selected, activations)):
        layer, position, feature = (int(v) for v in rows[i])
        nodes.append(
            {
                "id": f"f{k}",
                "kind": "feature",
                "site": site,
                "space": space_name,
                "layer": layer,
                "position": position,
                "index": {"feature": feature},
                "activation": activation,
                "anchor": _anchor(anchor, position, token_ids),
            }
        )
    source: dict[str, Any] = {"token_ids": token_ids}
    if tokenizer:
        source["tokenizer"] = dict(tokenizer)
    if targets := _attribution_targets(graph, len(token_ids) - 1):
        source["attribution_targets"] = targets
    if pruning:
        source["pruning"] = dict(pruning)
    doc: dict[str, Any] = {
        "schema_version": 1,
        "kind": "graph_subset",
        "name": name,
        "model": dict(model),
        "spaces": {space_name: dict(space)},
        "source": source,
        "nodes": nodes,
    }
    if description is not None:
        doc["description"] = description
    if edges:
        # rows are targets and columns sources; the selected features are the first nodes, in this order
        n = len(selected)
        block = graph.adjacency_matrix[:n, :n].detach().cpu()
        doc["edges"] = [
            {"source": f"f{j}", "target": f"f{i}", "attribution": float(block[i, j])}
            for i, j in block.nonzero(as_tuple=False).tolist()
        ]
    return GraphSubset.from_mapping(doc, source=f"<graph subset {name}>")
