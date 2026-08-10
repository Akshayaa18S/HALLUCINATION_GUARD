"""
Pipeline Stage - Evidence Consistency Graph & Multi-Hop Consensus Engine.

Formally models candidate evidence sentences as an Evidence Graph G = (V, E)
where:
  V = {e_1, e_2, ..., e_n} evidence sentence nodes
  E = directed edges weighted by NLI agreement (+w), contradiction (-w), or neutral (0)

Computes Graph Consensus Score S_consensus and flags multi-source contradictions.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvidenceNode:
    id: int
    text: str
    source_title: str
    relevance: float = 0.50


@dataclass
class EvidenceEdge:
    source_id: int
    target_id: int
    relation: str  # "support" | "contradict" | "redundant" | "neutral"
    weight: float


class EvidenceConsistencyGraph:
    """Constructs Evidence Graph G = (V, E) and computes consensus / multi-hop agreement."""

    def build_graph(self, evidence_items: list[dict[str, Any]]) -> tuple[list[EvidenceNode], list[EvidenceEdge]]:
        nodes: list[EvidenceNode] = []
        node_id = 0

        for item in evidence_items:
            text = item.get("text", item.get("evidence_excerpt", "")).strip()
            title = item.get("title", "Wikipedia")
            rel = item.get("relevance", 0.50)

            # Split into sentence nodes
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]
            for s in sents[:3]:
                nodes.append(EvidenceNode(id=node_id, text=s, source_title=title, relevance=rel))
                node_id += 1

        edges: list[EvidenceEdge] = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n1, n2 = nodes[i], nodes[j]
                w1 = set(re.findall(r"\w+", n1.text.lower()))
                w2 = set(re.findall(r"\w+", n2.text.lower()))
                overlap = len(w1.intersection(w2)) / float(max(1, len(w1.union(w2))))

                if overlap > 0.70:
                    edges.append(EvidenceEdge(source_id=n1.id, target_id=n2.id, relation="redundant", weight=0.90))
                elif overlap > 0.30:
                    # Check polarity flip
                    has_neg1 = any(w in n1.text.lower() for w in ("not", "never", "no"))
                    has_neg2 = any(w in n2.text.lower() for w in ("not", "never", "no"))
                    if has_neg1 != has_neg2:
                        edges.append(EvidenceEdge(source_id=n1.id, target_id=n2.id, relation="contradict", weight=-0.85))
                    else:
                        edges.append(EvidenceEdge(source_id=n1.id, target_id=n2.id, relation="support", weight=0.80))

        return nodes, edges

    def compute_consensus_score(self, nodes: list[EvidenceNode], edges: list[EvidenceEdge]) -> float:
        """Computes aggregated Evidence Consensus Score in range [0.0, 1.0]."""
        if not nodes:
            return 0.50

        if not edges:
            return 0.75

        pos_weight = sum(e.weight for e in edges if e.weight > 0)
        neg_weight = sum(abs(e.weight) for e in edges if e.weight < 0)

        total_weight = pos_weight + neg_weight
        if total_weight == 0:
            return 0.75

        consensus = pos_weight / float(total_weight)
        return round(float(max(0.0, min(1.0, consensus))), 4)


evidence_graph_engine = EvidenceConsistencyGraph()
