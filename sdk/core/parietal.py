from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import difflib
import io, json
import networkx as nx

@dataclass
class Evidence:
    source: str
    url: Optional[str] = None
    timestamp: Optional[str] = None  # ISO8601
    confidence: float = 1.0
    snippet: Optional[str] = None

@dataclass
class Fact:
    h: str; r: str; t: str; evidence: Optional[Evidence]

def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=a.lower(), b=b.lower()).ratio()

class ParietalGraph:
    """Ontology-aware KG with light canonicalisation and evidence on edges."""
    def __init__(self, dedupe_threshold: float = 0.90):
        self._thr = dedupe_threshold
        self._g = nx.MultiDiGraph()

    def _canonicalise(self, term: str) -> str:
        for node in self._g.nodes:
            if _similar(term, self._g.nodes[node].get("label", node)) >= self._thr:
                return node
        return term

    def upsert_triplet(self, h: str, r: str, t: str, evidence: Optional[Evidence] = None) -> None:
        h_id = self._canonicalise(h)
        t_id = self._canonicalise(t)
        self._g.add_node(h_id, label=self._g.nodes.get(h_id, {}).get("label", h))
        self._g.add_node(t_id, label=self._g.nodes.get(t_id, {}).get("label", t))
        self._g.add_edge(h_id, t_id, key=r.lower(), relation=r, evidence=evidence)

    def nearest_facts(self, entity_or_term: str, k: int = 8) -> List[Fact]:
        # pick the closest node label, then take outgoing edges
        scored = [(_similar(entity_or_term, self._g.nodes[n].get("label", n)), n) for n in self._g.nodes]
        if not scored:
            return []
        scored.sort(reverse=True)
        best = [n for _, n in scored[:k]]
        facts: List[Fact] = []
        for n in best:
            for _, tgt, key, data in self._g.out_edges(n, keys=True, data=True):
                facts.append(Fact(
                    h=self._g.nodes[n].get("label", n),
                    r=data.get("relation", key),
                    t=self._g.nodes[tgt].get("label", tgt),
                    evidence=data.get("evidence")
                ))
        return facts[:k]

    def export(self, fmt: str = "graphml") -> bytes:
        if fmt == "graphml":
            buf = io.BytesIO()
            g = nx.MultiDiGraph()
            for node, data in self._g.nodes(data=True):
                g.add_node(node, **data)
            for u, v, key, data in self._g.edges(keys=True, data=True):
                evidence = data.get("evidence")
                payload = data.copy()
                if evidence is not None:
                    payload["evidence"] = json.dumps(evidence.__dict__)
                g.add_edge(u, v, key=key, **payload)
            nx.write_graphml(g, buf)
            return buf.getvalue()
        if fmt == "jsonl":
            buf = io.StringIO()
            for u, v, k, data in self._g.edges(keys=True, data=True):
                ev = data.get("evidence")
                buf.write(json.dumps({
                    "h": self._g.nodes[u].get("label", u),
                    "r": data.get("relation", k),
                    "t": self._g.nodes[v].get("label", v),
                    "evidence": ev.__dict__ if ev else None
                }) + "\n")
            return buf.getvalue().encode()
        raise ValueError(f"Unsupported format: {fmt}")
