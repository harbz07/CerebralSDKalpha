from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Dict
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

@dataclass
class ReasoningPath:
    """A chain of connected facts representing a reasoning path."""
    facts: List[Fact]
    confidence: float
    length: int

    def __str__(self) -> str:
        path_str = " -> ".join(
            f"{f.h} --[{f.r}]--> {f.t}" for f in self.facts
        )
        return f"Path (conf={self.confidence:.2f}, len={self.length}): {path_str}"

def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=a.lower(), b=b.lower()).ratio()

class ParietalGraph:
    """Ontology-aware KG with light canonicalisation and evidence on edges.

    Enhanced with:
    - Path finding and reasoning chains
    - Subgraph extraction
    - Bidirectional fact retrieval
    - Temporal filtering
    - Confidence scoring
    """
    def __init__(self, dedupe_threshold: float = 0.90):
        self._thr = dedupe_threshold
        self._g = nx.MultiDiGraph()

    def _canonicalise(self, term: str) -> str:
        for node in self._g.nodes:
            if _similar(term, self._g.nodes[node].get("label", node)) >= self._thr:
                return node
        return term

    def upsert_triplet(self, h: str, r: str, t: str, evidence: Optional[Evidence] = None) -> None:
        """Add or update a triplet (h, r, t) with optional evidence."""
        h_id = self._canonicalise(h)
        t_id = self._canonicalise(t)
        self._g.add_node(h_id, label=self._g.nodes.get(h_id, {}).get("label", h))
        self._g.add_node(t_id, label=self._g.nodes.get(t_id, {}).get("label", t))
        self._g.add_edge(h_id, t_id, key=r.lower(), relation=r, evidence=evidence)

    def nearest_facts(self, entity_or_term: str, k: int = 8) -> List[Fact]:
        """Get facts for the nearest matching entities (outgoing edges only)."""
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

    def get_facts_bidirectional(self, entity: str, k: int = 8) -> Tuple[List[Fact], List[Fact]]:
        """Get both outgoing and incoming facts for an entity.

        Returns:
            Tuple of (outgoing_facts, incoming_facts)
        """
        ent_id = self._canonicalise(entity)
        if ent_id not in self._g.nodes:
            return ([], [])

        # Outgoing facts
        outgoing = []
        for _, tgt, key, data in self._g.out_edges(ent_id, keys=True, data=True):
            outgoing.append(Fact(
                h=self._g.nodes[ent_id].get("label", ent_id),
                r=data.get("relation", key),
                t=self._g.nodes[tgt].get("label", tgt),
                evidence=data.get("evidence")
            ))

        # Incoming facts
        incoming = []
        for src, _, key, data in self._g.in_edges(ent_id, keys=True, data=True):
            incoming.append(Fact(
                h=self._g.nodes[src].get("label", src),
                r=data.get("relation", key),
                t=self._g.nodes[ent_id].get("label", ent_id),
                evidence=data.get("evidence")
            ))

        return (outgoing[:k], incoming[:k])

    def find_paths(
        self,
        start: str,
        end: str,
        max_length: int = 3,
        limit: int = 5
    ) -> List[ReasoningPath]:
        """Find reasoning paths between two entities.

        Args:
            start: Starting entity
            end: Target entity
            max_length: Maximum path length (number of hops)
            limit: Maximum number of paths to return

        Returns:
            List of ReasoningPath objects, ordered by confidence
        """
        start_id = self._canonicalise(start)
        end_id = self._canonicalise(end)

        if start_id not in self._g.nodes or end_id not in self._g.nodes:
            return []

        paths = []
        try:
            # Find all simple paths up to max_length
            for path_nodes in nx.all_simple_paths(self._g, start_id, end_id, cutoff=max_length):
                # Extract facts along the path
                facts = []
                total_confidence = 1.0

                for i in range(len(path_nodes) - 1):
                    u, v = path_nodes[i], path_nodes[i + 1]
                    # Get edge data (take first edge if multiple)
                    edge_data = self._g.get_edge_data(u, v)
                    if edge_data:
                        key = list(edge_data.keys())[0]
                        data = edge_data[key]

                        ev = data.get("evidence")
                        if ev:
                            total_confidence *= ev.confidence

                        facts.append(Fact(
                            h=self._g.nodes[u].get("label", u),
                            r=data.get("relation", key),
                            t=self._g.nodes[v].get("label", v),
                            evidence=ev
                        ))

                if facts:
                    paths.append(ReasoningPath(
                        facts=facts,
                        confidence=total_confidence,
                        length=len(facts)
                    ))

                if len(paths) >= limit:
                    break
        except nx.NetworkXNoPath:
            pass

        # Sort by confidence (higher first)
        paths.sort(key=lambda p: p.confidence, reverse=True)
        return paths[:limit]

    def get_subgraph(self, entities: List[str], depth: int = 1) -> "ParietalGraph":
        """Extract a subgraph containing specified entities and their neighbors.

        Args:
            entities: List of entity names to include
            depth: How many hops to include (1 = direct neighbors only)

        Returns:
            New ParietalGraph containing the subgraph
        """
        # Canonicalize entity names
        entity_ids = set()
        for ent in entities:
            ent_id = self._canonicalise(ent)
            if ent_id in self._g.nodes:
                entity_ids.add(ent_id)

        if not entity_ids:
            return ParietalGraph(dedupe_threshold=self._thr)

        # Expand to include neighbors up to depth
        nodes_to_include = set(entity_ids)
        for _ in range(depth):
            neighbors = set()
            for node in nodes_to_include:
                neighbors.update(self._g.successors(node))
                neighbors.update(self._g.predecessors(node))
            nodes_to_include.update(neighbors)

        # Create new subgraph
        subgraph = ParietalGraph(dedupe_threshold=self._thr)
        for node in nodes_to_include:
            # Copy node attributes
            subgraph._g.add_node(node, **self._g.nodes[node])

        # Copy edges
        for u, v, key, data in self._g.edges(keys=True, data=True):
            if u in nodes_to_include and v in nodes_to_include:
                subgraph._g.add_edge(u, v, key=key, **data)

        return subgraph

    def get_relations_by_type(self, relation_type: str) -> List[Fact]:
        """Get all facts with a specific relation type."""
        facts = []
        for u, v, key, data in self._g.edges(keys=True, data=True):
            if data.get("relation", key).upper() == relation_type.upper():
                facts.append(Fact(
                    h=self._g.nodes[u].get("label", u),
                    r=data.get("relation", key),
                    t=self._g.nodes[v].get("label", v),
                    evidence=data.get("evidence")
                ))
        return facts

    def get_entities_by_degree(self, top_k: int = 10, direction: str = "out") -> List[Tuple[str, int]]:
        """Get most connected entities.

        Args:
            top_k: Number of top entities to return
            direction: "out" (outgoing), "in" (incoming), or "both"

        Returns:
            List of (entity_name, degree) tuples
        """
        if direction == "out":
            degrees = [(self._g.nodes[n].get("label", n), self._g.out_degree(n)) for n in self._g.nodes]
        elif direction == "in":
            degrees = [(self._g.nodes[n].get("label", n), self._g.in_degree(n)) for n in self._g.nodes]
        else:  # both
            degrees = [(self._g.nodes[n].get("label", n), self._g.degree(n)) for n in self._g.nodes]

        degrees.sort(key=lambda x: x[1], reverse=True)
        return degrees[:top_k]

    def stats(self) -> Dict[str, int]:
        """Get graph statistics."""
        relations: Dict[str, int] = {}
        for _, _, _, data in self._g.edges(keys=True, data=True):
            rel = data.get("relation", "unknown")
            relations[rel] = relations.get(rel, 0) + 1

        return {
            "nodes": self._g.number_of_nodes(),
            "edges": self._g.number_of_edges(),
            "relations": relations
        }

    def export(self, fmt: str = "graphml") -> bytes:
        """Export graph to GraphML or JSONL format."""
        if fmt == "graphml":
            buf = io.BytesIO()
            graph = nx.MultiDiGraph()
            graph.add_nodes_from(self._g.nodes(data=True))
            for u, v, key, data in self._g.edges(keys=True, data=True):
                clean = {k: v for k, v in data.items() if k != "evidence"}
                ev = data.get("evidence")
                if ev is not None:
                    clean["evidence"] = json.dumps(ev.__dict__)
                graph.add_edge(u, v, key=key, **clean)
            nx.write_graphml(graph, buf)
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
