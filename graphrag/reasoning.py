"""Multi-hop reasoning over the knowledge graph.

A question is answered in four steps:

1. **Entity linking** - find graph entities named in the question (longest match first, suffix-tolerant).
2. **Planning** - turn relation cues ("owns", "leads", "reads from", ...) into an ordered list of hops,
   starting from the cue nearest the linked entity.
3. **Traversal** - follow each hop through the graph, in whichever edge direction exists, keeping the path.
4. **Evidence** - every edge on every path carries the sentence it was extracted from.

Special intents: *impact* ("what breaks if X fails") walks dependencies in reverse and ranks the affected
components with PageRank; *connection* ("how is X related to Y") returns the shortest path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

Edge = Tuple[str, str, str, str]  # (src, rel, dst, chunk_id)

RELATION_CUES: List[Tuple[str, str]] = [
    (r"\bown(?:s|ed|er|ers|ing)?\b|\bresponsible for\b", "owned_by"),
    (r"\bdepend(?:s|ency|encies)?\b|\bupstream\b", "depends_on"),
    (r"\bstore[sd]?\b|\bdatabase\b|\bdatastore\b", "stores_in"),
    (r"\bruns? on\b|\bhosted\b|\bdeployed\b|\bplatform\b", "runs_on"),
    (r"\breads? from\b|\bbehind\b|\bpowers?\b|\bfeeds?\b", "reads_from"),
    (r"\blead(?:s|er)?\b|\bmanag(?:es|er)\b|\bheads?\b|\bin charge of\b", "leads"),
    (r"\bmembers?\b|\bworks? (?:in|on|for)\b", "member_of"),
    (r"\bpart of\b|\bdepartment\b|\bbelongs? to\b", "part_of"),
]
IMPACT_CUE = re.compile(r"\baffected\b|\bimpact(?:ed|s)?\b|\bbreaks?\b|\bfails?\b|\bgoes down\b|\boutage\b")
CONNECTION_CUE = re.compile(r"\bconnected\b|\brelated\b|\bpath\b|\blinked?\b")
TRANSITIVE_CUE = re.compile(r"\bindirect(?:ly)?\b|\btransitive(?:ly)?\b|\ball upstream\b|\bfull chain\b")
GENERIC_SUFFIXES = ("service", "team", "dashboard", "department", "gateway")
DEPENDENCY_RELS = {"depends_on", "reads_from"}
GENERIC_WORDS = {"support", "data", "usage", "event", "identity", "engineering"}


@dataclass
class Plan:
    intent: str                         # hops | impact | connection | describe | none
    seeds: List[str]
    hops: List[str] = field(default_factory=list)
    transitive: bool = False


@dataclass
class Result:
    answers: List[str]
    paths: List[List[Edge]]
    plan: Plan


class KnowledgeGraph:
    def __init__(self, entities: Sequence[Tuple[str, str]], edges: Sequence[Edge]) -> None:
        self.types: Dict[str, str] = dict(entities)
        self.edges: List[Edge] = list(edges)
        self.out: Dict[Tuple[str, str], List[Edge]] = {}
        self.inc: Dict[Tuple[str, str], List[Edge]] = {}
        for e in self.edges:
            self.out.setdefault((e[0], e[1]), []).append(e)
            self.inc.setdefault((e[2], e[1]), []).append(e)
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.types)
        for s, r, d, _ in self.edges:
            self.graph.add_edge(s, d, rel=r)
        self.pagerank = nx.pagerank(self.graph) if self.graph.number_of_edges() else {}
        self._aliases = self._build_aliases()

    # ---------------------------------------------------------------- linking
    def _build_aliases(self) -> Dict[str, str]:
        aliases = {name.lower(): name for name in self.types}
        short: Dict[str, List[str]] = {}
        for name in self.types:
            words = name.lower().split()
            if len(words) > 1 and words[-1] in GENERIC_SUFFIXES:
                short.setdefault(" ".join(words[:-1]), []).append(name)
        for alias, names in short.items():
            # only unambiguous short forms, and never a word that is also a relation cue ("platform", "support")
            clashes = any(re.search(p, alias) for p, _ in RELATION_CUES) or alias in GENERIC_WORDS
            if len(names) == 1 and alias not in aliases and not clashes:
                aliases[alias] = names[0]
        return aliases

    def link(self, question: str) -> List[Tuple[str, int]]:
        """Entities mentioned in the question, with their character position, longest alias first."""
        text = question.lower()
        taken: List[Tuple[int, int]] = []
        found: List[Tuple[str, int]] = []
        for alias in sorted(self._aliases, key=len, reverse=True):
            for m in re.finditer(rf"\b{re.escape(alias)}\b", text):
                if any(m.start() < e and m.end() > s for s, e in taken):
                    continue
                taken.append((m.start(), m.end()))
                name = self._aliases[alias]
                if name not in [f for f, _ in found]:
                    found.append((name, m.start()))
        return sorted(found, key=lambda x: x[1])

    # --------------------------------------------------------------- planning
    def plan(self, question: str) -> Plan:
        q = question.lower()
        linked = self.link(question)
        seeds = [n for n, _ in linked]
        if not seeds:
            return Plan("none", [])
        if len(seeds) >= 2 and CONNECTION_CUE.search(q):
            return Plan("connection", seeds[:2])
        if IMPACT_CUE.search(q):
            return Plan("impact", seeds[:1])

        # mask entity names so "Billing Service" does not trigger the "service" cues, etc.
        masked = q
        for name, pos in linked:
            alias_len = next(len(a) for a in sorted(self._aliases, key=len, reverse=True)
                             if self._aliases[a] == name and q.startswith(a, pos))
            masked = masked[:pos] + " " * alias_len + masked[pos + alias_len:]

        cues: List[Tuple[int, str]] = []
        for pattern, rel in RELATION_CUES:
            cues += [(m.start(), rel) for m in re.finditer(pattern, masked)]
        if not cues:
            return Plan("describe", seeds[:1])

        seed, seed_pos = linked[0]
        transitive = bool(TRANSITIVE_CUE.search(q))
        # distance of each relation's nearest cue from the entity: closer cues are tried first
        distance: Dict[str, int] = {}
        for pos, rel in cues:
            distance[rel] = min(distance.get(rel, 10**6), abs(pos - seed_pos))

        # Data-aware ordering: from the seed, repeatedly take the closest remaining relation that the graph
        # can actually follow from the current frontier. This makes word order in the question irrelevant.
        hops: List[str] = []
        frontier: Dict[str, List[Edge]] = {seed: []}
        remaining = sorted(distance, key=distance.get)
        while remaining:
            for rel in remaining:
                nxt = self._advance(frontier, rel, transitive)
                if nxt:
                    hops.append(rel)
                    remaining.remove(rel)
                    frontier = nxt
                    break
            else:
                break
        if not hops:
            hops = remaining[:1]          # nothing followable: keep the closest cue so the result is "no match"
        return Plan("hops", [seed], hops, transitive=transitive)

    def _advance(self, frontier: Dict[str, List[Edge]], rel: str, transitive: bool) -> Dict[str, List[Edge]]:
        if transitive and rel in DEPENDENCY_RELS:
            merged: Dict[str, List[Edge]] = {}
            for node, path in frontier.items():
                for nb, sub in self.closure(node, DEPENDENCY_RELS, reverse=False).items():
                    merged.setdefault(nb, path + sub)
            return merged
        return self.step(frontier, rel)

    # -------------------------------------------------------------- traversal
    def step(self, frontier: Dict[str, List[Edge]], rel: str) -> Dict[str, List[Edge]]:
        nxt: Dict[str, List[Edge]] = {}
        for node, path in frontier.items():
            outgoing = self.out.get((node, rel), [])
            incoming = self.inc.get((node, rel), [])
            for e in outgoing:
                nxt.setdefault(e[2], path + [e])
            if not outgoing:
                for e in incoming:
                    nxt.setdefault(e[0], path + [e])
        return nxt

    def closure(self, start: str, rels: Set[str], reverse: bool) -> Dict[str, List[Edge]]:
        """Every node reachable through ``rels`` (BFS), with the path that reached it."""
        seen: Dict[str, List[Edge]] = {start: []}
        queue = [start]
        while queue:
            node = queue.pop(0)
            candidates = []
            for rel in rels:
                candidates += [(e, e[0]) for e in self.inc.get((node, rel), [])] if reverse else \
                              [(e, e[2]) for e in self.out.get((node, rel), [])]
            for edge, nb in candidates:
                if nb not in seen:
                    seen[nb] = seen[node] + [edge]
                    queue.append(nb)
        seen.pop(start)
        return seen

    def answer(self, question: str) -> Result:
        plan = self.plan(question)
        if plan.intent == "none":
            return Result([], [], plan)

        if plan.intent == "describe":
            seed = plan.seeds[0]
            edges = [e for e in self.edges if seed in (e[0], e[2])]
            return Result([f"{s} {r.replace('_', ' ')} {d}" for s, r, d, _ in edges], [[e] for e in edges], plan)

        if plan.intent == "impact":
            reached = self.closure(plan.seeds[0], DEPENDENCY_RELS, reverse=True)
            ranked = sorted(reached, key=lambda n: -self.pagerank.get(n, 0.0))
            return Result(ranked, [reached[n] for n in ranked], plan)

        if plan.intent == "connection":
            a, b = plan.seeds
            try:
                nodes = nx.shortest_path(self.graph.to_undirected(as_view=True), a, b)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return Result([], [], plan)
            path = []
            for u, v in zip(nodes, nodes[1:]):
                path.append(next(e for e in self.edges if {e[0], e[2]} == {u, v}))
            return Result([" -> ".join(nodes)], [path], plan)

        # multi-hop
        frontier: Dict[str, List[Edge]] = {plan.seeds[0]: []}
        for rel in plan.hops:
            frontier = self._advance(frontier, rel, plan.transitive)
            if not frontier:
                break
        return Result(sorted(frontier), [frontier[n] for n in sorted(frontier)], plan)
