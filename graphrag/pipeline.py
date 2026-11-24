"""Ingest documents into the graph, then answer questions with graph reasoning plus hybrid retrieval."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .extract import Extractor, chunk_documents, default_extractor
from .reasoning import KnowledgeGraph
from .retrieval import HybridRetriever
from .store import GraphStore

# Below this TF-IDF cosine the best passage is treated as unrelated (out-of-distribution) and we abstain.
OOD_MIN_COSINE = 0.12
WHY_CUE = re.compile(r"\s*(why|explain|what caused)\b", re.IGNORECASE)


@dataclass
class Evidence:
    edge: str
    chunk_id: str
    text: str


@dataclass
class Answer:
    question: str
    answers: List[str]
    intent: str
    hops: List[str]
    evidence: List[List[Evidence]] = field(default_factory=list)
    passages: List[str] = field(default_factory=list)
    abstained: bool = False
    reason: str = ""

    def render(self) -> str:
        if self.abstained:
            return f"I don't know. ({self.reason})"
        lines = [f"Answer: {', '.join(self.answers) if self.answers else 'no match in the graph'}",
                 f"Plan:   {self.intent}" + (f" via {' -> '.join(self.hops)}" if self.hops else "")]
        for i, path in enumerate(self.evidence[:5], 1):
            lines.append(f"Path {i}:")
            lines += [f"   {ev.edge}\n      \"{ev.text}\" [{ev.chunk_id}]" for ev in path]
        if self.passages:
            lines.append("Supporting passages:")
            lines += [f"   - {p}" for p in self.passages]
        return "\n".join(lines)


class GraphRAG:
    def __init__(self, db_path: str | Path = ":memory:", extractor: Optional[Extractor] = None) -> None:
        self.store = GraphStore(db_path)
        self.extractor = extractor or default_extractor()
        self._load()

    def ingest(self, docs_dir: str | Path) -> dict:
        chunks = chunk_documents(Path(docs_dir).glob("*.md"))
        self.store.add_chunks(chunks)
        for chunk in chunks:
            self.store.add_triples(self.extractor.extract(chunk))
        self._load()
        return self.store.stats()

    def _load(self) -> None:
        self.kg = KnowledgeGraph(self.store.entities(), self.store.edges())
        rows = self.store.chunks()
        self.chunk_ids = [r[0] for r in rows]
        self._chunk_pos = {cid: i for i, cid in enumerate(self.chunk_ids)}
        self.retriever = HybridRetriever([r[2] for r in rows]) if rows else None

    def _with_context(self, index: int) -> str:
        """A passage plus the next sentence of the same document, so causes and explanations stay attached."""
        text = self.retriever.docs[index]
        doc, _, pos = self.chunk_ids[index].partition("#")
        nxt = f"{doc}#{int(pos) + 1}"
        if nxt in self._chunk_pos:
            text += " " + self.retriever.docs[self._chunk_pos[nxt]]
        return text

    def ask(self, question: str, top_k: int = 3) -> Answer:
        result = self.kg.answer(question)
        hits = self.retriever.search(question, top_k) if self.retriever else []
        passages = [self._with_context(h.index) for h in hits]

        # "why" questions need narrative context, not a single fact: answer from passages, keep graph facts
        if WHY_CUE.match(question) and hits:
            facts = [f"{s} {r.replace('_', ' ')} {d}" for s, r, d, _ in
                     [e for p in result.paths for e in p]][:5] if result.plan.intent == "describe" else []
            return Answer(question, facts, "retrieval", [], passages=passages)

        if result.plan.intent == "none":
            best = max((h.cosine for h in hits), default=0.0)
            if best < OOD_MIN_COSINE:
                return Answer(question, [], "none", [], abstained=True,
                              reason=f"no known entity and best passage similarity {best:.2f} < {OOD_MIN_COSINE}")
            return Answer(question, [], "retrieval", [], passages=passages)

        evidence = [[Evidence(f"{s} --{r}--> {d}", c, self.store.chunk_text(c)) for s, r, d, c in path]
                    for path in result.paths]
        return Answer(question, result.answers, result.plan.intent, result.plan.hops, evidence, passages)
