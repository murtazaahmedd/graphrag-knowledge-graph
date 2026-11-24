"""Turn documents into chunks and (subject, relation, object) triples.

Two extractors share one interface:

* ``RuleExtractor`` - deterministic patterns, no API key, used by default and in tests.
* ``LLMExtractor`` - asks any LiteLLM model for triples in JSON; used when ``GRAPHRAG_LLM_MODEL`` is set.

Both return the same ``Triple`` objects, so the rest of the pipeline never knows which one ran.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Protocol

# relation name -> (regex verb phrase, subject type, object type)
RELATIONS = {
    "owned_by": (r"is owned by(?: the)?", "Component", "Team"),
    "depends_on": (r"depends on(?: the)?", "Service", "Component"),
    "stores_in": (r"stores data in(?: the)?", "Service", "Datastore"),
    "runs_on": (r"runs on(?: the)?", "Service", "Platform"),
    "reads_from": (r"reads from(?: the)?", "Dashboard", "Service"),
    "part_of": (r"is part of(?: the)?", "Team", "Department"),
    "leads": (r"leads(?: the)?", "Person", "Team"),
    "member_of": (r"is a member of(?: the)?", "Person", "Team"),
}

# A name is a run of capitalised words (allowing "Pub/Sub", "GKE", "Cloud Run").
_NAME = r"((?:[A-Z][\w/]*)(?:\s+[A-Z][\w/]*)*)"


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    text: str


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    subject_type: str
    object_type: str
    chunk_id: str


class Extractor(Protocol):
    def extract(self, chunk: Chunk) -> List[Triple]: ...


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"^#.*$", "", text, flags=re.MULTILINE)
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if len(p.strip()) > 3]


def chunk_documents(paths: Iterable[Path], sentences_per_chunk: int = 1) -> List[Chunk]:
    """One chunk per sentence keeps every triple traceable to the exact sentence that states it."""
    chunks: List[Chunk] = []
    for path in sorted(paths):
        sentences = split_sentences(path.read_text(encoding="utf-8"))
        for i in range(0, len(sentences), sentences_per_chunk):
            text = " ".join(sentences[i:i + sentences_per_chunk])
            chunks.append(Chunk(id=f"{path.stem}#{i // sentences_per_chunk}", doc=path.name, text=text))
    return chunks


def _clean(name: str) -> str:
    name = re.sub(r"^(The|the)\s+", "", name.strip())
    return name.rstrip(".,")


class RuleExtractor:
    """Pattern-based extractor: '<Name> <verb phrase> <Name>.'"""

    def __init__(self) -> None:
        self._patterns = [
            (rel, re.compile(rf"(?:The\s+)?{_NAME}\s+{verb}\s+{_NAME}"), s_type, o_type)
            for rel, (verb, s_type, o_type) in RELATIONS.items()
        ]

    def extract(self, chunk: Chunk) -> List[Triple]:
        triples = []
        for sentence in split_sentences(chunk.text):
            for rel, pattern, s_type, o_type in self._patterns:
                for m in pattern.finditer(sentence):
                    subj, obj = _clean(m.group(1)), _clean(m.group(2))
                    if subj and obj and subj != obj:
                        triples.append(Triple(subj, rel, obj, _refine(subj, s_type), _refine(obj, o_type), chunk.id))
        return triples


def _refine(name: str, declared: str) -> str:
    """Narrow broad declared types using the entity's own name."""
    lowered = name.lower()
    for suffix, kind in (("dashboard", "Dashboard"), ("service", "Service"), ("gateway", "Service"),
                         ("team", "Team"), ("department", "Department"), ("postgres", "Datastore")):
        if lowered.endswith(suffix):
            return kind
    return "Service" if declared == "Component" else declared


class LLMExtractor:
    """Optional: extract triples with an LLM via LiteLLM. Output is validated against RELATIONS."""

    PROMPT = (
        "Extract facts from the text as JSON list of objects with keys subject, relation, object. "
        "Allowed relations: {relations}. Use exact entity names from the text. Text:\n{text}"
    )

    def __init__(self, model: str) -> None:
        import litellm  # imported lazily so the default path has no dependency on it

        self._litellm = litellm
        self._model = model

    def extract(self, chunk: Chunk) -> List[Triple]:
        reply = self._litellm.completion(
            model=self._model,
            messages=[{"role": "user", "content": self.PROMPT.format(relations=", ".join(RELATIONS), text=chunk.text)}],
        )
        try:
            raw = json.loads(reply.choices[0].message.content.strip().strip("`").removeprefix("json"))
        except (ValueError, AttributeError):
            return []
        out = []
        for item in raw if isinstance(raw, list) else []:
            rel = item.get("relation")
            if rel in RELATIONS and item.get("subject") and item.get("object"):
                _, s_type, o_type = RELATIONS[rel]
                s, o = _clean(item["subject"]), _clean(item["object"])
                out.append(Triple(s, rel, o, _refine(s, s_type), _refine(o, o_type), chunk.id))
        return out


def default_extractor() -> Extractor:
    model = os.getenv("GRAPHRAG_LLM_MODEL")
    return LLMExtractor(model) if model else RuleExtractor()
