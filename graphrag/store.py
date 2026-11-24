"""Graph store on DuckDB: entities, edges (each tied to its source chunk) and chunks."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import duckdb

from .extract import Chunk, Triple

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks   (id TEXT PRIMARY KEY, doc TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS entities (name TEXT PRIMARY KEY, type TEXT);
CREATE TABLE IF NOT EXISTS edges    (src TEXT, rel TEXT, dst TEXT, chunk_id TEXT,
                                     PRIMARY KEY (src, rel, dst));
"""


class GraphStore:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.con = duckdb.connect(str(path))
        self.con.execute(SCHEMA)

    def reset(self) -> None:
        for table in ("edges", "entities", "chunks"):
            self.con.execute(f"DELETE FROM {table}")

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        rows = [(c.id, c.doc, c.text) for c in chunks]
        if rows:
            self.con.executemany("INSERT OR REPLACE INTO chunks VALUES (?, ?, ?)", rows)

    def add_triples(self, triples: Iterable[Triple]) -> int:
        """Idempotent: re-ingesting the same corpus never duplicates an entity or an edge."""
        n = 0
        for t in triples:
            for name, kind in ((t.subject, t.subject_type), (t.object, t.object_type)):
                self.con.execute("INSERT INTO entities VALUES (?, ?) ON CONFLICT DO NOTHING", [name, kind])
            self.con.execute("INSERT INTO edges VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
                             [t.subject, t.relation, t.object, t.chunk_id])
            n += 1
        return n

    def chunks(self) -> List[Tuple[str, str, str]]:
        return self.con.execute("SELECT id, doc, text FROM chunks ORDER BY id").fetchall()

    def entities(self) -> List[Tuple[str, str]]:
        return self.con.execute("SELECT name, type FROM entities ORDER BY name").fetchall()

    def edges(self) -> List[Tuple[str, str, str, str]]:
        return self.con.execute("SELECT src, rel, dst, chunk_id FROM edges ORDER BY src, rel, dst").fetchall()

    def chunk_text(self, chunk_id: str) -> str:
        row = self.con.execute("SELECT text FROM chunks WHERE id = ?", [chunk_id]).fetchone()
        return row[0] if row else ""

    def stats(self) -> dict:
        q = lambda t: self.con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]  # noqa: E731
        return {"chunks": q("chunks"), "entities": q("entities"), "edges": q("edges")}
