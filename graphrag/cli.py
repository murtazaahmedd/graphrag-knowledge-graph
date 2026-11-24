"""Command line: ``python -m graphrag ingest data/docs`` then ``python -m graphrag ask "..."``."""
from __future__ import annotations

import argparse
import os

from .pipeline import GraphRAG

DB = os.getenv("GRAPHRAG_DB", "graph.duckdb")


def main() -> None:
    parser = argparse.ArgumentParser(prog="graphrag", description="GraphRAG over a knowledge graph")
    sub = parser.add_subparsers(dest="cmd", required=True)
    ing = sub.add_parser("ingest", help="extract a knowledge graph from markdown documents")
    ing.add_argument("docs", nargs="?", default="data/docs")
    ask = sub.add_parser("ask", help="answer a question with evidence paths")
    ask.add_argument("question")
    sub.add_parser("graph", help="print every edge in the graph")
    args = parser.parse_args()

    rag = GraphRAG(DB)
    if args.cmd == "ingest":
        rag.store.reset()
        print(rag.ingest(args.docs))
    elif args.cmd == "ask":
        print(rag.ask(args.question).render())
    else:
        for s, r, d, _ in rag.store.edges():
            print(f"{s} --{r}--> {d}")


if __name__ == "__main__":
    main()
