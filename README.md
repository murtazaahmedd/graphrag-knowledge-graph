# GraphRAG on a Knowledge Graph

Turn a folder of documents into a **knowledge graph**, then answer questions by **reasoning over the graph**,
with an **evidence path** for every answer, and abstain when the question is out of scope.

```text
$ python -m graphrag ask "Which database does the service behind the Revenue Dashboard store data in?"
Answer: Billing Postgres
Plan:   hops via reads_from -> stores_in
Path 1:
   Revenue Dashboard --reads_from--> Billing Service
      "The Revenue Dashboard reads from the Billing Service." [05_dashboards#1]
   Billing Service --stores_in--> Billing Postgres
      "The Billing Service stores data in Billing Postgres." [02_billing#4]
```

Plain vector RAG struggles with multi-hop questions like this one: the answer ("Billing Postgres") never appears
in the same passage as the thing asked about ("Revenue Dashboard"). A graph makes the hop explicit, and every edge
remembers the sentence it came from, so the answer is auditable.

Runs fully offline: no API key, no GPU, no external database.

## How it works

```text
 documents ──► sentence chunks ──► triple extraction ──► DuckDB graph store
                                   (rules, or an LLM)    entities · edges · chunks (each edge -> its source sentence)
                                                                   │
 question ──► entity linking ──► planner ──► graph traversal ──────┤──► answer + evidence paths
                │                  │            (multi-hop,         │
                │                  │             transitive,        │
                │                  │             impact, path)      │
                └──► hybrid retrieval: BM25 + TF-IDF, fused with RRF ──► supporting passages / abstain
```

| Step | What happens | Where |
|---|---|---|
| Extraction | Sentences become `(subject, relation, object)` triples with entity types. `RuleExtractor` is deterministic; set `GRAPHRAG_LLM_MODEL` to use any LiteLLM model instead, validated against the same relation schema. | `graphrag/extract.py` |
| Storage | DuckDB tables for chunks, entities and edges. Inserts are idempotent, so re-ingesting never duplicates. | `graphrag/store.py` |
| Entity linking | Longest-match on entity names and unambiguous short forms ("billing service"), skipping generic words that are also relation cues ("platform"). | `graphrag/reasoning.py` |
| Planning | Relation cues in the question ("owns", "leads", "behind", "store") become hops. Hops are ordered by what the graph can actually follow from the current node, so question word order does not matter. | `KnowledgeGraph.plan` |
| Reasoning | Multi-hop traversal in either edge direction; transitive dependency closure ("directly or indirectly"); **impact analysis** (reverse dependencies ranked by PageRank); shortest connection path between two entities. | `KnowledgeGraph.answer` |
| Hybrid retrieval | BM25 and TF-IDF cosine fused with Reciprocal Rank Fusion. Used for "why" questions and as supporting context. | `graphrag/retrieval.py` |
| Abstention | No linked entity and best passage similarity below a threshold means "I don't know" instead of a guess. | `graphrag/pipeline.py` |

## Results

20-question eval over the sample knowledge base (`evals/questions.json`), run in CI:

| Question type | Example | Correct |
|---|---|---|
| 1-hop | Which services does the Platform Team own? | 6/6 |
| 2-hop | Who leads the team that owns the Billing Service? | 6/6 |
| 3-hop | Who manages the department of the team that owns the Revenue Dashboard? | 2/2 |
| Transitive | What does the Revenue Dashboard depend on, directly or indirectly? | 1/1 |
| Impact | What breaks if the Event Gateway goes down? | 2/2 |
| Connection | How is Ayesha Khan connected to the Revenue Dashboard? | 1/1 |
| Out of scope | What is the capital of France? (must abstain) | 2/2 |

p95 latency is under 5 ms per question. The sample corpus is small and written for this demo, so treat these
numbers as a regression gate rather than a benchmark.

## Quick start

```bash
pip install -r requirements.txt
make ingest                                   # builds graph.duckdb from data/docs
make ask Q="What is affected if the Identity Service fails?"
make test                                     # 12 unit tests
make evals                                    # accuracy by question type
```

Use your own documents: put markdown files in a folder and run `python -m graphrag ingest path/to/folder`.
For free-form prose, the rule extractor will miss facts; set `GRAPHRAG_LLM_MODEL` (for example
`gemini/gemini-2.5-flash`) and install `litellm` to extract with an LLM.

## Project layout

```text
graphrag/
  extract.py     chunking, RuleExtractor, LLMExtractor
  store.py       DuckDB graph store
  retrieval.py   BM25, TF-IDF vector index, RRF fusion
  reasoning.py   entity linking, planner, traversal, PageRank impact, shortest path
  pipeline.py    GraphRAG: ingest() and ask()
  cli.py         command line
data/docs/       sample knowledge base (a fictional company's services, teams and incidents)
evals/           question set and runner
tests/           unit tests
```

## Ideas to extend

- Swap TF-IDF for BGE or sentence-transformer embeddings and store vectors with `pgvector`.
- Let an LLM write the traversal plan, then verify it against the schema before running it.
- Add community detection (Leiden) and summaries per community, as in Microsoft's GraphRAG.
- Load the graph into Neo4j and translate plans into Cypher.
