"""Run the question set against a freshly ingested graph and report accuracy by question kind."""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from graphrag import GraphRAG
from graphrag.extract import RuleExtractor

ROOT = Path(__file__).resolve().parents[1]


def run(min_accuracy: float = 1.0) -> int:
    rag = GraphRAG(":memory:", extractor=RuleExtractor())
    rag.ingest(ROOT / "data" / "docs")
    questions = json.loads((ROOT / "evals" / "questions.json").read_text())

    by_kind = defaultdict(lambda: [0, 0])
    latencies, failures = [], []
    for item in questions:
        start = time.perf_counter()
        ans = rag.ask(item["q"])
        latencies.append((time.perf_counter() - start) * 1000)
        ok = ans.abstained if item["expect"] == "ABSTAIN" else (not ans.abstained and sorted(ans.answers) == sorted(item["expect"]))
        by_kind[item["kind"]][0] += ok
        by_kind[item["kind"]][1] += 1
        if not ok:
            failures.append((item["q"], item["expect"], ans.answers, ans.intent, ans.hops))

    total = sum(v[0] for v in by_kind.values()), sum(v[1] for v in by_kind.values())
    print(f"{'kind':<14}{'correct':>10}")
    for kind, (ok, n) in by_kind.items():
        print(f"{kind:<14}{ok:>6}/{n}")
    latencies.sort()
    print(f"\nOverall: {total[0]}/{total[1]}  |  p95 latency {latencies[int(0.95 * (len(latencies) - 1))]:.1f} ms")
    for q, exp, got, intent, hops in failures:
        print(f"  FAIL  {q}\n        expected {exp}, got {got} ({intent} {hops})")
    return 0 if total[0] / total[1] >= min_accuracy else 1


if __name__ == "__main__":
    sys.exit(run())
