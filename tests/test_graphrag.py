from pathlib import Path

import pytest

from graphrag import GraphRAG
from graphrag.extract import Chunk, RuleExtractor
from graphrag.retrieval import BM25, HybridRetriever, rrf

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"


@pytest.fixture(scope="module")
def rag():
    r = GraphRAG(":memory:", extractor=RuleExtractor())
    r.ingest(DOCS)
    return r


def test_rule_extractor_reads_relations_and_types():
    triples = RuleExtractor().extract(Chunk("x#0", "x.md", "The Billing Service is owned by the Payments Team."))
    assert [(t.subject, t.relation, t.object, t.subject_type, t.object_type) for t in triples] == [
        ("Billing Service", "owned_by", "Payments Team", "Service", "Team")]


def test_ingest_is_idempotent(rag):
    before = rag.store.stats()
    rag.ingest(DOCS)
    assert rag.store.stats() == before


def test_every_edge_keeps_its_source_sentence(rag):
    for src, rel, dst, chunk_id in rag.store.edges():
        text = rag.store.chunk_text(chunk_id)
        assert src in text and dst in text, (src, rel, dst, text)


def test_two_hop_answer_has_evidence_path(rag):
    ans = rag.ask("Who leads the team that owns the Billing Service?")
    assert ans.answers == ["Ayesha Khan"]
    assert ans.hops == ["owned_by", "leads"]
    assert [ev.chunk_id for ev in ans.evidence[0]] == ["02_billing#1", "07_people#0"]


def test_question_word_order_does_not_matter(rag):
    a = rag.ask("Which database does the service behind the Revenue Dashboard store data in?")
    assert a.answers == ["Billing Postgres"]


def test_impact_analysis_walks_dependencies_in_reverse(rag):
    ans = rag.ask("What is affected if the Identity Service fails?")
    assert set(ans.answers) == {"Billing Service", "Ticket Service", "Revenue Dashboard", "Support Dashboard"}


def test_generic_word_is_not_linked_as_an_entity(rag):
    # "platform" is a relation cue here, not the Platform Team
    assert rag.ask("Which platform does the service that the Support Dashboard reads from run on?").answers == ["GKE"]


def test_out_of_scope_question_abstains(rag):
    ans = rag.ask("What is the capital of France?")
    assert ans.abstained and not ans.answers


def test_why_question_returns_the_causal_passage(rag):
    ans = rag.ask("Why did the Revenue Dashboard show zero revenue?")
    assert ans.intent == "retrieval"
    assert "expired credential" in ans.passages[0]


def test_bm25_prefers_rarer_terms():
    bm = BM25(["billing invoices", "billing usage", "usage events usage"])
    scores = bm.scores("invoices")
    assert scores.argmax() == 0 and scores[1] == 0


def test_rrf_rewards_agreement_between_rankers():
    fused = rrf([[1, 0, 2], [1, 2, 0]])
    assert max(fused, key=fused.get) == 1


def test_hybrid_retriever_finds_paraphrase():
    hits = HybridRetriever(["The Ticket Service routes tickets to agents.", "Invoices go out monthly."]).search("ticket routing")
    assert hits[0].index == 0
