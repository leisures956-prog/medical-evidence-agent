from __future__ import annotations

from dataclasses import dataclass

import pytest
from app.medical_models import MedicalDocument, RetrievalHit
from app.pgvector_retriever import (
    PersistentHybridRetriever,
    PersistentVectorRetriever,
    split_text,
)
from app.pgvector_store import (
    PgVectorStore,
    PgVectorUnavailable,
    schema_statements,
    vector_literal,
)
from app.search import SearchService


def test_vector_literal_is_deterministic_and_rejects_non_finite_values():
    assert vector_literal([1, 0.125, -0.5]) == "[1,0.125,-0.5]"
    with pytest.raises(ValueError, match="NaN"):
        vector_literal([float("nan")])


def test_schema_contains_dimension_and_hnsw_index():
    statements = schema_statements(3)
    sql = "\n".join(statements)
    assert "embedding vector(3)" in sql
    assert "vector_cosine_ops" in sql
    assert "rag_corpus_versions" in sql
    assert "content_sha256" in sql


def test_split_text_is_deterministic_and_validates_overlap():
    assert split_text("abcdef", max_chars=4, overlap=1) == ["abcd", "def"]
    with pytest.raises(ValueError, match="overlap"):
        split_text("abc", max_chars=4, overlap=4)


@dataclass
class FakeEmbedder:
    model_name: str = "fake-embedding-v1"

    def encode(self, texts: list[str]) -> list[list[float]]:
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append([1.0, 0.0] if "alpha" in lowered else [0.0, 1.0])
        return rows


class FakeStore:
    def __init__(self):
        self.records = []
        self.initialized = None

    def upsert(self, records, *, corpus_version, model_name):
        self.records = list(records)
        self.initialized = (corpus_version, model_name, len(self.records[0].embedding))
        return len({record.document.id for record in self.records})

    def search(self, query_embedding, *, corpus_version, model_name, limit):
        query = list(query_embedding)
        hits = []
        for record in self.records:
            score = sum(a * b for a, b in zip(query, record.embedding, strict=True))
            hits.append(
                RetrievalHit(
                    document=record.document,
                    score=score,
                    semantic_score=score,
                    retrieval_method="vector_pg",
                )
            )
        return sorted(hits, key=lambda hit: (-hit.score, hit.document.id))[:limit]

    def list_documents(self, *, corpus_version):
        return list({record.document.id: record.document for record in self.records}.values())


def _docs():
    return [
        MedicalDocument(id="doc-alpha", title="Alpha study", abstract="Alpha evidence"),
        MedicalDocument(id="doc-beta", title="Beta study", abstract="Beta evidence"),
    ]


def test_persistent_vector_index_and_search_are_injectable_offline():
    store = FakeStore()
    retriever = PersistentVectorRetriever(
        store,
        embedder=FakeEmbedder(),
        corpus_version="test-v1",
        chunk_size=100,
        chunk_overlap=0,
    )
    assert retriever.index_documents(_docs()) == 2
    hits = retriever.search("alpha question", top_k=1)
    assert hits[0].document.id == "doc-alpha"
    assert hits[0].retrieval_method == "vector_pg"
    assert store.initialized == ("test-v1", "fake-embedding-v1", 2)


def test_persistent_hybrid_keeps_new_method_name_and_keyword_score():
    store = FakeStore()
    vector = PersistentVectorRetriever(
        store,
        embedder=FakeEmbedder(),
        corpus_version="test-v1",
        chunk_size=100,
        chunk_overlap=0,
    )
    vector.index_documents(_docs())
    retriever = PersistentHybridRetriever(vector, documents=_docs(), semantic_weight=0.5)
    hits = retriever.search("alpha", top_k=2)
    assert hits[0].document.id == "doc-alpha"
    assert hits[0].retrieval_method == "hybrid_pg"
    assert hits[0].keyword_score == 1.0


def test_search_service_exposes_vector_pg_without_loading_real_database():
    class FakePersistent:
        def search(self, query: str, *, top_k: int):
            return [
                RetrievalHit(
                    document=MedicalDocument(
                        id="pubmed:1",
                        source="pubmed",
                        title="Persistent vector result",
                        abstract="A stored result",
                        url="https://pubmed.ncbi.nlm.nih.gov/1/",
                    ),
                    score=0.88,
                    retrieval_method="vector_pg",
                )
            ][:top_k]

    service = SearchService(persistent_vector_retriever=FakePersistent())
    results, total = service.search("stored", 1, "vector_pg")
    assert total == 1
    assert results[0].id == "pubmed:1"
    assert results[0].score == 0.88


def test_missing_database_configuration_is_explicit():
    with pytest.raises(PgVectorUnavailable, match="VECTOR_DATABASE_URL"):
        PgVectorStore()._connect()
