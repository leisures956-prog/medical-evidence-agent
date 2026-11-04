"""Opt-in PostgreSQL + pgvector checks.

Run with ``$env:PGVECTOR_DATABASE_URL = ...`` and
``pytest -m integration tests/integration/test_pgvector_integration.py``.
The default offline suite never requires a database.
"""

from __future__ import annotations

import os
import uuid

import pytest

from app.medical_models import MedicalDocument
from app.pgvector_store import PgVectorStore, VectorRecord, content_sha256


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.getenv("PGVECTOR_DATABASE_URL"),
    reason="设置 PGVECTOR_DATABASE_URL 才运行 pgvector 集成测试",
)
def test_pgvector_round_trip():
    pytest.importorskip("psycopg")
    corpus_version = f"test-pgvector-{uuid.uuid4().hex[:12]}"
    store = PgVectorStore(os.environ["PGVECTOR_DATABASE_URL"])
    document = MedicalDocument(
        id="pubmed:test-pgvector",
        source="pubmed",
        title="Persistent vector test",
        abstract="A deterministic database round trip.",
        url="https://pubmed.ncbi.nlm.nih.gov/test-pgvector/",
        metadata={"pmid": "test-pgvector"},
    )
    record = VectorRecord(
        document=document,
        chunk_id=f"{corpus_version}:{document.id}:0",
        chunk_index=0,
        chunk_text=document.retrieval_text(),
        embedding=(1.0, 0.0),
        content_sha256=content_sha256(document.retrieval_text()),
    )
    try:
        assert store.upsert(
            [record],
            corpus_version=corpus_version,
            model_name="test-model",
        ) == 1
        hits = store.search(
            [1.0, 0.0],
            corpus_version=corpus_version,
            model_name="test-model",
            limit=1,
        )
        assert hits
        assert hits[0].document.id == document.id
        assert hits[0].score == pytest.approx(1.0)
    finally:
        store.delete_corpus(corpus_version)
