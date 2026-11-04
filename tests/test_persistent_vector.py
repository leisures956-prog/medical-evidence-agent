from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.local_vector_store import LocalVectorStore, chunk_documents
from app.main import app, get_search_service
from app.medical_models import MedicalDocument, RetrievalHit
from app.persistent_retriever import (
    LocalPersistentHybridRetriever,
    LocalPersistentVectorRetriever,
)
from app.search import SearchService


pytestmark = pytest.mark.api_runtime


class FakeEmbedder:
    model_name = "fake-model"

    def encode(self, texts):
        rows = []
        for text in texts:
            value = text.lower()
            rows.append(
                [
                    1.0 if "python" in value else 0.0,
                    1.0 if "fastapi" in value else 0.0,
                    1.0 if "切片" in value else 0.0,
                ]
            )
        return np.asarray(rows, dtype=np.float32)


def _documents():
    return [
        MedicalDocument(
            id="doc-1",
            source="local",
            title="Python 列表",
            abstract="Python 数据结构",
        ),
        MedicalDocument(
            id="doc-2",
            source="local",
            title="Python 切片",
            abstract="Python 切片规则",
        ),
        MedicalDocument(
            id="doc-3",
            source="local",
            title="FastAPI",
            abstract="现代 Python Web API",
        ),
    ]


def _build_store(tmp_path: Path):
    documents = _documents()
    records = chunk_documents(
        documents,
        corpus_version="test-v1",
        chunk_size=1200,
        chunk_overlap=120,
    )
    embeddings = FakeEmbedder().encode([record.chunk_text for record in records])
    store = LocalVectorStore(tmp_path / "index")
    store.rebuild(
        records,
        embeddings,
        corpus_version="test-v1",
        model_name="fake-model",
        source_path="test",
    )
    return store, documents


def test_local_store_rebuild_and_reload(tmp_path):
    store, documents = _build_store(tmp_path)

    data = store.load()

    assert data.manifest["corpus_version"] == "test-v1"
    assert data.manifest["model_name"] == "fake-model"
    assert data.manifest["document_count"] == len(documents)
    assert data.embeddings.shape[0] == len(data.records)
    assert len(store.list_documents()) == 3


def test_local_store_upsert_and_delete(tmp_path):
    store, _ = _build_store(tmp_path)
    replacement = MedicalDocument(
        id="doc-2",
        source="local",
        title="Python 切片更新",
        abstract="更新后的切片说明",
    )
    records = chunk_documents(
        [replacement], corpus_version="test-v1", chunk_size=1200, chunk_overlap=120
    )
    store.upsert(records, FakeEmbedder().encode([record.chunk_text for record in records]))

    assert store.list_documents()[1].title == "Python 切片更新"
    assert store.delete_documents(["doc-1"]) == 1
    assert [document.id for document in store.list_documents()] == ["doc-2", "doc-3"]


def test_persistent_vector_and_hybrid_retrieval(tmp_path):
    store, _ = _build_store(tmp_path)
    vector = LocalPersistentVectorRetriever(store, embedder=FakeEmbedder())
    hybrid = LocalPersistentHybridRetriever(vector)

    vector_hits = vector.search("FastAPI", top_k=1)
    hybrid_hits = hybrid.search("Python 切片", top_k=1)

    assert vector_hits[0].document.id == "doc-3"
    assert vector_hits[0].retrieval_method == "vector_persisted"
    assert hybrid_hits[0].document.id == "doc-2"
    assert hybrid_hits[0].retrieval_method == "hybrid_persisted"


class FakePersistentRetriever:
    def __init__(self, method):
        self.method = method

    def search(self, query, *, top_k):
        document = _documents()[0]
        return [
            RetrievalHit(
                document=document,
                score=0.9,
                retrieval_method=self.method,
            )
        ][:top_k]


def test_api_accepts_persisted_search_modes():
    service = SearchService(
        local_persistent_vector_retriever=FakePersistentRetriever(
            "vector_persisted"
        ),
        local_persistent_hybrid_retriever=FakePersistentRetriever(
            "hybrid_persisted"
        ),
    )
    client = TestClient(app)
    app.dependency_overrides[get_search_service] = lambda: service
    try:
        vector_response = client.post(
            "/search",
            json={"query": "Python", "source": "vector_persisted"},
        )
        hybrid_response = client.post(
            "/search",
            json={"query": "Python", "source": "hybrid_persisted"},
        )
    finally:
        app.dependency_overrides.clear()

    assert vector_response.status_code == 200
    assert hybrid_response.status_code == 200
    assert vector_response.json()["source"] == "vector_persisted"
    assert hybrid_response.json()["source"] == "hybrid_persisted"
