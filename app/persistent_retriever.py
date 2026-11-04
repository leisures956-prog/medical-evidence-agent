"""基于本地磁盘向量索引的向量和混合检索。"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from app.local_vector_store import LocalVectorStore, LocalVectorStoreError
from app.medical_models import MedicalDocument, RetrievalHit


class PersistentEmbeddingError(LocalVectorStoreError):
    """Embedding 模型缺失、加载失败或输出维度不正确。"""


def _as_matrix(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise PersistentEmbeddingError("Embedding 输出必须是非空二维数组。")
    if not np.isfinite(array).all():
        raise PersistentEmbeddingError("Embedding 输出包含 NaN 或无穷大。")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise PersistentEmbeddingError("Embedding 输出包含零向量。")
    return array / norms


class LocalSentenceTransformerEmbedder:
    """延迟加载本地 SentenceTransformer 模型。"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise PersistentEmbeddingError(
                "缺少 sentence-transformers，请安装 requirements-rag.txt。"
            ) from exc
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as exc:
            raise PersistentEmbeddingError(
                f"Embedding 模型加载失败：{self.model_name}。"
            ) from exc
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        try:
            encoded = self._load().encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except PersistentEmbeddingError:
            raise
        except Exception as exc:
            raise PersistentEmbeddingError("Embedding 编码失败。") from exc
        return _as_matrix(encoded)


class LocalPersistentVectorRetriever:
    """从 D 盘索引读取文档向量并执行余弦相似度检索。"""

    def __init__(
        self,
        store: LocalVectorStore,
        *,
        model_name: str | None = None,
        embedder: Any | None = None,
    ):
        self.store = store
        self.data = store.load()
        manifest_model = str(self.data.manifest["model_name"])
        if model_name and model_name != manifest_model:
            raise PersistentEmbeddingError(
                "查询模型与索引模型不一致："
                f"{model_name!r} != {manifest_model!r}。"
            )
        self.model_name = model_name or manifest_model
        self.embedder = embedder or LocalSentenceTransformerEmbedder(self.model_name)

    @classmethod
    def from_env(cls) -> "LocalPersistentVectorRetriever":
        project_root = Path(__file__).resolve().parents[1]
        root = Path(
            os.getenv(
                "LOCAL_VECTOR_INDEX_DIR",
                str(project_root / "artifacts" / "local_vector_index_v1"),
            )
        )
        return cls(
            LocalVectorStore(root),
            model_name=os.getenv("EMBEDDING_MODEL") or None,
        )

    def refresh(self) -> None:
        self.data = self.store.load()

    def documents(self) -> list[MedicalDocument]:
        return self.store.list_documents()

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        if not query.strip() or top_k < 1 or not self.data.records:
            return []
        query_vector = _as_matrix(self.embedder.encode([query]))[0]
        if query_vector.shape[0] != self.data.embeddings.shape[1]:
            raise PersistentEmbeddingError(
                "查询向量维度与索引不一致："
                f"{query_vector.shape[0]} != {self.data.embeddings.shape[1]}。"
            )
        scores = self.data.embeddings @ query_vector
        best_by_document: dict[str, tuple[float, MedicalDocument]] = {}
        for index in np.argsort(-scores):
            record = self.data.records[int(index)]
            score = float(scores[int(index)])
            previous = best_by_document.get(record.document.id)
            if previous is None or score > previous[0]:
                best_by_document[record.document.id] = (score, record.document)

        ranked = sorted(
            best_by_document.values(),
            key=lambda item: (-item[0], item[1].id),
        )
        return [
            RetrievalHit(
                document=document,
                score=score,
                semantic_score=score,
                retrieval_method="vector_persisted",
            )
            for score, document in ranked[:top_k]
        ]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", text.lower(), flags=re.UNICODE))


class LocalPersistentHybridRetriever:
    """在持久化向量候选上叠加关键词分数。"""

    def __init__(
        self,
        vector_retriever: LocalPersistentVectorRetriever,
        *,
        documents: list[MedicalDocument] | None = None,
        semantic_weight: float = 0.7,
    ):
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight 必须在 0 到 1 之间。")
        self.vector_retriever = vector_retriever
        self.documents = documents
        self.semantic_weight = semantic_weight

    def _documents(self) -> list[MedicalDocument]:
        if self.documents is None:
            self.documents = self.vector_retriever.documents()
        return self.documents

    @staticmethod
    def _keyword_score(query: str, document: MedicalDocument) -> float:
        query_terms = _tokens(query)
        if not query_terms:
            return 0.0
        document_terms = _tokens(document.retrieval_text())
        return len(query_terms & document_terms) / len(query_terms)

    def search(self, query: str, *, top_k: int = 5) -> list[RetrievalHit]:
        if not query.strip() or top_k < 1:
            return []
        documents = self._documents()
        semantic_hits = self.vector_retriever.search(query, top_k=max(top_k * 3, 10))
        semantic_by_id = {hit.document.id: hit for hit in semantic_hits}
        document_by_id = {document.id: document for document in documents}
        document_by_id.update({hit.document.id: hit.document for hit in semantic_hits})

        keyword_scores = {
            document.id: self._keyword_score(query, document)
            for document in documents
        }
        candidate_ids = set(semantic_by_id) | {
            document_id
            for document_id, score in keyword_scores.items()
            if score > 0
        }
        if not candidate_ids:
            return []

        max_keyword = max(keyword_scores.values(), default=1.0) or 1.0
        combined: list[RetrievalHit] = []
        for document_id in candidate_ids:
            semantic_hit = semantic_by_id.get(document_id)
            raw_semantic = (
                semantic_hit.semantic_score or semantic_hit.score
                if semantic_hit is not None
                else 0.0
            )
            normalized_semantic = (
                max(0.0, min(1.0, (raw_semantic + 1.0) / 2.0))
                if semantic_hit is not None
                else 0.0
            )
            keyword_score = keyword_scores.get(document_id, 0.0)
            normalized_keyword = keyword_score / max_keyword
            score = (
                self.semantic_weight * normalized_semantic
                + (1 - self.semantic_weight) * normalized_keyword
            )
            combined.append(
                RetrievalHit(
                    document=document_by_id[document_id],
                    score=score,
                    semantic_score=raw_semantic,
                    keyword_score=keyword_score,
                    retrieval_method="hybrid_persisted",
                )
            )

        combined.sort(key=lambda hit: (-hit.score, hit.document.id))
        return combined[:top_k]
