from __future__ import annotations

import os

from app.medical_models import MedicalDocument, RetrievalHit


class EmbeddingError(RuntimeError):
    """Raised when local embedding dependencies or the model are unavailable."""


class SemanticRetriever:
    """In-memory cosine-similarity retriever using a local SentenceTransformer."""

    def __init__(
        self,
        documents: list[MedicalDocument],
        model_name: str | None = None,
        embedder=None,
    ):
        self.documents = documents
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.embedder = embedder
        self._matrix = None

    def _load_embedder(self):
        if self.embedder is not None:
            return self.embedder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "请先安装 requirements-rag.txt 中的 Embedding 依赖。"
            ) from exc
        try:
            self.embedder = SentenceTransformer(self.model_name)
        except Exception as exc:
            raise EmbeddingError(f"Embedding 模型加载失败：{self.model_name}") from exc
        return self.embedder

    def build_index(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise EmbeddingError("缺少 numpy，请先安装 Embedding 依赖。") from exc

        model = self._load_embedder()
        texts = [document.retrieval_text() for document in self.documents]
        self._matrix = np.asarray(
            model.encode(texts, convert_to_numpy=True, normalize_embeddings=True),
            dtype=np.float32,
        )

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        if not query.strip() or not self.documents:
            return []
        if self._matrix is None:
            self.build_index()

        try:
            import numpy as np
        except ImportError as exc:
            raise EmbeddingError("缺少 numpy，请先安装 Embedding 依赖。") from exc

        model = self._load_embedder()
        query_vector = model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        scores = self._matrix @ query_vector
        indices = np.argsort(-scores)[:top_k]

        return [
            RetrievalHit(
                document=self.documents[int(index)],
                score=float(scores[index]),
                semantic_score=float(scores[index]),
                retrieval_method="semantic",
            )
            for index in indices
        ]
