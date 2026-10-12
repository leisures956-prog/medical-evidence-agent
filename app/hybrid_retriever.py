from __future__ import annotations

import re

from app.medical_models import MedicalDocument, RetrievalHit
from app.semantic_retriever import SemanticRetriever


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w]+", text.lower(), flags=re.UNICODE))


class HybridRetriever:
    """Combine lexical overlap and semantic cosine similarity."""

    def __init__(
        self,
        documents: list[MedicalDocument],
        semantic_retriever: SemanticRetriever,
        semantic_weight: float = 0.7,
    ):
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight 必须在 0 到 1 之间。")
        self.documents = documents
        self.semantic_retriever = semantic_retriever
        self.semantic_weight = semantic_weight

    @staticmethod
    def _keyword_score(query: str, document: MedicalDocument) -> float:
        query_terms = _tokens(query)
        if not query_terms:
            return 0.0
        document_terms = _tokens(document.retrieval_text())
        return len(query_terms & document_terms) / len(query_terms)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalHit]:
        semantic_hits = self.semantic_retriever.search(query, top_k=max(top_k * 3, 10))
        semantic_by_id = {hit.document.id: hit for hit in semantic_hits}

        keyword_scores = {
            document.id: self._keyword_score(query, document)
            for document in self.documents
        }
        candidate_ids = set(semantic_by_id) | {
            document_id for document_id, score in keyword_scores.items() if score > 0
        }
        if not candidate_ids:
            return []

        max_keyword = max(keyword_scores.values(), default=1.0) or 1.0
        document_by_id = {document.id: document for document in self.documents}
        combined: list[RetrievalHit] = []

        for document_id in candidate_ids:
            semantic_score = semantic_by_id.get(document_id)
            raw_semantic = (semantic_score.semantic_score or 0.0) if semantic_score else 0.0
            normalized_semantic = max(0.0, min(1.0, (raw_semantic + 1.0) / 2.0)) if semantic_score else 0.0
            normalized_keyword = keyword_scores[document_id] / max_keyword
            score = (
                self.semantic_weight * normalized_semantic
                + (1 - self.semantic_weight) * normalized_keyword
            )
            combined.append(
                RetrievalHit(
                    document=document_by_id[document_id],
                    score=score,
                    semantic_score=raw_semantic,
                    keyword_score=keyword_scores[document_id],
                    retrieval_method="hybrid",
                )
            )

        combined.sort(key=lambda hit: (-hit.score, hit.document.id))
        return combined[:top_k]
