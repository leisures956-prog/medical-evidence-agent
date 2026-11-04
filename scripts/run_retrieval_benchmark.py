"""Compare retrieval methods on a frozen snapshot.

This benchmark deliberately evaluates all methods on the same frozen candidate
pool and the same relevance labels.  It does not query PubMed, so the result is
reproducible once the snapshot, labels and embedding model are fixed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Literal

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.corpus_snapshot import (
    BenchmarkQuestion,
    FrozenDocument,
    LabelsFile,
    SnapshotBundle,
    SnapshotError,
    load_snapshot,
)
from app.cross_encoder_reranker import (
    CrossEncoderReranker,
    CrossEncoderRerankerError,
)
from app.hybrid_retriever import HybridRetriever
from app.keyword_baseline import rank_documents
from app.local_vector_store import LocalVectorStore, LocalVectorStoreError
from app.medical_models import MedicalDocument, RetrievalHit
from app.metadata_filter import RetrievalFilters
from app.multi_query_retriever import MultiQueryRetriever
from app.persistent_retriever import (
    LocalPersistentVectorRetriever,
    PersistentEmbeddingError,
)
from app.quality_reranker import QualityReranker, quality_flags
from app.semantic_retriever import EmbeddingError, SemanticRetriever


MethodName = Literal[
    "keyword",
    "vector",
    "hybrid",
    "vector_persisted",
    "hybrid_persisted",
    "hybrid_persisted_reranked",
    "hybrid_persisted_cross_reranked",
    "hybrid_persisted_cross_quality_reranked",
    "advanced_persisted",
    "advanced_persisted_filtered",
]
CandidateScope = Literal["question_pool", "global_corpus"]
DEFAULT_METHODS: tuple[MethodName, ...] = ("keyword", "vector", "hybrid")
METHOD_CHOICES: tuple[MethodName, ...] = (
    "keyword",
    "vector",
    "hybrid",
    "vector_persisted",
    "hybrid_persisted",
    "hybrid_persisted_reranked",
    "hybrid_persisted_cross_reranked",
    "hybrid_persisted_cross_quality_reranked",
    "advanced_persisted",
    "advanced_persisted_filtered",
)
DEFAULT_TOP_KS: tuple[int, ...] = (1, 3, 5)


def _to_medical_document(document: FrozenDocument) -> MedicalDocument:
    """Convert a frozen document to the common retrieval document model."""

    return MedicalDocument(
        id=f"pubmed:{document.pmid}",
        source="pubmed",
        title=document.title,
        abstract=document.abstract,
        authors=list(document.authors),
        year=document.year,
        doi=document.doi,
        journal=document.journal,
        url=document.source_url,
        metadata={
            "pmid": document.pmid,
            "corpus_version": document.corpus_version,
        },
    )


def _pmid_from_hit(hit: RetrievalHit) -> str:
    value = hit.document.metadata.get("pmid")
    if value:
        return str(value)
    if hit.document.id.startswith("pubmed:"):
        return hit.document.id.split(":", 1)[1]
    return hit.document.id


def _ranked_hit_rows(hits: Iterable[RetrievalHit], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, hit in enumerate(list(hits)[:top_k], start=1):
        rows.append(
            {
                "rank": rank,
                "pmid": _pmid_from_hit(hit),
                "score": round(float(hit.score), 8),
                "semantic_score": (
                    round(float(hit.semantic_score), 8)
                    if hit.semantic_score is not None
                    else None
                ),
                "keyword_score": (
                    round(float(hit.keyword_score), 8)
                    if hit.keyword_score is not None
                    else None
                ),
                "quality_flags": list(quality_flags(hit.document)),
            }
        )
    return rows


def _keyword_rows(
    question: str,
    documents: list[FrozenDocument],
    top_k: int,
) -> list[dict[str, Any]]:
    rows = rank_documents(question, documents, top_k=top_k)
    documents_by_pmid = {document.pmid: document for document in documents}
    output: list[dict[str, Any]] = []
    for row in rows:
        frozen = documents_by_pmid[str(row["pmid"])]
        output.append(
            {
                **row,
                "semantic_score": None,
                "keyword_score": float(row["score"]),
                "quality_flags": list(
                    quality_flags(_to_medical_document(frozen))
                ),
            }
        )
    return output


def _quality_at_k(rows: list[dict[str, Any]], k: int) -> dict[str, float]:
    """Summarise visible document-quality markers in the returned top-k."""

    selected = rows[:k]
    denominator = len(selected)
    if not denominator:
        return {
            "abstract_missing_rate": 0.0,
            "editorial_rate": 0.0,
            "correction_rate": 0.0,
            "review_rate": 0.0,
            "any_quality_flag_rate": 0.0,
        }

    def rate(flag: str) -> float:
        return sum(flag in row.get("quality_flags", []) for row in selected) / denominator

    return {
        "abstract_missing_rate": rate("abstract_missing"),
        "editorial_rate": rate("editorial"),
        "correction_rate": rate("correction"),
        "review_rate": rate("review"),
        "any_quality_flag_rate": sum(
            bool(row.get("quality_flags")) for row in selected
        ) / denominator,
    }


def _dcg(relevances: list[int]) -> float:
    import math

    return sum(
        (2**relevance - 1) / math.log2(index + 2)
        for index, relevance in enumerate(relevances)
    )


def _ndcg_at_k(
    ranked_pmids: list[str],
    relevance_by_pmid: dict[str, int],
    k: int,
) -> float:
    actual = [relevance_by_pmid.get(pmid, 0) for pmid in ranked_pmids[:k]]
    ideal = sorted(relevance_by_pmid.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    return _dcg(actual) / ideal_dcg if ideal_dcg else 0.0


def _case_metrics(
    ranked_pmids: list[str],
    relevance_by_pmid: dict[str, int],
    top_ks: tuple[int, ...],
) -> dict[str, Any]:
    relevant_count = sum(value > 0 for value in relevance_by_pmid.values())
    recall: dict[str, float | None] = {}
    ndcg: dict[str, float] = {}

    for k in top_ks:
        top_ids = ranked_pmids[:k]
        if relevant_count:
            retrieved_relevant = sum(
                (relevance_by_pmid.get(pmid) or 0) > 0 for pmid in top_ids
            )
            recall[f"@{k}"] = retrieved_relevant / relevant_count
        else:
            recall[f"@{k}"] = None
        ndcg[f"@{k}"] = _ndcg_at_k(ranked_pmids, relevance_by_pmid, k)

    mrr = 0.0
    for rank, pmid in enumerate(ranked_pmids, start=1):
        if (relevance_by_pmid.get(pmid) or 0) > 0:
            mrr = 1.0 / rank
            break

    return {
        "recall": recall,
        "ndcg": ndcg,
        "mrr": mrr,
        "has_relevant_label": relevant_count > 0,
    }


def _average(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        value = ordered[middle]
    else:
        value = (ordered[middle - 1] + ordered[middle]) / 2
    return round(value, 6)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    import math

    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return round(ordered[min(rank - 1, len(ordered) - 1)], 6)


def _candidate_documents(
    bundle: SnapshotBundle,
    question: BenchmarkQuestion,
    candidate_scope: CandidateScope,
) -> list[FrozenDocument]:
    if candidate_scope == "global_corpus":
        return list(bundle.documents)
    return [
        document
        for document in bundle.documents
        if question.id in document.candidate_question_ids
    ]


def _labels_by_question(bundle: SnapshotBundle) -> dict[str, dict[str, int | None]]:
    labels: dict[str, dict[str, int | None]] = {}
    for label in bundle.labels.labels:
        labels.setdefault(label.question_id, {})[label.pmid] = label.relevance
    return labels


def _load_embedding_model(
    model_name: str | None,
    embedder: object | None,
) -> tuple[object | None, float]:
    """Load the model once, outside the per-query latency measurements."""

    if embedder is not None:
        return embedder, 0.0
    probe = SemanticRetriever([], model_name=model_name)
    started = time.perf_counter()
    loaded = probe._load_embedder()
    return loaded, (time.perf_counter() - started) * 1000


def _prepare_semantic_retriever(
    documents: list[FrozenDocument],
    *,
    model_name: str | None,
    embedder: object | None,
) -> tuple[SemanticRetriever, float]:
    """Build one shared candidate index for vector and hybrid methods."""

    medical_documents = [_to_medical_document(document) for document in documents]
    semantic = SemanticRetriever(
        medical_documents,
        model_name=model_name,
        embedder=embedder,
    )
    if not medical_documents:
        return semantic, 0.0
    started = time.perf_counter()
    semantic.build_index()
    return semantic, (time.perf_counter() - started) * 1000


def _prepare_persistent_retriever(
    index_dir: Path | str | None,
    *,
    model_name: str | None,
    embedder: object | None,
) -> tuple[LocalPersistentVectorRetriever, float]:
    """Load one prebuilt local index outside query latency measurements."""

    project_root = Path(__file__).resolve().parents[1]
    root = Path(
        index_dir
        or project_root / "artifacts" / "local_vector_index_v1"
    )
    started = time.perf_counter()
    retriever = LocalPersistentVectorRetriever(
        LocalVectorStore(root),
        model_name=model_name,
        embedder=embedder,
    )
    return retriever, (time.perf_counter() - started) * 1000


def _persistent_rank_hits(
    question: str,
    documents: list[FrozenDocument],
    method: MethodName,
    top_k: int,
    retriever: LocalPersistentVectorRetriever,
    cross_encoder_reranker: CrossEncoderReranker | None = None,
) -> list[RetrievalHit]:
    """Rank only the current benchmark candidate pool using a disk index.

    The local index contains the complete frozen corpus.  We retrieve all
    document-level semantic scores once, then restrict and re-score them using
    the same candidate pool as the keyword/vector/hybrid baselines.  This keeps
    the benchmark comparable without changing the production retriever API.
    """

    candidate_documents = [_to_medical_document(document) for document in documents]
    candidate_by_pmid = {
        str(document.metadata.get("pmid") or document.id): document
        for document in candidate_documents
    }
    if not candidate_by_pmid:
        return []

    # Asking for all unique documents ensures that a candidate which is not in
    # the global top-k is still available for the candidate-pool comparison.
    semantic_hits = retriever.search(
        question,
        top_k=max(len(retriever.data.records), len(candidate_by_pmid), 1),
    )
    semantic_by_pmid = {
        _pmid_from_hit(hit): hit
        for hit in semantic_hits
        if _pmid_from_hit(hit) in candidate_by_pmid
    }

    if method == "vector_persisted":
        return sorted(
            semantic_by_pmid.values(),
            key=lambda hit: (-hit.score, _pmid_from_hit(hit)),
        )[:top_k]

    keyword_scores = {
        pmid: HybridRetriever._keyword_score(question, document)
        for pmid, document in candidate_by_pmid.items()
    }
    max_keyword = max(keyword_scores.values(), default=1.0) or 1.0
    combined: list[RetrievalHit] = []
    for pmid, document in candidate_by_pmid.items():
        semantic_hit = semantic_by_pmid.get(pmid)
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
        keyword_score = keyword_scores[pmid]
        score = 0.7 * normalized_semantic + 0.3 * (keyword_score / max_keyword)
        combined.append(
            RetrievalHit(
                document=document,
                score=score,
                semantic_score=raw_semantic,
                keyword_score=keyword_score,
                retrieval_method="hybrid_persisted",
            )
        )
    combined.sort(key=lambda hit: (-hit.score, _pmid_from_hit(hit)))
    if method == "hybrid_persisted_reranked":
        return QualityReranker().rerank(combined, top_k=top_k)
    if method == "hybrid_persisted_cross_quality_reranked":
        if cross_encoder_reranker is None:
            raise ValueError("Cross-Encoder 重排方法需要预先加载模型。")
        return cross_encoder_reranker.rerank_with_quality(
            question,
            combined,
            top_k=top_k,
        )
    if method == "hybrid_persisted_cross_reranked":
        if cross_encoder_reranker is None:
            raise ValueError("Cross-Encoder 方法需要预先加载重排模型。")
        return cross_encoder_reranker.rerank(
            question,
            combined,
            top_k=top_k,
        )
    return combined[:top_k]


def _rank_one(
    question: BenchmarkQuestion,
    documents: list[FrozenDocument],
    method: MethodName,
    top_k: int,
    *,
    semantic_retriever: SemanticRetriever | None,
    persistent_retriever: LocalPersistentVectorRetriever | None,
    cross_encoder_reranker: CrossEncoderReranker | None,
    retrieval_filters: RetrievalFilters | None,
) -> tuple[list[dict[str, Any]], float, dict[str, Any] | None]:
    """Rank one query while excluding model/index construction from latency."""

    started = time.perf_counter()

    if method == "keyword":
        rows = _keyword_rows(question.question, documents, top_k)
        return rows, (time.perf_counter() - started) * 1000, None

    if method in {"advanced_persisted", "advanced_persisted_filtered"}:
        if persistent_retriever is None or cross_encoder_reranker is None:
            raise ValueError("advanced_persisted 需要持久化索引和 Cross-Encoder。")

        class CandidatePoolHybridRetriever:
            def search(self, query: str, *, top_k: int = 5):
                return _persistent_rank_hits(
                    query,
                    documents,
                    "hybrid_persisted",
                    top_k,
                    persistent_retriever,
                )

        advanced = MultiQueryRetriever(
            CandidatePoolHybridRetriever(),
            cross_encoder_reranker,
        )
        result = advanced.search_with_diagnostics(
            question.question,
            top_k=top_k,
            candidate_limit=max(len(documents), top_k),
            filters=(retrieval_filters if method == "advanced_persisted_filtered" else None),
        )
        rows = _ranked_hit_rows(result.hits, top_k)
        diagnostics = {
            "queries": list(result.diagnostics.queries),
            "query_variant_count": len(result.diagnostics.queries),
            "raw_hit_count": result.diagnostics.raw_hit_count,
            "unique_hit_count": result.diagnostics.unique_hit_count,
            "duplicates_removed": result.diagnostics.duplicates_removed,
            "eligible_hit_count": result.diagnostics.eligible_hit_count,
            "filtered_out_count": result.diagnostics.filtered_out_count,
            "excluded_by_year": result.diagnostics.excluded_by_year,
            "excluded_by_abstract": result.diagnostics.excluded_by_abstract,
            "excluded_by_article_type": (
                result.diagnostics.excluded_by_article_type
            ),
        }
        return rows, (time.perf_counter() - started) * 1000, diagnostics

    if method in {
        "vector_persisted",
        "hybrid_persisted",
        "hybrid_persisted_reranked",
        "hybrid_persisted_cross_reranked",
        "hybrid_persisted_cross_quality_reranked",
    }:
        if persistent_retriever is None:
            raise ValueError("持久化向量方法需要预先加载本地索引。")
        hits = _persistent_rank_hits(
            question.question,
            documents,
            method,
            top_k,
            persistent_retriever,
            cross_encoder_reranker,
        )
        rows = _ranked_hit_rows(hits, top_k)
        return rows, (time.perf_counter() - started) * 1000, None

    if not documents:
        return [], (time.perf_counter() - started) * 1000, None
    if semantic_retriever is None:
        raise ValueError("vector/hybrid 方法需要预先建立 SemanticRetriever。")

    if method == "vector":
        hits = semantic_retriever.search(question.question, top_k=top_k)
    else:
        hits = HybridRetriever(
            semantic_retriever.documents,
            semantic_retriever,
        ).search(question.question, top_k=top_k)
    rows = _ranked_hit_rows(hits, top_k)
    return rows, (time.perf_counter() - started) * 1000, None


def _validate_methods(methods: Iterable[str]) -> tuple[MethodName, ...]:
    selected = tuple(methods)
    if not selected:
        raise ValueError("至少选择一种检索方法。")
    invalid = sorted(set(selected) - set(METHOD_CHOICES))
    if invalid:
        raise ValueError(f"不支持的检索方法：{invalid}")
    if len(set(selected)) != len(selected):
        raise ValueError("检索方法不能重复。")
    return selected  # type: ignore[return-value]


def evaluate_retrieval_snapshot(
    bundle: SnapshotBundle,
    *,
    methods: Iterable[MethodName] = DEFAULT_METHODS,
    top_ks: tuple[int, ...] = DEFAULT_TOP_KS,
    candidate_scope: CandidateScope = "question_pool",
    model_name: str | None = None,
    embedder: object | None = None,
    label_source: str | None = None,
    persistent_index_dir: Path | str | None = None,
    cross_encoder_model_name: str | None = None,
    cross_encoder_reranker: CrossEncoderReranker | None = None,
    retrieval_filters: RetrievalFilters | None = None,
) -> dict[str, Any]:
    """Evaluate selected methods under identical corpus and label conditions."""

    selected_methods = _validate_methods(methods)
    if not top_ks or any(k < 1 for k in top_ks):
        raise ValueError("top_ks 必须包含正整数。")
    top_ks = tuple(sorted(set(top_ks)))
    top_k = max(top_ks)
    labels_by_question = _labels_by_question(bundle)
    needs_embedding = any(
        method in {
            "vector",
            "hybrid",
            "vector_persisted",
            "hybrid_persisted",
            "hybrid_persisted_reranked",
            "hybrid_persisted_cross_reranked",
            "hybrid_persisted_cross_quality_reranked",
            "advanced_persisted",
            "advanced_persisted_filtered",
        }
        for method in selected_methods
    )
    shared_embedder, embedding_model_load_ms = _load_embedding_model(
        model_name,
        embedder,
    ) if needs_embedding else (None, 0.0)
    needs_persistent = any(
        method in {
            "vector_persisted",
            "hybrid_persisted",
            "hybrid_persisted_reranked",
            "hybrid_persisted_cross_reranked",
            "hybrid_persisted_cross_quality_reranked",
            "advanced_persisted",
            "advanced_persisted_filtered",
        }
        for method in selected_methods
    )
    persistent_retriever: LocalPersistentVectorRetriever | None = None
    persistent_index_load_ms = 0.0
    if needs_persistent:
        persistent_retriever, persistent_index_load_ms = _prepare_persistent_retriever(
            persistent_index_dir,
            model_name=None,
            embedder=shared_embedder,
        )

    needs_cross_encoder = any(
        method in {
            "hybrid_persisted_cross_reranked",
            "hybrid_persisted_cross_quality_reranked",
            "advanced_persisted",
            "advanced_persisted_filtered",
        }
        for method in selected_methods
    )
    cross_encoder_model_load_ms = 0.0
    active_cross_encoder = cross_encoder_reranker
    if needs_cross_encoder:
        active_cross_encoder = active_cross_encoder or CrossEncoderReranker(
            model_name=cross_encoder_model_name
        )
        started = time.perf_counter()
        active_cross_encoder.load_model()
        cross_encoder_model_load_ms = (time.perf_counter() - started) * 1000

    active_filters = retrieval_filters or RetrievalFilters(
        require_abstract=True,
        exclude_article_types=("editorial", "correction"),
    )

    cases_by_method: dict[str, list[dict[str, Any]]] = {
        method: [] for method in selected_methods
    }

    for question in bundle.questions:
        candidates = _candidate_documents(bundle, question, candidate_scope)
        candidate_ids = {document.pmid for document in candidates}
        relevance_by_pmid = labels_by_question.get(question.id, {})
        missing_pmids = sorted(
            pmid for pmid in candidate_ids if relevance_by_pmid.get(pmid) is None
        )
        document_ids = {document.pmid for document in bundle.documents}
        unknown_label_pmids = sorted(
            pmid for pmid in relevance_by_pmid if pmid not in document_ids
        )
        status = (
            "complete"
            if not missing_pmids and not unknown_label_pmids
            else "incomplete_labels"
        )

        semantic_retriever: SemanticRetriever | None = None
        index_build_ms = 0.0
        if needs_embedding:
            semantic_retriever, index_build_ms = _prepare_semantic_retriever(
                candidates,
                model_name=model_name,
                embedder=shared_embedder,
            )

        for method in selected_methods:
            ranked, query_latency_ms, retrieval_diagnostics = _rank_one(
                question,
                candidates,
                method,
                top_k,
                semantic_retriever=semantic_retriever,
                persistent_retriever=persistent_retriever,
                cross_encoder_reranker=active_cross_encoder,
                retrieval_filters=active_filters,
            )
            ranked_pmids = [str(item["pmid"]) for item in ranked]
            case: dict[str, Any] = {
                "question_id": question.id,
                "question": question.question,
                "candidate_count": len(candidates),
                "label_count": len(relevance_by_pmid),
                "missing_label_pmids": missing_pmids,
                "unknown_label_pmids": unknown_label_pmids,
                "status": status,
                # Kept for compatibility; this now means warm query latency.
                "latency_ms": round(query_latency_ms, 3),
                "query_latency_ms": round(query_latency_ms, 3),
                "index_build_ms": round(
                    index_build_ms if method in {"vector", "hybrid"} else 0.0,
                    3,
                ),
                "ranked": ranked,
                "retrieval_diagnostics": retrieval_diagnostics,
                "quality_at_k": {
                    f"@{k}": _quality_at_k(ranked, k) for k in top_ks
                },
            }
            if status == "complete":
                case["metrics"] = _case_metrics(
                    ranked_pmids,
                    {pmid: int(value or 0) for pmid, value in relevance_by_pmid.items()},
                    top_ks,
                )
            else:
                case["metrics"] = None
            cases_by_method[method].append(case)

    method_results: dict[str, dict[str, Any]] = {}
    for method in selected_methods:
        cases = cases_by_method[method]
        completed_cases = [case for case in cases if case["status"] == "complete"]
        recall: dict[str, float | None] = {}
        ndcg: dict[str, float | None] = {}
        for k in top_ks:
            recall_values = [
                float(case["metrics"]["recall"][f"@{k}"])
                for case in completed_cases
                if case["metrics"]["recall"][f"@{k}"] is not None
            ]
            ndcg_values = [
                float(case["metrics"]["ndcg"][f"@{k}"])
                for case in completed_cases
            ]
            recall[f"@{k}"] = _average(recall_values)
            ndcg[f"@{k}"] = _average(ndcg_values)

        query_latencies = [float(case["query_latency_ms"]) for case in cases]
        index_latencies = [float(case["index_build_ms"]) for case in cases]
        query_average = _average(query_latencies)
        diagnostics_rows = [
            case["retrieval_diagnostics"]
            for case in cases
            if case["retrieval_diagnostics"] is not None
        ]
        quality_metrics: dict[str, dict[str, float | None]] = {}
        for k in top_ks:
            quality_metrics[f"@{k}"] = {
                name: _average(
                    [
                        float(case["quality_at_k"][f"@{k}"][name])
                        for case in cases
                    ]
                )
                for name in (
                    "abstract_missing_rate",
                    "editorial_rate",
                    "correction_rate",
                    "review_rate",
                    "any_quality_flag_rate",
                )
            }
        method_results[method] = {
            "method": method,
            "total_cases": len(bundle.questions),
            "completed_cases": len(completed_cases),
            "incomplete_cases": len(bundle.questions) - len(completed_cases),
            "metrics": {
                "recall": recall,
                "ndcg": ndcg,
                "mrr": _average(
                    [float(case["metrics"]["mrr"]) for case in completed_cases]
                ),
                "no_result_rate": (
                    sum(1 for case in cases if not case["ranked"])
                    / len(cases)
                    if cases
                    else None
                ),
                "average_latency_ms": query_average,
                "average_query_latency_ms": query_average,
                "median_query_latency_ms": _median(query_latencies),
                "p95_query_latency_ms": _percentile(query_latencies, 0.95),
                "average_index_build_ms": _average(index_latencies),
                "average_query_variant_count": _average(
                    [float(item["query_variant_count"]) for item in diagnostics_rows]
                ),
                "average_duplicates_removed": _average(
                    [float(item["duplicates_removed"]) for item in diagnostics_rows]
                ),
                "average_unique_hit_count": _average(
                    [float(item["unique_hit_count"]) for item in diagnostics_rows]
                ),
                "average_filtered_out_count": _average(
                    [float(item["filtered_out_count"]) for item in diagnostics_rows]
                ),
                "average_excluded_by_year": _average(
                    [float(item["excluded_by_year"]) for item in diagnostics_rows]
                ),
                "average_excluded_by_abstract": _average(
                    [float(item["excluded_by_abstract"]) for item in diagnostics_rows]
                ),
                "average_excluded_by_article_type": _average(
                    [
                        float(item["excluded_by_article_type"])
                        for item in diagnostics_rows
                    ]
                ),
                "quality_at_k": quality_metrics,
            },
            "cases": cases,
        }

    comparison: dict[str, dict[str, float | None]] = {}
    for metric in ("recall", "ndcg"):
        for k in top_ks:
            key = f"{metric}@{k}"
            comparison[key] = {
                method: method_results[method]["metrics"][metric][f"@{k}"]
                for method in selected_methods
            }
    comparison["mrr"] = {
        method: method_results[method]["metrics"]["mrr"]
        for method in selected_methods
    }
    comparison["no_result_rate"] = {
        method: method_results[method]["metrics"]["no_result_rate"]
        for method in selected_methods
    }
    comparison["average_latency_ms"] = {
        method: method_results[method]["metrics"]["average_latency_ms"]
        for method in selected_methods
    }
    comparison["average_query_latency_ms"] = {
        method: method_results[method]["metrics"]["average_query_latency_ms"]
        for method in selected_methods
    }
    comparison["median_query_latency_ms"] = {
        method: method_results[method]["metrics"]["median_query_latency_ms"]
        for method in selected_methods
    }
    comparison["p95_query_latency_ms"] = {
        method: method_results[method]["metrics"]["p95_query_latency_ms"]
        for method in selected_methods
    }
    comparison["average_index_build_ms"] = {
        method: method_results[method]["metrics"]["average_index_build_ms"]
        for method in selected_methods
    }
    for diagnostic_name in (
        "average_query_variant_count",
        "average_duplicates_removed",
        "average_unique_hit_count",
        "average_filtered_out_count",
        "average_excluded_by_year",
        "average_excluded_by_abstract",
        "average_excluded_by_article_type",
    ):
        comparison[diagnostic_name] = {
            method: method_results[method]["metrics"][diagnostic_name]
            for method in selected_methods
        }
    for k in top_ks:
        for quality_name in (
            "abstract_missing_rate",
            "editorial_rate",
            "correction_rate",
            "review_rate",
            "any_quality_flag_rate",
        ):
            comparison[f"{quality_name}@{k}"] = {
                method: method_results[method]["metrics"]["quality_at_k"][f"@{k}"][quality_name]
                for method in selected_methods
            }

    return {
        "benchmark": "retrieval-benchmark-v1",
        "corpus_version": bundle.manifest.corpus_version,
        "annotation_version": bundle.labels.annotation_version,
        "coverage_scope": bundle.manifest.coverage_scope,
        "candidate_scope": candidate_scope,
        "label_source": label_source,
        "metric_scopes": {
            "relevance": "complete_labeled_cases_only",
            "quality": "all_cases",
            "latency": "all_cases",
        },
        "filter_config": {
            "year_from": active_filters.year_from,
            "year_to": active_filters.year_to,
            "require_abstract": active_filters.require_abstract,
            "exclude_article_types": list(active_filters.exclude_article_types),
        },
        "timing": {
            "embedding_model_load_ms": round(embedding_model_load_ms, 3),
            "persistent_index_load_ms": round(persistent_index_load_ms, 3),
            "cross_encoder_model_load_ms": round(
                cross_encoder_model_load_ms, 3
            ),
            "query_latency_excludes_model_load": True,
            "query_latency_excludes_index_build": True,
            "index_build_shared_by_vector_and_hybrid": True,
            "persistent_index_loaded_once": needs_persistent,
            "cross_encoder_loaded_once": needs_cross_encoder,
        },
        "methods": method_results,
        "comparison": comparison,
        "metric_note": (
            "Recall/MRR/nDCG 只在完整标注的候选池上计算，不代表 PubMed 全库召回率；"
            "使用 assistant draft 标签时，结果只能作为流程试跑，不能视为人工金标准；"
            "查询延迟已排除模型加载和候选索引构建时间；"
            "相关性指标仅统计完整标注问题，质量与延迟指标统计全部问题；"
            "vector_persisted/hybrid_persisted 使用预构建的本地磁盘索引；"
            "hybrid_persisted_reranked 在混合检索候选上应用标题/摘要质量惩罚。"
            "hybrid_persisted_cross_reranked 使用本地 Cross-Encoder 进行相关性重排。"
            "hybrid_persisted_cross_quality_reranked 在相关性分数上叠加可解释的文献质量惩罚。"
            "advanced_persisted 使用确定性查询改写、Multi-Query、RRF 融合、去重和融合重排。"
            "advanced_persisted_filtered 额外应用显式年份、摘要和文献类型过滤。"
        ),
    }


def _load_bundle(snapshot: Path, labels_path: Path | None) -> SnapshotBundle:
    bundle = load_snapshot(snapshot)
    if labels_path is None:
        return bundle
    try:
        labels = LabelsFile.model_validate_json(labels_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SnapshotError(f"无法读取标签文件：{labels_path}") from exc
    except ValueError as exc:
        raise SnapshotError(f"标签文件格式不正确：{labels_path}") from exc
    if labels.corpus_version != bundle.manifest.corpus_version:
        raise SnapshotError(
            "标签文件与快照的 corpus_version 不一致："
            f"{labels.corpus_version} != {bundle.manifest.corpus_version}"
        )
    return bundle.model_copy(update={"labels": labels})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="在冻结语料上比较关键词、内存向量、混合和持久化检索。"
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("data/benchmark/pubmed_snapshot_v1"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="标签 JSON；不提供时使用 snapshot/labels_v1.json。",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHOD_CHOICES,
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--model", help="本地 SentenceTransformer 模型路径或名称。")
    parser.add_argument(
        "--cross-encoder-model",
        help="本地 Cross-Encoder 模型目录；不会自动联网下载。",
    )
    parser.add_argument("--filter-year-from", type=int)
    parser.add_argument("--filter-year-to", type=int)
    parser.add_argument("--filter-require-abstract", action="store_true")
    parser.add_argument(
        "--filter-exclude-article-types",
        nargs="+",
        choices=("editorial", "correction", "review"),
    )
    parser.add_argument(
        "--candidate-scope",
        choices=("question_pool", "global_corpus"),
        default="question_pool",
    )
    parser.add_argument(
        "--persistent-index",
        type=Path,
        help=(
            "本地持久化索引目录；默认使用 "
            "artifacts/local_vector_index_v1。"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmark/retrieval_benchmark_v1.json"),
    )
    args = parser.parse_args()

    try:
        bundle = _load_bundle(args.snapshot, args.labels)
        cli_filters = (
            RetrievalFilters(
                year_from=args.filter_year_from,
                year_to=args.filter_year_to,
                require_abstract=args.filter_require_abstract,
                exclude_article_types=tuple(
                    args.filter_exclude_article_types or ()
                ),
            )
            if any(
                (
                    args.filter_year_from is not None,
                    args.filter_year_to is not None,
                    args.filter_require_abstract,
                    bool(args.filter_exclude_article_types),
                )
            )
            else None
        )
        result = evaluate_retrieval_snapshot(
            bundle,
            methods=tuple(args.methods),
            candidate_scope=args.candidate_scope,
            model_name=args.model,
            label_source=str(args.labels) if args.labels else None,
            persistent_index_dir=args.persistent_index,
            cross_encoder_model_name=args.cross_encoder_model,
            retrieval_filters=cli_filters,
        )
    except (
        SnapshotError,
        EmbeddingError,
        PersistentEmbeddingError,
        CrossEncoderRerankerError,
        LocalVectorStoreError,
        OSError,
        ValueError,
    ) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
