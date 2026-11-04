"""Deterministic keyword baseline and metrics for a frozen candidate corpus."""

from __future__ import annotations

import math
import re
import time
from typing import Iterable, Literal

from app.corpus_snapshot import FrozenDocument, SnapshotBundle


CandidateScope = Literal["question_pool", "global_corpus"]
_TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def keyword_score(query: str, document: FrozenDocument) -> float:
    """Score by unique query-token overlap; this is intentionally not semantic."""

    query_terms = _tokens(query)
    if not query_terms:
        return 0.0
    document_terms = _tokens(f"{document.title}\n{document.abstract}")
    return len(query_terms & document_terms) / len(query_terms)


def rank_documents(
    query: str,
    documents: Iterable[FrozenDocument],
    *,
    top_k: int = 5,
) -> list[dict[str, object]]:
    if top_k < 1:
        raise ValueError("top_k 必须大于等于 1。")
    ranked = [
        (score, document)
        for document in documents
        for score in (keyword_score(query, document),)
        if score > 0
    ]
    ranked.sort(key=lambda item: (-item[0], item[1].pmid))
    return [
        {"rank": rank, "pmid": document.pmid, "score": score}
        for rank, (score, document) in enumerate(ranked[:top_k], start=1)
    ]


def _dcg(relevances: list[int]) -> float:
    return sum((2**relevance - 1) / math.log2(index + 2) for index, relevance in enumerate(relevances))


def _ndcg_at_k(ranked_pmids: list[str], relevance_by_pmid: dict[str, int], k: int) -> float:
    actual = [relevance_by_pmid.get(pmid, 0) for pmid in ranked_pmids[:k]]
    ideal = sorted(relevance_by_pmid.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    return _dcg(actual) / ideal_dcg if ideal_dcg else 0.0


def evaluate_snapshot(
    bundle: SnapshotBundle,
    *,
    top_ks: tuple[int, ...] = (1, 3, 5),
    candidate_scope: CandidateScope = "question_pool",
) -> dict[str, object]:
    """Evaluate only fully labelled cases and report incomplete coverage explicitly."""

    if not top_ks or any(k < 1 for k in top_ks):
        raise ValueError("top_ks 必须包含正整数。")
    documents_by_pmid = {document.pmid: document for document in bundle.documents}
    labels_by_question: dict[str, dict[str, int | None]] = {}
    for label in bundle.labels.labels:
        labels_by_question.setdefault(label.question_id, {})[label.pmid] = label.relevance

    cases: list[dict[str, object]] = []
    completed_cases: list[dict[str, object]] = []
    for question in bundle.questions:
        if candidate_scope == "question_pool":
            candidates = [
                document
                for document in bundle.documents
                if question.id in document.candidate_question_ids
            ]
        else:
            candidates = list(bundle.documents)
        candidate_ids = {document.pmid for document in candidates}
        relevance_by_pmid = labels_by_question.get(question.id, {})
        missing_pmids = sorted(
            pmid for pmid in candidate_ids if relevance_by_pmid.get(pmid) is None
        )
        unknown_label_pmids = sorted(
            pmid for pmid in relevance_by_pmid if pmid not in documents_by_pmid
        )
        started = time.perf_counter()
        ranked = rank_documents(question.question, candidates, top_k=max(top_ks))
        latency_ms = round((time.perf_counter() - started) * 1000, 3)
        ranked_pmids = [str(item["pmid"]) for item in ranked]

        case: dict[str, object] = {
            "question_id": question.id,
            "question": question.question,
            "candidate_count": len(candidates),
            "label_count": len(relevance_by_pmid),
            "missing_label_pmids": missing_pmids,
            "unknown_label_pmids": unknown_label_pmids,
            "status": "complete" if not missing_pmids and not unknown_label_pmids else "incomplete_labels",
            "latency_ms": latency_ms,
            "ranked": ranked,
        }
        if case["status"] != "complete":
            case["metrics"] = None
            cases.append(case)
            continue

        relevant_count = sum(1 for value in relevance_by_pmid.values() if value is not None and value > 0)
        case_metrics: dict[str, object] = {
            "recall": {},
            "ndcg": {},
            "mrr": 0.0,
            "has_relevant_label": relevant_count > 0,
        }
        for k in top_ks:
            top_ids = ranked_pmids[:k]
            if relevant_count:
                retrieved_relevant = sum(
                    1 for pmid in top_ids if (relevance_by_pmid.get(pmid) or 0) > 0
                )
                case_metrics["recall"][f"@{k}"] = retrieved_relevant / relevant_count
            else:
                case_metrics["recall"][f"@{k}"] = None
            case_metrics["ndcg"][f"@{k}"] = _ndcg_at_k(
                ranked_pmids, {pmid: int(value or 0) for pmid, value in relevance_by_pmid.items()}, k
            )
        for rank, pmid in enumerate(ranked_pmids, start=1):
            if (relevance_by_pmid.get(pmid) or 0) > 0:
                case_metrics["mrr"] = 1.0 / rank
                break
        case["metrics"] = case_metrics
        cases.append(case)
        completed_cases.append(case)

    def average(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 6) if values else None

    overall_recall: dict[str, float | None] = {}
    overall_ndcg: dict[str, float | None] = {}
    for k in top_ks:
        recall_values = [
            float(case["metrics"]["recall"][f"@{k}"])
            for case in completed_cases
            if case["metrics"]["recall"][f"@{k}"] is not None
        ]
        ndcg_values = [float(case["metrics"]["ndcg"][f"@{k}"]) for case in completed_cases]
        overall_recall[f"@{k}"] = average(recall_values)
        overall_ndcg[f"@{k}"] = average(ndcg_values)

    return {
        "benchmark": "keyword-baseline-v1",
        "corpus_version": bundle.manifest.corpus_version,
        "annotation_version": bundle.manifest.annotation_version,
        "coverage_scope": bundle.manifest.coverage_scope,
        "candidate_scope": candidate_scope,
        "metric_note": "Recall/MRR/nDCG 只在完整标注的候选池上计算，不代表 PubMed 全库召回率。",
        "total_cases": len(bundle.questions),
        "completed_cases": len(completed_cases),
        "incomplete_cases": len(bundle.questions) - len(completed_cases),
        "metrics": {
            "recall": overall_recall,
            "ndcg": overall_ndcg,
            "mrr": average([float(case["metrics"]["mrr"]) for case in completed_cases]),
            "no_result_rate": (
                sum(1 for case in completed_cases if not case["ranked"]) / len(completed_cases)
                if completed_cases
                else None
            ),
            "average_latency_ms": average([float(case["latency_ms"]) for case in completed_cases]),
        },
        "cases": cases,
    }
