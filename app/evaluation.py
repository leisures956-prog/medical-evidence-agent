from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from app.citation_validation import validate_citations
from app.evidence_extractor import EvidenceExtractor, EvidenceMode
from app.medical_models import StructuredEvidence
from app.search import SearchHit, SearchService, medical_document


class EvaluationQuestion(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    required_terms: list[str] = Field(default_factory=list)
    min_results: int = Field(default=1, ge=0)


class EvaluationCaseResult(BaseModel):
    id: str
    question: str
    retrieved_count: int = Field(ge=0)
    retrieval_ok: bool
    topic_match: bool
    json_complete: bool
    citation_valid: bool
    abstract_evidence_coverage: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
    errors: list[str] = Field(default_factory=list)


class EvaluationSummary(BaseModel):
    total_cases: int = Field(ge=0)
    retrieval_success_rate: float = Field(ge=0, le=1)
    topic_match_rate: float = Field(ge=0, le=1)
    json_complete_rate: float = Field(ge=0, le=1)
    citation_valid_rate: float = Field(ge=0, le=1)
    abstract_evidence_coverage: float = Field(ge=0, le=1)
    average_latency_ms: float = Field(ge=0)
    llm_calls: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    cases: list[EvaluationCaseResult] = Field(default_factory=list)


class SearchBackend(Protocol):
    def search(
        self,
        query: str,
        limit: int,
        source: str = "local",
        candidate_limit: int = 30,
    ) -> tuple[list[SearchHit], int]:
        ...


def load_evaluation_questions(path: Path | None = None) -> list[EvaluationQuestion]:
    path = path or Path(__file__).resolve().parents[1] / "data" / "evaluation_questions.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"评测问题文件读取失败：{path}") from exc
    if not isinstance(payload, list):
        raise RuntimeError("评测问题文件必须是 JSON 数组。")
    return [EvaluationQuestion.model_validate(item) for item in payload]


def _topic_match(hits: list[SearchHit], required_terms: list[str]) -> bool:
    if not required_terms:
        return True
    corpus = "\n".join(
        f"{hit.title}\n{hit.content}" for hit in hits
    ).lower()
    return all(term.lower() in corpus for term in required_terms)


def _json_complete(records: list[StructuredEvidence]) -> bool:
    required_fields = {
        "pmid",
        "source_url",
        "title",
        "title_evidence_ids",
        "research_object",
        "disease",
        "method",
        "dataset",
        "main_findings",
        "limitations",
        "evidence_spans",
    }
    for record in records:
        if not required_fields.issubset(record.model_dump().keys()):
            return False
        if not record.pmid or not record.source_url or not record.evidence_spans:
            return False
        if not record.title_evidence_ids:
            return False
    return True


def _citation_answer(records: list[StructuredEvidence]) -> str:
    references = [
        f"[{index}:{record.title_evidence_ids[0]}]"
        for index, record in enumerate(records, start=1)
        if record.title_evidence_ids
    ]
    return "检索证据：" + " ".join(references)


def run_fixed_evaluation(
    service: SearchBackend,
    questions: list[EvaluationQuestion] | None = None,
    *,
    top_k: int = 3,
    candidate_limit: int = 20,
    mode: EvidenceMode = "metadata_only",
    extractor: EvidenceExtractor | None = None,
    estimated_cost_per_call_usd: float = 0.0,
) -> EvaluationSummary:
    """Run the fixed evaluation set without asserting unstable PubMed PMIDs."""

    cases = questions or load_evaluation_questions()
    extractor = extractor or EvidenceExtractor()
    results: list[EvaluationCaseResult] = []
    total_records = 0
    abstract_records = 0
    llm_calls = 0

    for case in cases:
        started = time.perf_counter()
        errors: list[str] = []
        hits: list[SearchHit] = []
        records: list[StructuredEvidence] = []
        try:
            hits, _ = service.search(
                case.question,
                top_k,
                "pubmed_hybrid",
                candidate_limit=candidate_limit,
            )
            if len(hits) < case.min_results:
                errors.append(
                    f"检索结果不足：得到 {len(hits)} 篇，期望至少 {case.min_results} 篇。"
                )
            records = [
                extractor.extract(medical_document(hit), mode=mode) for hit in hits
            ]
            if mode == "strict_llm":
                llm_calls += len(hits)
        except Exception as exc:
            errors.append(str(exc))

        retrieval_ok = len(hits) >= case.min_results
        topic_match = _topic_match(hits, case.required_terms) if hits else False
        json_complete = _json_complete(records) if records else False
        citation_result = validate_citations(_citation_answer(records), records)
        citation_valid = citation_result.valid if records else False
        total_records += len(records)
        abstract_records += sum(
            any(span.source_section == "abstract" for span in record.evidence_spans)
            for record in records
        )
        latency_ms = (time.perf_counter() - started) * 1000
        results.append(
            EvaluationCaseResult(
                id=case.id,
                question=case.question,
                retrieved_count=len(hits),
                retrieval_ok=retrieval_ok,
                topic_match=topic_match,
                json_complete=json_complete,
                citation_valid=citation_valid,
                abstract_evidence_coverage=(
                    sum(
                        any(span.source_section == "abstract" for span in record.evidence_spans)
                        for record in records
                    )
                    / len(records)
                    if records
                    else 0.0
                ),
                latency_ms=latency_ms,
                errors=errors + citation_result.errors,
            )
        )

    total = len(results)
    denominator = total or 1
    return EvaluationSummary(
        total_cases=total,
        retrieval_success_rate=sum(item.retrieval_ok for item in results) / denominator,
        topic_match_rate=sum(item.topic_match for item in results) / denominator,
        json_complete_rate=sum(item.json_complete for item in results) / denominator,
        citation_valid_rate=sum(item.citation_valid for item in results) / denominator,
        abstract_evidence_coverage=(abstract_records / total_records if total_records else 0.0),
        average_latency_ms=(sum(item.latency_ms for item in results) / denominator),
        llm_calls=llm_calls,
        estimated_cost_usd=llm_calls * estimated_cost_per_call_usd,
        cases=results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="运行固定 PubMed RAG 评测集")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument(
        "--mode", choices=("metadata_only", "strict_llm"), default="metadata_only"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = run_fixed_evaluation(
        SearchService(),
        top_k=args.top_k,
        candidate_limit=args.candidate_limit,
        mode=args.mode,
    )
    output = summary.model_dump_json(indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
