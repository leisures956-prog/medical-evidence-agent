"""Run fixed answer/citation evaluation in mock or imported-response mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.answer_evaluation import (
    AnswerEvaluationCase,
    AnswerObservation,
    load_answer_evaluation_cases,
    summarize_answer_evaluation,
)
from app.medical_models import CitationBinding


def _mock_observation(case: AnswerEvaluationCase) -> AnswerObservation:
    if case.expected_behavior == "abstain":
        reason = case.expected_abstention_reasons[0] if case.expected_abstention_reasons else "insufficient_evidence"
        return AnswerObservation(
            answer="当前固定语料没有足够证据回答该问题。",
            abstained=True,
            abstention_reason=reason,
        )

    claim = case.expected_claims[0]
    terms = "、".join(claim.support_terms)
    exact_text = "The abstract reports " + ", ".join(claim.support_terms) + "."
    binding = CitationBinding(
        citation_number=1,
        pmid="mock-pmid",
        evidence_id="abstract-1",
        source_section=claim.expected_source_section,
        exact_text=exact_text,
        source_url="https://pubmed.ncbi.nlm.nih.gov/mock-pmid/",
    )
    return AnswerObservation(
        answer=f"The abstract reports {terms} [1:abstract-1].",
        abstained=False,
        citations=[1],
        citation_bindings=[binding],
    )


def _load_observations(path: Path) -> dict[str, AnswerObservation]:
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"回答结果文件读取失败：{path}") from exc
    rows = payload.get("answers") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise RuntimeError("回答结果文件必须是数组，或包含 answers 数组。")
    observations: dict[str, AnswerObservation] = {}
    for row in rows:
        if not isinstance(row, dict) or not row.get("case_id"):
            raise RuntimeError("每条回答结果必须包含 case_id。")
        observations[str(row["case_id"])] = AnswerObservation.model_validate(row)
    return observations


def main() -> None:
    parser = argparse.ArgumentParser(description="运行回答与引用支持评测")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--mode", choices=("mock", "json"), default="mock")
    parser.add_argument("--input", type=Path, default=None, help="/answer 结果 JSON 文件")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cases = load_answer_evaluation_cases(args.dataset)
    if args.mode == "json":
        if args.input is None:
            parser.error("--mode json 必须提供 --input。")
        observations = _load_observations(args.input)
    else:
        observations = {case.id: _mock_observation(case) for case in cases}

    summary = summarize_answer_evaluation(cases, observations)
    payload = summary.model_dump_json(indent=2, ensure_ascii=False)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
