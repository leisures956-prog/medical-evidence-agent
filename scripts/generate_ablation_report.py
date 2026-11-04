"""Generate CSV and Markdown summaries from a retrieval benchmark JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SUMMARY_COLUMNS = (
    "method",
    "completed_cases",
    "total_cases",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "mrr",
    "ndcg_at_1",
    "ndcg_at_3",
    "ndcg_at_5",
    "abstract_missing_rate_at_5",
    "editorial_rate_at_5",
    "any_quality_flag_rate_at_5",
    "average_query_latency_ms",
    "median_query_latency_ms",
    "p95_query_latency_ms",
    "average_query_variant_count",
    "average_duplicates_removed",
    "average_filtered_out_count",
)


def _nested(mapping: dict[str, Any], *keys: str):
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def build_summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, payload in result.get("methods", {}).items():
        metrics = payload.get("metrics", {})
        rows.append(
            {
                "method": method,
                "completed_cases": payload.get("completed_cases"),
                "total_cases": payload.get("total_cases"),
                "recall_at_1": _nested(metrics, "recall", "@1"),
                "recall_at_3": _nested(metrics, "recall", "@3"),
                "recall_at_5": _nested(metrics, "recall", "@5"),
                "mrr": metrics.get("mrr"),
                "ndcg_at_1": _nested(metrics, "ndcg", "@1"),
                "ndcg_at_3": _nested(metrics, "ndcg", "@3"),
                "ndcg_at_5": _nested(metrics, "ndcg", "@5"),
                "abstract_missing_rate_at_5": _nested(
                    metrics, "quality_at_k", "@5", "abstract_missing_rate"
                ),
                "editorial_rate_at_5": _nested(
                    metrics, "quality_at_k", "@5", "editorial_rate"
                ),
                "any_quality_flag_rate_at_5": _nested(
                    metrics, "quality_at_k", "@5", "any_quality_flag_rate"
                ),
                "average_query_latency_ms": metrics.get(
                    "average_query_latency_ms"
                ),
                "median_query_latency_ms": metrics.get(
                    "median_query_latency_ms"
                ),
                "p95_query_latency_ms": metrics.get("p95_query_latency_ms"),
                "average_query_variant_count": metrics.get(
                    "average_query_variant_count"
                ),
                "average_duplicates_removed": metrics.get(
                    "average_duplicates_removed"
                ),
                "average_filtered_out_count": metrics.get(
                    "average_filtered_out_count"
                ),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _display(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(result: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    scopes = result.get("metric_scopes", {})
    filters = result.get("filter_config", {})
    lines = [
        "# Stage 4 Retrieval Ablation Report (Pilot)",
        "",
        "## Experiment scope",
        "",
        f"- Corpus version: `{result.get('corpus_version')}`",
        f"- Annotation version: `{result.get('annotation_version')}`",
        f"- Candidate scope: `{result.get('candidate_scope')}`",
        f"- Relevance metrics: `{scopes.get('relevance')}`",
        f"- Quality metrics: `{scopes.get('quality')}`",
        f"- Latency metrics: `{scopes.get('latency')}`",
        "- Recall/MRR/nDCG are candidate-pool pilot metrics, not PubMed-wide recall.",
        "",
        "## Filter configuration",
        "",
        "```json",
        json.dumps(filters, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Results",
        "",
        "| Method | Labeled | Recall@5 | nDCG@5 | MRR | Missing abstract@5 | Editorial@5 | Avg ms | Median ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {completed}/{total} | {recall} | {ndcg} | {mrr} | "
            "{missing} | {editorial} | {average} | {median} | {p95} |".format(
                method=row["method"],
                completed=row["completed_cases"],
                total=row["total_cases"],
                recall=_display(row["recall_at_5"]),
                ndcg=_display(row["ndcg_at_5"]),
                mrr=_display(row["mrr"]),
                missing=_display(row["abstract_missing_rate_at_5"]),
                editorial=_display(row["editorial_rate_at_5"]),
                average=_display(row["average_query_latency_ms"], 1),
                median=_display(row["median_query_latency_ms"], 1),
                p95=_display(row["p95_query_latency_ms"], 1),
            )
        )

    by_method = {row["method"]: row for row in rows}
    baseline = by_method.get("hybrid_persisted")
    quality = by_method.get("hybrid_persisted_reranked")
    cross = by_method.get("hybrid_persisted_cross_reranked")
    advanced = by_method.get("advanced_persisted")
    filtered = by_method.get("advanced_persisted_filtered")
    lines.extend(["", "## Observed pilot findings", ""])
    if baseline and quality:
        lines.append(
            "- Rule-based quality reranking kept Recall@5 at "
            f"{_display(quality['recall_at_5'])}, reduced missing-abstract@5 "
            f"from {_display(baseline['abstract_missing_rate_at_5'])} to "
            f"{_display(quality['abstract_missing_rate_at_5'])}, and changed "
            f"nDCG@5 from {_display(baseline['ndcg_at_5'])} to "
            f"{_display(quality['ndcg_at_5'])}."
        )
    if baseline and cross:
        lines.append(
            "- Cross-Encoder reranking showed no observed Recall@5 gain in this "
            f"pilot ({_display(baseline['recall_at_5'])} vs "
            f"{_display(cross['recall_at_5'])}) while average latency increased "
            f"from {_display(baseline['average_query_latency_ms'], 1)} ms to "
            f"{_display(cross['average_query_latency_ms'], 1)} ms."
        )
    if advanced and quality:
        lines.append(
            "- Multi-query retrieval and deduplication expanded candidate coverage, "
            "but no additional Recall@5 gain was observed over the quality-aware "
            f"single-query pipeline ({_display(advanced['recall_at_5'])})."
        )
    if advanced and filtered:
        lines.append(
            "- Hard filtering reduced missing-abstract@5 from "
            f"{_display(advanced['abstract_missing_rate_at_5'])} to "
            f"{_display(filtered['abstract_missing_rate_at_5'])}, while nDCG@5 "
            f"changed from {_display(advanced['ndcg_at_5'])} to "
            f"{_display(filtered['ndcg_at_5'])}."
        )

    lines.extend(["", "## Recommended modes", ""])
    if baseline:
        lines.append(
            "- Balanced default: `hybrid_persisted` — strong pilot relevance with "
            "low local latency."
        )
    if quality:
        lines.append(
            "- Evidence-quality default: `hybrid_persisted_reranked` — removes most "
            "missing-abstract/editorial results without Cross-Encoder latency."
        )
    if filtered:
        lines.append(
            "- Strict demonstration mode: `advanced_persisted` with explicit filters — "
            "use when traceable evidence completeness matters more than latency."
        )

    completed = max((row.get("completed_cases") or 0 for row in rows), default=0)
    total = max((row.get("total_cases") or 0 for row in rows), default=0)
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            f"- Only {completed}/{total} questions have complete relevance labels.",
            "- Quality and latency summaries use all questions.",
            "- A method with more components is not automatically better.",
            "- Multi-query coverage and deduplication are engineering outcomes; a Recall gain must be demonstrated separately.",
            "- Hard filters can improve evidence completeness while lowering graded ranking quality.",
            "",
            "## Current limitations",
            "",
            "- Small candidate-pool annotation rather than exhaustive PubMed judgments.",
            "- Query latency is measured locally and excludes model/index loading.",
            "- The deterministic query expansion dictionary covers only selected medical concepts.",
            "- Results should be described as pilot findings until more questions are fully labeled.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成阶段4消融CSV和Markdown报告。")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.input.read_text(encoding="utf-8"))
    rows = build_summary_rows(result)
    write_csv(rows, args.csv)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(
        render_markdown(result, rows),
        encoding="utf-8",
    )
    print(f"CSV: {args.csv}")
    print(f"Markdown: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
