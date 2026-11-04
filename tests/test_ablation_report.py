import csv
import json

from scripts.generate_ablation_report import (
    build_summary_rows,
    render_markdown,
    write_csv,
)


def _result():
    return {
        "corpus_version": "test-v1",
        "annotation_version": "labels-v1",
        "candidate_scope": "question_pool",
        "metric_scopes": {
            "relevance": "complete_labeled_cases_only",
            "quality": "all_cases",
            "latency": "all_cases",
        },
        "filter_config": {"require_abstract": True},
        "methods": {
            "hybrid_persisted": {
                "completed_cases": 1,
                "total_cases": 2,
                "metrics": {
                    "recall": {"@1": 0.5, "@3": 1.0, "@5": 1.0},
                    "ndcg": {"@1": 1.0, "@3": 1.0, "@5": 1.0},
                    "mrr": 1.0,
                    "quality_at_k": {
                        "@5": {
                            "abstract_missing_rate": 0.1,
                            "editorial_rate": 0.0,
                            "any_quality_flag_rate": 0.1,
                        }
                    },
                    "average_query_latency_ms": 10.0,
                    "median_query_latency_ms": 9.0,
                    "p95_query_latency_ms": 12.0,
                    "average_query_variant_count": None,
                    "average_duplicates_removed": None,
                    "average_filtered_out_count": None,
                },
            }
        },
    }


def test_summary_rows_keep_relevance_quality_and_latency_metrics():
    rows = build_summary_rows(_result())

    assert rows[0]["method"] == "hybrid_persisted"
    assert rows[0]["recall_at_5"] == 1.0
    assert rows[0]["abstract_missing_rate_at_5"] == 0.1
    assert rows[0]["p95_query_latency_ms"] == 12.0


def test_csv_and_markdown_are_generated_from_same_rows(tmp_path):
    result = _result()
    rows = build_summary_rows(result)
    csv_path = tmp_path / "report.csv"
    write_csv(rows, csv_path)
    markdown = render_markdown(result, rows)

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        parsed = list(csv.DictReader(file))

    assert parsed[0]["method"] == "hybrid_persisted"
    assert "hybrid_persisted" in markdown
    assert "Only 1/2 questions" in markdown
    assert json.dumps(result["filter_config"], ensure_ascii=False, indent=2) in markdown
    assert "Observed pilot findings" in markdown
    assert "Recommended modes" in markdown
