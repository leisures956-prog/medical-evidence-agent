"""Run the reproducible keyword-only baseline on a frozen snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.corpus_snapshot import SnapshotError, load_snapshot
from app.keyword_baseline import evaluate_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate keyword retrieval on a frozen corpus.")
    parser.add_argument("--snapshot", type=Path, default=Path("data/benchmark/pubmed_snapshot_v1"))
    parser.add_argument("--labels", type=Path, help="edited labels JSON; defaults to snapshot/labels_v1.json")
    parser.add_argument("--output", type=Path, default=Path("data/benchmark/keyword_baseline_v1.json"))
    parser.add_argument("--candidate-scope", choices=("question_pool", "global_corpus"), default="question_pool")
    args = parser.parse_args()

    try:
        bundle = load_snapshot(args.snapshot)
        if args.labels:
            from app.corpus_snapshot import LabelsFile

            bundle = bundle.model_copy(
                update={
                    "labels": LabelsFile.model_validate_json(
                        args.labels.read_text(encoding="utf-8")
                    )
                }
            )
        result = evaluate_snapshot(bundle, candidate_scope=args.candidate_scope)
    except (SnapshotError, OSError, ValueError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
