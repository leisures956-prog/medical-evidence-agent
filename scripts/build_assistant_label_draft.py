"""Build a transparent, model-assisted relevance-label draft.

This script never modifies the frozen corpus or the original labels template.
It writes a separate ``labels_v1_assistant_draft.json`` file so that a human
annotator can review and correct the proposed labels before using them as an
evaluation reference.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Allow ``python scripts/build_assistant_label_draft.py`` from the repository
# root without requiring the caller to set PYTHONPATH manually.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.corpus_snapshot import LabelsFile
from app.labeling import load_labels, save_labels_atomic


# Scores are in the same per-question order as labels_v1.json. The script
# validates the question id and row count before applying any score, so an
# accidental change in the template cannot silently shift annotations.
DRAFT_SCORES: dict[str, list[int]] = {
    "q01": [1, 2, 2, 1, 2, 0, 0, 1, 0, 2, 2, 1, 2, 2, 1, 2, 2, 0, 1, 1],
    "q02": [1, 1, 2, 2, 1, 0, 2, 2, 2, 2, 1, 2, 1, 1, 0, 2, 1, 1, 1, 1],
    "q03": [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 0, 0, 2, 0, 2, 0, 0, 2],
    "q04": [2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 0, 2, 2, 2],
    "q05": [1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2],
    "q06": [0, 1, 2, 1, 1, 2, 1, 2, 1, 1, 0, 1, 0, 0, 0, 1, 2, 1, 2, 2],
    "q07": [2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    "q08": [0, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 1, 1],
    "q09": [1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 0, 1, 2, 2, 1, 2, 2, 1, 1],
    "q10": [1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2],
    "q11": [0, 1, 1, 2, 2, 1, 2, 0, 2, 0, 1, 2, 1, 2, 1, 2, 2, 1, 2, 0],
    "q12": [2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2],
}


def _review_note(score: int) -> str:
    if score == 2:
        return "模型辅助初稿：标题/摘要与问题直接相关；仍需人工复核。"
    if score == 1:
        return "模型辅助初稿：部分相关、间接相关或信息不完整；建议优先人工复核。"
    return "模型辅助初稿：主题或研究对象不匹配；仍需人工复核。"


def build_draft(input_path: Path, output_path: Path) -> LabelsFile:
    template = load_labels(input_path)
    expected_question_ids = set(DRAFT_SCORES)
    actual_question_ids = {label.question_id for label in template.labels}
    if actual_question_ids != expected_question_ids:
        raise ValueError(
            "标签模板的问题 id 与辅助标注表不一致："
            f"实际={sorted(actual_question_ids)}，预设={sorted(expected_question_ids)}"
        )

    positions: Counter[str] = Counter()
    updated = []
    for label in template.labels:
        index = positions[label.question_id]
        scores = DRAFT_SCORES[label.question_id]
        if index >= len(scores):
            raise ValueError(f"{label.question_id} 的模板行数超过预设辅助标注数量。")
        score = scores[index]
        positions[label.question_id] += 1
        updated.append(
            label.model_copy(
                update={
                    "relevance": score,
                    "annotator": "assistant_draft_v1",
                    "notes": _review_note(score),
                }
            )
        )

    expected_counts = {question_id: len(scores) for question_id, scores in DRAFT_SCORES.items()}
    if dict(positions) != expected_counts:
        raise ValueError(f"模板行数不完整：实际={dict(positions)}，预设={expected_counts}")

    draft = template.model_copy(update={"labels": updated})
    save_labels_atomic(draft, output_path)
    return draft


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 PubMed 相关性辅助标注初稿")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/benchmark/pubmed_snapshot_v1/labels_v1.json"),
        help="原始标签模板（只读）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmark/pubmed_snapshot_v1/labels_v1_assistant_draft.json"),
        help="辅助标注初稿输出路径",
    )
    args = parser.parse_args()
    if args.output.resolve() == args.input.resolve():
        raise SystemExit("拒绝将辅助标注输出到原始 labels_v1.json。")
    draft = build_draft(args.input, args.output)
    counts = Counter(label.relevance for label in draft.labels)
    print(f"已生成：{args.output}")
    print(f"总条目：{len(draft.labels)}；0={counts[0]}，1={counts[1]}，2={counts[2]}")
    print("注意：这是模型辅助初稿，不是人工金标准；请复核后再用于最终评测。")


if __name__ == "__main__":
    main()
