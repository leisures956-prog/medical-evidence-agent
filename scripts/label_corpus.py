"""Interactive terminal annotator for a frozen PubMed candidate snapshot."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.corpus_snapshot import SnapshotError
from app.labeling import (
    annotation_counts,
    load_annotation_context,
    pending_label_indices,
    save_labels_atomic,
    set_relevance,
)


def _print_wrapped(text: str, width: int = 100) -> None:
    value = text or "（摘要为空）"
    print("\n".join(textwrap.wrap(value, width=width, replace_whitespace=False, drop_whitespace=True)))


def _show_item(bundle, labels_file, index: int, pending_position: int, pending_total: int) -> None:
    label = labels_file.labels[index]
    question = next((item.question for item in bundle.questions if item.id == label.question_id), label.question_id)
    document = next((item for item in bundle.documents if item.pmid == label.pmid), None)
    print("\n" + "=" * 100)
    print(f"进度：{pending_position}/{pending_total}（本次待处理）")
    print(f"问题：{question}")
    print(f"PMID：{label.pmid}")
    if document is None:
        print("警告：快照中找不到该 PMID。")
        return
    print(f"标题：{document.title}")
    print(f"年份：{document.year or '未报告'}    DOI：{document.doi or '未报告'}")
    print("摘要：")
    _print_wrapped(document.abstract)
    print("-" * 100)
    print("2=高度相关  1=部分相关  0=不相关  s=跳过  q=保存并退出")


def run_interactive(
    snapshot_dir: Path,
    output_path: Path | None = None,
    *,
    question_ids: list[str] | None = None,
    max_items: int | None = None,
) -> int:
    bundle, labels_file, output = load_annotation_context(snapshot_dir, output_path)
    available_question_ids = {question.id for question in bundle.questions}
    selected_question_ids = set(question_ids or [])
    unknown_question_ids = sorted(selected_question_ids - available_question_ids)
    if unknown_question_ids:
        raise SnapshotError(
            "问题 ID 不存在于当前快照：" + ", ".join(unknown_question_ids)
        )

    pending = pending_label_indices(
        labels_file,
        question_ids=question_ids,
        max_items=max_items,
    )
    counts = annotation_counts(labels_file)
    print(f"语料版本：{bundle.manifest.corpus_version}")
    print(f"标签文件：{output}")
    print(f"当前进度：{counts['annotated']}/{counts['total']}，待处理 {counts['pending']} 条")
    if selected_question_ids:
        print("本次问题筛选：" + ", ".join(question_ids or []))
    if max_items is not None:
        print(f"本次最多处理：{max_items} 条")
    print(f"本次待处理：{len(pending)} 条")
    if not pending:
        print("所有候选关系都已标注，无需继续操作。")
        return 0

    cursor = 0
    while cursor < len(pending):
        index = pending[cursor]
        _show_item(bundle, labels_file, index, cursor + 1, len(pending))
        try:
            choice = input("请输入 0/1/2/s/q：").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n检测到退出，正在保存当前进度。")
            save_labels_atomic(labels_file, output)
            return 0

        if choice == "q":
            save_labels_atomic(labels_file, output)
            print(f"已保存：{output}")
            return 0
        if choice == "s":
            cursor += 1
            continue
        if choice not in {"0", "1", "2"}:
            print("输入无效，请输入 0、1、2、s 或 q。")
            continue

        labels_file = set_relevance(labels_file, index, int(choice))
        save_labels_atomic(labels_file, output)
        cursor += 1

    print(f"\n标注完成，结果已保存到：{output}")
    print(annotation_counts(labels_file))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="逐条标注冻结 PubMed 候选文献。")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("data/benchmark/pubmed_snapshot_v1"),
        help="冻结语料目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="标注结果路径；默认写入 snapshot/labels_v1_annotated.json",
    )
    parser.add_argument(
        "--question-id",
        dest="question_ids",
        nargs="+",
        help="只标注指定问题，例如：--question-id q01 q05 q08",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="本次最多展示和处理多少条待标注记录",
    )
    args = parser.parse_args()
    try:
        return run_interactive(
            args.snapshot,
            args.output,
            question_ids=args.question_ids,
            max_items=args.max_items,
        )
    except (SnapshotError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
