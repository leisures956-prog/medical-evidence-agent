"""Freeze PubMed candidate pools for later offline retrieval experiments.

The command requires either ``--records`` (offline JSON input) or ``--live``.
This prevents an accidental network call while running tests or preparing a
demo.  The output directory contains a manifest, questions, documents and an
empty relevance-label template.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.corpus_snapshot import SnapshotError, build_snapshot, load_questions, write_snapshot
from app.medical_models import MedicalDocument
from app.pubmed_client_v2 import PubMedClientV2


def _load_offline_records(path: Path, question_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SnapshotError(f"无法读取离线候选文件：{path}") from exc
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"离线候选文件不是有效 JSON：{path}") from exc

    if isinstance(payload, dict) and isinstance(payload.get("records_by_question"), dict):
        payload = payload["records_by_question"]
    if isinstance(payload, dict):
        result = {question_id: list(payload.get(question_id, [])) for question_id in question_ids}
    elif isinstance(payload, list):
        result = defaultdict(list)
        for item in payload:
            if not isinstance(item, dict) or not item.get("question_id"):
                raise SnapshotError("列表格式的离线候选记录必须包含 question_id。")
            record = item.get("document", item)
            result[str(item["question_id"])].append(record)
        result = dict(result)
    else:
        raise SnapshotError("离线候选文件应是 question_id 到记录数组的对象，或记录数组。")
    unknown = sorted(set(result) - set(question_ids))
    if unknown:
        raise SnapshotError(f"离线候选文件包含未知 question_id：{unknown}")
    return result


def _fetch_live(questions, candidate_limit: int) -> dict[str, list[MedicalDocument]]:
    client = PubMedClientV2()
    records: dict[str, list[MedicalDocument]] = {}
    for question in questions:
        print(f"Fetching {question.id}: {question.question}", file=sys.stderr)
        records[question.id] = client.search(question.question, limit=candidate_limit)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze PubMed candidates into a versioned snapshot.")
    parser.add_argument("--questions", type=Path, default=Path("data/benchmark/questions_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("data/benchmark/pubmed_snapshot_v1"))
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--records", type=Path, help="offline JSON candidate records")
    parser.add_argument("--live", action="store_true", help="explicitly allow PubMed network requests")
    parser.add_argument(
        "--force",
        action="store_true",
        help="允许覆盖已有快照；默认会保护已有 documents/labels 文件",
    )
    args = parser.parse_args()

    if bool(args.records) == bool(args.live):
        parser.error("必须二选一：提供 --records 离线输入，或显式指定 --live。")
    protected_files = [
        args.output / "manifest.json",
        args.output / "documents.jsonl",
        args.output / "labels_v1.json",
    ]
    if not args.force and any(path.exists() for path in protected_files):
        parser.error(f"输出目录已有冻结文件：{args.output}；如确实要重建，请显式添加 --force。")
    try:
        questions = load_questions(args.questions)
        if args.records:
            records = _load_offline_records(args.records, [question.id for question in questions])
        else:
            records = _fetch_live(questions, args.candidate_limit)
        bundle = build_snapshot(questions, records, candidate_limit=args.candidate_limit)
        write_snapshot(bundle, args.output)
    except SnapshotError as exc:
        parser.error(str(exc))
    except Exception as exc:
        print(f"冻结语料失败：{exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "output": str(args.output),
                "corpus_version": bundle.manifest.corpus_version,
                "question_count": bundle.manifest.question_count,
                "document_count": bundle.manifest.document_count,
                "label_count": len(bundle.labels.labels),
                "coverage_scope": bundle.manifest.coverage_scope,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
