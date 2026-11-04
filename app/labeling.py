"""Safe, resumable helpers for manually annotating a frozen corpus."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from app.corpus_snapshot import LabelsFile, RelevanceLabel, SnapshotError, SnapshotBundle, load_snapshot


def label_key(label: RelevanceLabel) -> tuple[str, str]:
    return label.question_id, label.pmid


def load_labels(path: Path) -> LabelsFile:
    try:
        return LabelsFile.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SnapshotError(f"无法读取标签文件：{path}") from exc
    except ValueError as exc:
        raise SnapshotError(f"标签文件格式错误：{path}") from exc


def merge_resume_labels(template: LabelsFile, previous: LabelsFile | None) -> LabelsFile:
    """Keep previous annotations while adding any new template rows."""

    if previous is None:
        return template
    if previous.corpus_version != template.corpus_version:
        raise SnapshotError("已有标注的 corpus_version 与当前快照不一致。")
    if previous.annotation_version != template.annotation_version:
        raise SnapshotError("已有标注的 annotation_version 与当前模板不一致。")

    previous_by_key = {label_key(label): label for label in previous.labels}
    merged: list[RelevanceLabel] = []
    for label in template.labels:
        old = previous_by_key.get(label_key(label))
        merged.append(old or label)
    return template.model_copy(update={"labels": merged})


def set_relevance(labels_file: LabelsFile, index: int, relevance: int) -> LabelsFile:
    if relevance not in (0, 1, 2):
        raise ValueError("relevance 必须是 0、1 或 2。")
    if not 0 <= index < len(labels_file.labels):
        raise IndexError("标签索引超出范围。")
    updated = labels_file.labels[index].model_copy(update={"relevance": relevance})
    labels = list(labels_file.labels)
    labels[index] = updated
    return labels_file.model_copy(update={"labels": labels})


def annotation_counts(labels_file: LabelsFile) -> dict[str, int]:
    counts = {"pending": 0, "relevance_0": 0, "relevance_1": 0, "relevance_2": 0}
    for label in labels_file.labels:
        if label.relevance is None:
            counts["pending"] += 1
        else:
            counts[f"relevance_{label.relevance}"] += 1
    counts["annotated"] = len(labels_file.labels) - counts["pending"]
    counts["total"] = len(labels_file.labels)
    return counts


def pending_label_indices(
    labels_file: LabelsFile,
    *,
    question_ids: list[str] | None = None,
    max_items: int | None = None,
) -> list[int]:
    """Return pending label indices for an optional pilot subset.

    The returned indices still refer to the original full ``LabelsFile``. This
    lets the annotator save a complete label structure while only presenting a
    selected subset to the user.
    """

    if max_items is not None and max_items < 1:
        raise ValueError("max_items 必须大于等于 1。")

    selected = set(question_ids or [])
    pending = [
        index
        for index, label in enumerate(labels_file.labels)
        if label.relevance is None
        and (not selected or label.question_id in selected)
    ]
    if max_items is not None:
        pending = pending[:max_items]
    return pending


def save_labels_atomic(labels_file: LabelsFile, path: Path) -> None:
    """Write a label file via a same-directory temporary file and replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(labels_file.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def load_annotation_context(snapshot_dir: Path, output_path: Path | None = None) -> tuple[SnapshotBundle, LabelsFile, Path]:
    """Load a snapshot and merge an existing annotated output for resuming."""

    bundle = load_snapshot(snapshot_dir)
    output = output_path or snapshot_dir / "labels_v1_annotated.json"
    previous = load_labels(output) if output.exists() else None
    labels = merge_resume_labels(bundle.labels, previous)
    return bundle, labels, output
