"""Versioned PubMed candidate snapshots and relevance-label schemas.

The snapshot is intentionally small and explicit.  It freezes the exact
title/abstract candidates used by later retrieval experiments, while making
the limitation of result-pool annotation visible in the manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from pydantic import BaseModel, Field, model_validator

from app.medical_models import MedicalDocument


CORPUS_VERSION = "pubmed-snapshot-v1"
ANNOTATION_VERSION = "labels-v1"
HASH_ALGORITHM = "sha256"
CoverageScope = Literal["candidate_pool_only", "expanded_corpus"]


class SnapshotError(ValueError):
    """Raised when a corpus snapshot is malformed or internally inconsistent."""


class BenchmarkQuestion(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    required_terms: list[str] = Field(default_factory=list)
    min_results: int = Field(default=1, ge=0)


class FrozenDocument(BaseModel):
    """A PubMed document as it existed when a corpus snapshot was created."""

    pmid: str = Field(min_length=1)
    title: str = Field(min_length=1)
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    journal: str | None = None
    source_url: str = Field(min_length=1)
    fetched_at: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_version: str = CORPUS_VERSION
    candidate_question_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_hash(self):
        expected = content_sha256(self.title, self.abstract)
        if expected != self.content_sha256:
            raise ValueError("content_sha256 与标题/摘要内容不匹配。")
        if len(set(self.candidate_question_ids)) != len(self.candidate_question_ids):
            raise ValueError("candidate_question_ids 不能重复。")
        return self


class SnapshotManifest(BaseModel):
    """Metadata needed to reproduce and interpret a frozen corpus."""

    corpus_version: str = CORPUS_VERSION
    annotation_version: str = ANNOTATION_VERSION
    source: Literal["PubMed E-utilities"] = "PubMed E-utilities"
    created_at: str = Field(min_length=1)
    fetched_at: str = Field(min_length=1)
    question_count: int = Field(ge=0)
    question_ids: list[str] = Field(default_factory=list)
    candidate_limit_per_question: int = Field(ge=1)
    document_count: int = Field(ge=0)
    hash_algorithm: Literal["sha256"] = HASH_ALGORITHM
    hash_input: str = "normalized_title + newline + normalized_abstract"
    coverage_scope: CoverageScope = "candidate_pool_only"
    pool_definition: str = (
        "Union of the frozen PubMed candidate pools; labels cover only the "
        "candidate pool returned for each question."
    )


class RelevanceLabel(BaseModel):
    """One human label; ``None`` means that annotation is still pending."""

    question_id: str = Field(min_length=1)
    pmid: str = Field(min_length=1)
    relevance: int | None = Field(default=None, ge=0, le=2)
    coverage_scope: CoverageScope = "candidate_pool_only"
    annotation_version: str = ANNOTATION_VERSION
    annotator: str | None = None
    notes: str = ""


class LabelsFile(BaseModel):
    annotation_version: str = ANNOTATION_VERSION
    corpus_version: str = CORPUS_VERSION
    coverage_scope: CoverageScope = "candidate_pool_only"
    labels: list[RelevanceLabel] = Field(default_factory=list)


class SnapshotBundle(BaseModel):
    manifest: SnapshotManifest
    questions: list[BenchmarkQuestion]
    documents: list[FrozenDocument]
    labels: LabelsFile


def _normalise_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    return re.sub(r"\s+", " ", value).strip()


def content_sha256(title: str, abstract: str) -> str:
    """Return the stable hash used to detect changes to title/abstract text."""

    payload = f"{_normalise_text(title)}\n{_normalise_text(abstract)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_questions(path: Path) -> list[BenchmarkQuestion]:
    """Load either the versioned wrapper or the original list format."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SnapshotError(f"无法读取问题集：{path}") from exc
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"问题集不是有效 JSON：{path}") from exc

    if isinstance(payload, dict):
        raw_questions = payload.get("questions")
    else:
        raw_questions = payload
    if not isinstance(raw_questions, list):
        raise SnapshotError("问题集必须是数组，或包含 questions 数组的对象。")

    try:
        questions = [BenchmarkQuestion.model_validate(item) for item in raw_questions]
    except Exception as exc:  # pydantic's error is useful to callers as context
        raise SnapshotError(f"问题集字段不符合要求：{exc}") from exc
    ids = [item.id for item in questions]
    if len(ids) != len(set(ids)):
        raise SnapshotError("问题集 id 必须唯一。")
    return questions


def _pmid_from_document(document: MedicalDocument) -> str:
    pmid = document.metadata.get("pmid")
    if pmid:
        return str(pmid)
    if document.id.startswith("pubmed:"):
        return document.id.split(":", 1)[1]
    raise SnapshotError(f"文献缺少 PMID：{document.id}")


def _to_medical_document(value: MedicalDocument | Mapping[str, Any]) -> MedicalDocument:
    if isinstance(value, MedicalDocument):
        return value
    payload = dict(value)
    if "abstract" not in payload and "content" in payload:
        payload["abstract"] = payload.pop("content")
    pmid = payload.get("pmid") or payload.get("metadata", {}).get("pmid")
    if pmid and not payload.get("id"):
        payload["id"] = f"pubmed:{pmid}"
    payload.setdefault("source", "pubmed")
    payload.setdefault("title", f"PubMed article {pmid or 'unknown'}")
    payload.setdefault("abstract", "")
    payload.setdefault("metadata", {})
    payload["metadata"] = dict(payload["metadata"])
    if pmid:
        payload["metadata"].setdefault("pmid", str(pmid))
    try:
        return MedicalDocument.model_validate(payload)
    except Exception as exc:
        raise SnapshotError(f"文献记录字段不符合要求：{exc}") from exc


def build_snapshot(
    questions: Iterable[BenchmarkQuestion],
    records_by_question: Mapping[str, Iterable[MedicalDocument | Mapping[str, Any]]],
    *,
    candidate_limit: int = 20,
    fetched_at: str | None = None,
    corpus_version: str = CORPUS_VERSION,
    annotation_version: str = ANNOTATION_VERSION,
) -> SnapshotBundle:
    """Build a deterministic union snapshot and a pending-label template."""

    if candidate_limit < 1:
        raise SnapshotError("candidate_limit 必须大于等于 1。")
    question_list = list(questions)
    question_ids = [question.id for question in question_list]
    if len(question_ids) != len(set(question_ids)):
        raise SnapshotError("问题集 id 必须唯一。")

    timestamp = fetched_at or utc_now_iso()
    documents_by_pmid: dict[str, FrozenDocument] = {}
    memberships: dict[str, set[str]] = {}

    for question in question_list:
        records = list(records_by_question.get(question.id, []))[:candidate_limit]
        for raw in records:
            document = _to_medical_document(raw)
            pmid = _pmid_from_document(document)
            current = FrozenDocument(
                pmid=pmid,
                title=_normalise_text(document.title) or f"PubMed article {pmid}",
                abstract=_normalise_text(document.abstract),
                authors=list(document.authors),
                year=document.year,
                doi=document.doi,
                journal=document.journal,
                source_url=document.url or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                fetched_at=timestamp,
                content_sha256=content_sha256(document.title, document.abstract),
                corpus_version=corpus_version,
                candidate_question_ids=[],
            )
            previous = documents_by_pmid.get(pmid)
            if previous and previous.content_sha256 != current.content_sha256:
                raise SnapshotError(f"同一 PMID 出现不同标题/摘要：{pmid}")
            if previous is None:
                documents_by_pmid[pmid] = current
            memberships.setdefault(pmid, set()).add(question.id)

    documents: list[FrozenDocument] = []
    for pmid in sorted(documents_by_pmid):
        document = documents_by_pmid[pmid]
        documents.append(
            document.model_copy(
                update={"candidate_question_ids": sorted(memberships.get(pmid, set()))}
            )
        )

    labels = [
        RelevanceLabel(
            question_id=question_id,
            pmid=pmid,
            relevance=None,
            coverage_scope="candidate_pool_only",
            annotation_version=annotation_version,
        )
        for question_id in question_ids
        for pmid in sorted(pmid for pmid, ids in memberships.items() if question_id in ids)
    ]
    manifest = SnapshotManifest(
        corpus_version=corpus_version,
        annotation_version=annotation_version,
        created_at=timestamp,
        fetched_at=timestamp,
        question_count=len(question_list),
        question_ids=question_ids,
        candidate_limit_per_question=candidate_limit,
        document_count=len(documents),
        coverage_scope="candidate_pool_only",
    )
    return SnapshotBundle(
        manifest=manifest,
        questions=question_list,
        documents=documents,
        labels=LabelsFile(
            annotation_version=annotation_version,
            corpus_version=corpus_version,
            coverage_scope="candidate_pool_only",
            labels=labels,
        ),
    )


def write_snapshot(bundle: SnapshotBundle, output_dir: Path) -> None:
    """Write manifest, questions, documents and pending labels atomically enough for a CLI."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(bundle.manifest.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "questions.json").write_text(
        json.dumps(
            {
                "question_set_version": "questions-v1",
                "questions": [item.model_dump(mode="json") for item in bundle.questions],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "documents.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for document in bundle.documents:
            handle.write(json.dumps(document.model_dump(mode="json"), ensure_ascii=False) + "\n")
    (output_dir / "labels_v1.json").write_text(
        json.dumps(bundle.labels.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_snapshot(snapshot_dir: Path) -> SnapshotBundle:
    """Load and validate a previously written snapshot."""

    try:
        manifest = SnapshotManifest.model_validate_json(
            (snapshot_dir / "manifest.json").read_text(encoding="utf-8")
        )
        questions = load_questions(snapshot_dir / "questions.json")
        documents = [
            FrozenDocument.model_validate(json.loads(line))
            for line in (snapshot_dir / "documents.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        labels = LabelsFile.model_validate_json(
            (snapshot_dir / "labels_v1.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SnapshotError(f"无法读取或校验语料快照：{snapshot_dir}") from exc

    if manifest.document_count != len(documents):
        raise SnapshotError("manifest.document_count 与 documents.jsonl 数量不一致。")
    if labels.corpus_version != manifest.corpus_version:
        raise SnapshotError("标签文件与语料快照的 corpus_version 不一致。")
    if labels.annotation_version != manifest.annotation_version:
        raise SnapshotError("标签文件与 manifest 的 annotation_version 不一致。")
    return SnapshotBundle(
        manifest=manifest,
        questions=questions,
        documents=documents,
        labels=labels,
    )


def save_labels(labels_file: LabelsFile, path: Path) -> None:
    """Persist manually edited labels without changing the frozen documents."""

    path.write_text(
        json.dumps(labels_file.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
