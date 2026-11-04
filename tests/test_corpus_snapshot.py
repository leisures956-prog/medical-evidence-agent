import json
from pathlib import Path

import pytest

from app.corpus_snapshot import (
    BenchmarkQuestion,
    SnapshotError,
    build_snapshot,
    content_sha256,
    load_snapshot,
    write_snapshot,
)
from app.medical_models import MedicalDocument


def _document(pmid: str, title: str, abstract: str = "Abstract text") -> MedicalDocument:
    return MedicalDocument(
        id=f"pubmed:{pmid}",
        source="pubmed",
        title=title,
        abstract=abstract,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        metadata={"pmid": pmid},
    )


def test_content_hash_normalises_whitespace():
    assert content_sha256(" A\nB ", " C\tD ") == content_sha256("A B", "C D")


def test_build_and_roundtrip_snapshot(tmp_path: Path):
    questions = [
        BenchmarkQuestion(id="q01", question="lung cancer"),
        BenchmarkQuestion(id="q02", question="immunotherapy"),
    ]
    records = {
        "q01": [_document("1", "Lung cancer study"), _document("2", "Other study")],
        "q02": [_document("1", "Lung cancer study")],
    }
    bundle = build_snapshot(
        questions,
        records,
        candidate_limit=2,
        fetched_at="2026-10-29T00:00:00+00:00",
    )
    assert [document.pmid for document in bundle.documents] == ["1", "2"]
    assert bundle.documents[0].candidate_question_ids == ["q01", "q02"]
    assert len(bundle.labels.labels) == 3

    write_snapshot(bundle, tmp_path)
    loaded = load_snapshot(tmp_path)
    assert loaded.manifest.document_count == 2
    assert loaded.documents[0].content_sha256 == bundle.documents[0].content_sha256
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["coverage_scope"] == "candidate_pool_only"


def test_conflicting_duplicate_pmid_is_rejected():
    question = BenchmarkQuestion(id="q01", question="query")
    records = {
        "q01": [
            _document("1", "Title A"),
            _document("1", "Title B"),
        ]
    }
    with pytest.raises(SnapshotError, match="不同标题/摘要"):
        build_snapshot([question], records)
