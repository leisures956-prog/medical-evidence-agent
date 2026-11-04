from pathlib import Path

from app.corpus_snapshot import BenchmarkQuestion, LabelsFile, RelevanceLabel, build_snapshot, write_snapshot
from app.labeling import (
    annotation_counts,
    load_annotation_context,
    pending_label_indices,
    save_labels_atomic,
    set_relevance,
)
from app.medical_models import MedicalDocument


def _document(pmid: str, title: str) -> MedicalDocument:
    return MedicalDocument(
        id=f"pubmed:{pmid}",
        source="pubmed",
        title=title,
        abstract="Abstract",
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        metadata={"pmid": pmid},
    )


def test_set_relevance_and_counts():
    labels = LabelsFile(
        labels=[
            RelevanceLabel(question_id="q01", pmid="1"),
            RelevanceLabel(question_id="q01", pmid="2"),
        ]
    )
    labels = set_relevance(labels, 0, 2)
    assert labels.labels[0].relevance == 2
    assert annotation_counts(labels) == {
        "pending": 1,
        "relevance_0": 0,
        "relevance_1": 0,
        "relevance_2": 1,
        "annotated": 1,
        "total": 2,
    }


def test_resume_keeps_previous_annotations_and_writes_atomically(tmp_path: Path):
    question = BenchmarkQuestion(id="q01", question="lung cancer")
    bundle = build_snapshot([question], {"q01": [_document("1", "Lung cancer study")]})
    write_snapshot(bundle, tmp_path)

    output = tmp_path / "labels_v1_annotated.json"
    labels = bundle.labels.model_copy(
        update={"labels": [bundle.labels.labels[0].model_copy(update={"relevance": 2})]}
    )
    save_labels_atomic(labels, output)

    loaded_bundle, resumed, resumed_path = load_annotation_context(tmp_path)
    assert loaded_bundle.manifest.document_count == 1
    assert resumed_path == output
    assert resumed.labels[0].relevance == 2
    assert not list(tmp_path.glob(".labels_v1_annotated.json.*.tmp"))


def test_cli_context_defaults_to_non_destructive_output(tmp_path: Path):
    question = BenchmarkQuestion(id="q01", question="lung cancer")
    bundle = build_snapshot([question], {"q01": [_document("1", "Lung cancer study")]})
    write_snapshot(bundle, tmp_path)
    _, _, output = load_annotation_context(tmp_path)
    assert output.name == "labels_v1_annotated.json"
    assert (tmp_path / "labels_v1.json").exists()


def test_pending_label_indices_support_question_filter_and_cap():
    labels = LabelsFile(
        labels=[
            RelevanceLabel(question_id="q01", pmid="1"),
            RelevanceLabel(question_id="q02", pmid="2"),
            RelevanceLabel(question_id="q01", pmid="3"),
            RelevanceLabel(question_id="q03", pmid="4", relevance=1),
        ]
    )

    assert pending_label_indices(labels, question_ids=["q01"]) == [0, 2]
    assert pending_label_indices(labels, question_ids=["q01", "q02"], max_items=1) == [0]


def test_pending_label_indices_rejects_non_positive_cap():
    labels = LabelsFile(labels=[RelevanceLabel(question_id="q01", pmid="1")])

    try:
        pending_label_indices(labels, max_items=0)
    except ValueError as exc:
        assert "大于等于 1" in str(exc)
    else:
        raise AssertionError("非正 max_items 应该被拒绝")
