from pathlib import Path

from app.corpus_snapshot import BenchmarkQuestion, build_snapshot, write_snapshot
from app.keyword_baseline import evaluate_snapshot, keyword_score, rank_documents
from app.medical_models import MedicalDocument


def _document(pmid: str, title: str, abstract: str = "") -> MedicalDocument:
    return MedicalDocument(
        id=f"pubmed:{pmid}",
        source="pubmed",
        title=title,
        abstract=abstract,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        metadata={"pmid": pmid},
    )


def test_keyword_ranking_is_deterministic_and_uses_overlap():
    question = BenchmarkQuestion(id="q01", question="lung cancer")
    documents = [
        _document("2", "Cancer overview"),
        _document("1", "Lung cancer treatment"),
    ]
    bundle = build_snapshot([question], {"q01": documents})
    assert keyword_score(question.question, bundle.documents[0]) >= 0
    ranked = rank_documents(question.question, bundle.documents, top_k=2)
    assert [item["pmid"] for item in ranked] == ["1", "2"]


def test_incomplete_labels_do_not_produce_overall_metrics():
    question = BenchmarkQuestion(id="q01", question="lung cancer")
    bundle = build_snapshot(
        [question],
        {"q01": [_document("1", "Lung cancer treatment"), _document("2", "Cancer overview")]},
    )
    result = evaluate_snapshot(bundle)
    assert result["completed_cases"] == 0
    assert result["incomplete_cases"] == 1
    assert result["metrics"]["mrr"] is None
    assert result["cases"][0]["status"] == "incomplete_labels"


def test_complete_labels_calculate_recall_mrr_and_ndcg(tmp_path: Path):
    question = BenchmarkQuestion(id="q01", question="lung cancer")
    bundle = build_snapshot(
        [question],
        {"q01": [_document("1", "Lung cancer treatment"), _document("2", "Cancer overview")]},
    )
    bundle.labels.labels[0].relevance = 2
    bundle.labels.labels[1].relevance = 0
    result = evaluate_snapshot(bundle)
    assert result["completed_cases"] == 1
    assert result["metrics"]["recall"]["@1"] == 1.0
    assert result["metrics"]["mrr"] == 1.0
    assert result["metrics"]["ndcg"]["@1"] == 1.0

    write_snapshot(bundle, tmp_path)
    assert (tmp_path / "labels_v1.json").exists()
