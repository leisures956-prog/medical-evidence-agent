import json

import pytest
from fastapi.testclient import TestClient

from app.evidence_extractor import EvidenceExtractionError, EvidenceExtractor
from app.evidence_validator import EvidenceValidationError, validate_evidence
from app.main import app, get_search_service
from app.medical_models import (
    EvidenceValue,
    EvidenceSpan,
    MedicalDocument,
    RetrievalHit,
    StructuredEvidence,
)
from app.search import SearchService


pytestmark = pytest.mark.api_runtime


def make_document() -> MedicalDocument:
    return MedicalDocument(
        id="pubmed:123456",
        source="pubmed",
        title="Example medical evidence",
        abstract=(
            "Adults with lung cancer received a cohort study intervention. "
            "The study used a public clinical dataset. "
            "Improved survival was observed. "
            "The single-center design was a limitation."
        ),
        url="https://pubmed.ncbi.nlm.nih.gov/123456/",
        metadata={"pmid": "123456"},
    )


def empty_value(status="not_reported_in_abstract"):
    return EvidenceValue(value=None, status=status, evidence_ids=[])


def test_metadata_only_keeps_pmid_and_title_evidence():
    evidence = EvidenceExtractor().extract(make_document())

    assert evidence.pmid == "123456"
    assert evidence.title_evidence_ids == ["title-1"]
    assert evidence.evidence_spans[0].source_section == "title"
    assert evidence.evidence_spans[0].exact_text == "Example medical evidence"
    assert evidence.method.status == "not_reported_in_abstract"
    assert evidence.method.value is None


def test_metadata_only_marks_missing_abstract():
    document = make_document()
    document.abstract = ""

    evidence = EvidenceExtractor().extract(document)

    assert evidence.method.status == "abstract_unavailable"


def test_validator_accepts_exact_abstract_span():
    document = make_document()
    sentence = "Improved survival was observed."
    evidence = StructuredEvidence(
        pmid="123456",
        source_url=document.url,
        title=document.title,
        title_evidence_ids=["title-1"],
        research_object=empty_value(),
        disease=EvidenceValue(value="lung cancer", status="reported", evidence_ids=["disease-1"]),
        method=empty_value(),
        dataset=empty_value(),
        main_findings=EvidenceValue(value="Improved survival", status="reported", evidence_ids=["finding-1"]),
        limitations=empty_value(),
        evidence_spans=[
            EvidenceSpan(
                evidence_id="title-1",
                pmid="123456",
                source_section="title",
                exact_text=document.title,
                source_url=document.url,
            ),
            EvidenceSpan(
                evidence_id="disease-1",
                pmid="123456",
                source_section="abstract",
                exact_text="Adults with lung cancer",
                source_url=document.url,
            ),
            EvidenceSpan(
                evidence_id="finding-1",
                pmid="123456",
                source_section="abstract",
                exact_text=sentence,
                source_url=document.url,
            ),
        ],
    )

    assert validate_evidence(document, evidence).pmid == "123456"


def test_validator_rejects_fabricated_span():
    document = make_document()
    evidence = StructuredEvidence(
        pmid="123456",
        source_url=document.url,
        title=document.title,
        title_evidence_ids=["title-1"],
        research_object=empty_value(),
        disease=empty_value(),
        method=EvidenceValue(value="A fabricated method", status="reported", evidence_ids=["fake-1"]),
        dataset=empty_value(),
        main_findings=empty_value(),
        limitations=empty_value(),
        evidence_spans=[
            EvidenceSpan(
                evidence_id="title-1",
                pmid="123456",
                source_section="title",
                exact_text=document.title,
                source_url=document.url,
            ),
            EvidenceSpan(
                evidence_id="fake-1",
                pmid="123456",
                source_section="abstract",
                exact_text="This sentence is not in the abstract.",
                source_url=document.url,
            ),
        ],
    )

    with pytest.raises(EvidenceValidationError, match="原文片段"):
        validate_evidence(document, evidence)


class ValidProvider:
    def complete(self, messages):
        document = make_document()
        payload = {
            "pmid": "123456",
            "source_url": document.url,
            "title": document.title,
            "title_evidence_ids": ["title-1"],
            "research_object": {
                "value": "Adults with lung cancer",
                "status": "reported",
                "evidence_ids": ["object-1"],
            },
            "disease": {
                "value": "lung cancer",
                "status": "reported",
                "evidence_ids": ["disease-1"],
            },
            "method": {
                "value": "cohort study",
                "status": "reported",
                "evidence_ids": ["method-1"],
            },
            "dataset": {
                "value": "a public clinical dataset",
                "status": "reported",
                "evidence_ids": ["dataset-1"],
            },
            "main_findings": {
                "value": "Improved survival",
                "status": "reported",
                "evidence_ids": ["finding-1"],
            },
            "limitations": {
                "value": "single-center design",
                "status": "reported",
                "evidence_ids": ["limitation-1"],
            },
            "evidence_spans": [
                {
                    "evidence_id": "title-1",
                    "pmid": "123456",
                    "source_section": "title",
                    "exact_text": document.title,
                    "source_url": document.url,
                },
                {
                    "evidence_id": "object-1",
                    "pmid": "123456",
                    "source_section": "abstract",
                    "exact_text": "Adults with lung cancer",
                    "source_url": document.url,
                },
                {
                    "evidence_id": "disease-1",
                    "pmid": "123456",
                    "source_section": "abstract",
                    "exact_text": "lung cancer",
                    "source_url": document.url,
                },
                {
                    "evidence_id": "method-1",
                    "pmid": "123456",
                    "source_section": "abstract",
                    "exact_text": "cohort study",
                    "source_url": document.url,
                },
                {
                    "evidence_id": "dataset-1",
                    "pmid": "123456",
                    "source_section": "abstract",
                    "exact_text": "a public clinical dataset",
                    "source_url": document.url,
                },
                {
                    "evidence_id": "finding-1",
                    "pmid": "123456",
                    "source_section": "abstract",
                    "exact_text": "Improved survival was observed.",
                    "source_url": document.url,
                },
                {
                    "evidence_id": "limitation-1",
                    "pmid": "123456",
                    "source_section": "abstract",
                    "exact_text": "The single-center design was a limitation.",
                    "source_url": document.url,
                },
            ],
        }
        return json.dumps(payload)


def test_strict_llm_requires_verbatim_spans():
    evidence = EvidenceExtractor(provider=ValidProvider()).extract(
        make_document(), mode="strict_llm"
    )

    assert evidence.method.value == "cohort study"
    assert evidence.method.evidence_ids == ["method-1"]


def test_strict_llm_rejects_invalid_json():
    class InvalidProvider:
        def complete(self, messages):
            return "not json"

    with pytest.raises(EvidenceExtractionError, match="合法"):
        EvidenceExtractor(provider=InvalidProvider()).extract(
            make_document(), mode="strict_llm"
        )


class FakePubMedHybrid:
    def search(self, query, candidate_limit, top_k):
        return [
            RetrievalHit(
                document=make_document(),
                score=0.91,
                retrieval_method="pubmed_hybrid",
            )
        ][:top_k]


def test_evidence_api_returns_metadata_only_records():
    service = SearchService(pubmed_hybrid_retriever=FakePubMedHybrid())
    app.dependency_overrides[get_search_service] = lambda: service
    try:
        response = TestClient(app).post(
            "/evidence",
            json={"question": "lung cancer", "top_k": 1, "candidate_limit": 20},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_method"] == "pubmed_hybrid"
    assert payload["extraction_mode"] == "metadata_only"
    assert payload["records"][0]["pmid"] == "123456"
    assert payload["records"][0]["evidence_spans"][0]["source_section"] == "title"
