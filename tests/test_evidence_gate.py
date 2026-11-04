from app.evidence_gate import build_answer_evidence_records, evaluate_evidence
from app.medical_models import MedicalDocument, RetrievalHit


def make_hit(abstract: str) -> RetrievalHit:
    return RetrievalHit(
        document=MedicalDocument(
            id="pubmed:123456",
            source="pubmed",
            title="Example trial",
            abstract=abstract,
            url="https://pubmed.ncbi.nlm.nih.gov/123456/",
            metadata={"pmid": "123456"},
        ),
        score=0.9,
        retrieval_method="pubmed_hybrid",
    )


def test_no_hits_are_refused():
    decision = evaluate_evidence([])

    assert decision.allowed is False
    assert decision.reason == "no_retrieval_results"


def test_missing_abstract_is_refused_without_calling_a_model():
    decision = evaluate_evidence([make_hit("")])

    assert decision.allowed is False
    assert decision.reason == "abstract_unavailable"
    assert decision.records[0].evidence_spans[0].source_section == "title"


def test_usable_abstract_creates_verbatim_traceable_span():
    abstract = "The intervention improved progression-free survival."
    decision = evaluate_evidence([make_hit(abstract)])

    assert decision.allowed is True
    record = decision.records[0]
    assert record.pmid == "123456"
    assert record.evidence_spans[1].evidence_id == "abstract-1"
    assert record.evidence_spans[1].exact_text == abstract


def test_short_abstract_can_be_marked_insufficient():
    decision = evaluate_evidence(
        [make_hit("Short")],
        minimum_abstract_characters=20,
    )

    assert decision.allowed is False
    assert decision.reason == "insufficient_evidence"


def test_local_documents_get_stable_non_pubmed_source_url():
    hit = RetrievalHit(
        document=MedicalDocument(id="doc-1", title="Local", abstract="Local evidence text."),
        score=1.0,
        retrieval_method="local",
    )

    record = build_answer_evidence_records([hit])[0]

    assert record.pmid == "doc-1"
    assert record.source_url == "urn:document:doc-1"
