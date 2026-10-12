import pytest

from app.agent_nodes import citation_check_node
from app.citation_validation import validate_citations
from app.grounded_llm import GroundedAnswerer, LLMProviderError
from app.medical_models import (
    EvidenceSpan,
    EvidenceValue,
    MedicalDocument,
    RetrievalHit,
    StructuredEvidence,
)


def make_record(with_abstract_span: bool = False) -> StructuredEvidence:
    document_url = "https://pubmed.ncbi.nlm.nih.gov/123456/"
    spans = [
        EvidenceSpan(
            evidence_id="title-1",
            pmid="123456",
            source_section="title",
            exact_text="Example evidence",
            source_url=document_url,
        )
    ]
    main_findings = EvidenceValue(
        value=None,
        status="not_reported_in_abstract",
        evidence_ids=[],
    )
    if with_abstract_span:
        spans.append(
            EvidenceSpan(
                evidence_id="finding-1",
                pmid="123456",
                source_section="abstract",
                exact_text="The intervention improved survival.",
                source_url=document_url,
            )
        )
        main_findings = EvidenceValue(
            value="improved survival",
            status="reported",
            evidence_ids=["finding-1"],
        )

    empty = EvidenceValue(
        value=None,
        status="not_reported_in_abstract",
        evidence_ids=[],
    )
    return StructuredEvidence(
        pmid="123456",
        source_url=document_url,
        title="Example evidence",
        title_evidence_ids=["title-1"],
        research_object=empty,
        disease=empty,
        method=empty,
        dataset=empty,
        main_findings=main_findings,
        limitations=empty,
        evidence_spans=spans,
    )


def test_plain_citation_binds_the_only_evidence_span():
    validation = validate_citations("The title supports this claim [1].", [make_record()])

    assert validation.valid is True
    assert validation.citations == [1]
    assert validation.bindings[0].evidence_id == "title-1"
    assert validation.bindings[0].source_section == "title"


def test_multiple_spans_require_explicit_evidence_id():
    record = make_record(with_abstract_span=True)

    plain = validate_citations("The finding is supported [1].", [record])
    explicit = validate_citations("The finding is supported [1:finding-1].", [record])

    assert plain.valid is False
    assert "唯一定位" in plain.errors[0]
    assert explicit.valid is True
    assert explicit.bindings[0].exact_text == "The intervention improved survival."


def test_invalid_citation_and_evidence_id_are_rejected():
    validation = validate_citations(
        "Claim [2] and another claim [1:missing].", [make_record()]
    )

    assert validation.valid is False
    assert any("无效引用" in error for error in validation.errors)
    assert any("不存在的 evidence_id" in error for error in validation.errors)
    assert validation.bindings == []


def test_citation_check_node_exposes_traceable_bindings():
    state = {
        "answer": "The title supports this claim [1].",
        "evidence_records": [make_record()],
    }

    checked = citation_check_node(state)

    assert checked.get("errors", []) == []
    assert checked["citations"] == [1]
    assert checked["citation_bindings"][0].evidence_id == "title-1"


class Provider:
    def __init__(self, answer: str):
        self.answer_text = answer

    def complete(self, messages):
        return self.answer_text


def test_grounded_answerer_rejects_unbound_source_level_citation():
    hit = RetrievalHit(
        document=MedicalDocument(
            id="pubmed:123456",
            title="Example evidence",
            abstract="The intervention improved survival.",
            url="https://pubmed.ncbi.nlm.nih.gov/123456/",
            metadata={"pmid": "123456"},
        ),
        score=0.9,
        retrieval_method="pubmed_hybrid",
    )

    with pytest.raises(LLMProviderError, match="唯一定位"):
        GroundedAnswerer(Provider("The finding is supported [1].")).answer(
            "question",
            [hit],
            evidence_records=[make_record(with_abstract_span=True)],
        )


def test_grounded_answerer_returns_explicit_citation_binding():
    hit = RetrievalHit(
        document=MedicalDocument(id="pubmed:123456", title="Example evidence"),
        score=0.9,
        retrieval_method="pubmed_hybrid",
    )
    result = GroundedAnswerer(
        Provider("The finding is supported [1:finding-1].")
    ).answer(
        "question",
        [hit],
        evidence_records=[make_record(with_abstract_span=True)],
    )

    assert result.citations == [1]
    assert result.citation_bindings[0].evidence_id == "finding-1"
