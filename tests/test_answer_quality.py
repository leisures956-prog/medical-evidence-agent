from app.answer_quality import assess_answer_quality, is_unreported_answer
from app.medical_models import CitationBinding


def binding(section="abstract"):
    return CitationBinding(
        citation_number=1,
        pmid="123456",
        evidence_id="abstract-1" if section == "abstract" else "title-1",
        source_section=section,
        exact_text="Evidence text.",
        source_url="https://pubmed.ncbi.nlm.nih.gov/123456/",
    )


def test_every_claim_with_a_citation_passes_structural_quality_check():
    quality = assess_answer_quality(
        "PCT was elevated [1:abstract-1]. IL-18 was elevated [1:abstract-2].",
        [binding()],
    )

    assert quality.valid is True
    assert quality.claim_count == 2
    assert quality.citation_coverage == 1.0


def test_uncited_followup_claim_is_detected():
    quality = assess_answer_quality(
        "PCT was elevated [1:abstract-1]. This may support early risk assessment.",
        [binding()],
    )

    assert quality.valid is False
    assert quality.claim_count == 2
    assert quality.cited_claim_count == 1
    assert quality.uncited_claims == ["This may support early risk assessment."]


def test_placeholder_and_title_only_evidence_are_reported():
    quality = assess_answer_quality(
        "A specific clinical effect was reported [1:evidence_id].",
        [binding("title")],
    )

    assert quality.valid is False
    assert any("evidence_id" in warning for warning in quality.warnings)
    assert any("标题证据" in warning for warning in quality.warnings)


def test_not_reported_answer_is_classified_as_abstention_text():
    assert is_unreported_answer(
        "The study did not report five-year mortality outcomes [1:abstract-4]."
    )
    assert is_unreported_answer(
        "The study did not specify the antibiotic dose [1:abstract-5]."
    )
    assert not is_unreported_answer(
        "The study reported elevated lactate associated with PICU admission [1:abstract-9]."
    )


def test_abstract_not_reported_note_can_follow_a_cited_scope_statement():
    answer = (
        "The study investigated perceptions of COVID-19 vaccine effectiveness "
        "among older adults [1:abstract-4]. "
        "The abstract does not report a clinical-protection measurement."
    )
    evidence = CitationBinding(
        citation_number=1,
        pmid="123456",
        evidence_id="abstract-4",
        source_section="abstract",
        exact_text=(
            "This study investigates perceptions of COVID-19 vaccine effectiveness "
            "among older adults by racial background."
        ),
        source_url="https://pubmed.ncbi.nlm.nih.gov/123456/",
    )
    quality = assess_answer_quality(answer, [evidence])
    assert quality.valid
    assert quality.uncited_claims == []


def test_unsupported_measurement_negation_is_invalid_even_with_related_citation():
    evidence = CitationBinding(
        citation_number=1,
        pmid="123456",
        evidence_id="abstract-4",
        source_section="abstract",
        exact_text=(
            "This study investigates perceptions of COVID-19 vaccine effectiveness "
            "among older adults by racial background."
        ),
        source_url="https://pubmed.ncbi.nlm.nih.gov/123456/",
    )
    quality = assess_answer_quality(
        "The study investigated vaccine perceptions and did not measure clinical "
        "protection [1:abstract-4].",
        [evidence],
    )
    assert not quality.valid
    assert any("未测量" in warning for warning in quality.warnings)
