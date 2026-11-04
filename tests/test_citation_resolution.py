from app.citation_resolution import (
    attach_unique_evidence_to_uncited_claims,
    resolve_plain_citations,
)
from app.medical_models import EvidenceSpan, EvidenceValue, StructuredEvidence


def record(spans):
    empty = EvidenceValue(value=None, status="not_reported_in_abstract", evidence_ids=[])
    return StructuredEvidence(
        pmid="123456",
        source_url="https://pubmed.ncbi.nlm.nih.gov/123456/",
        title="Example",
        title_evidence_ids=["title-1"],
        research_object=empty,
        disease=empty,
        method=empty,
        dataset=empty,
        main_findings=empty,
        limitations=empty,
        evidence_spans=[
            EvidenceSpan(
                evidence_id="title-1",
                pmid="123456",
                source_section="title",
                exact_text="Example",
                source_url="https://pubmed.ncbi.nlm.nih.gov/123456/",
            ),
            *[
                EvidenceSpan(
                    evidence_id=f"abstract-{index}",
                    pmid="123456",
                    source_section="abstract",
                    exact_text=text,
                    source_url="https://pubmed.ncbi.nlm.nih.gov/123456/",
                )
                for index, text in enumerate(spans, start=1)
            ],
        ],
    )


def test_plain_citation_is_resolved_to_unique_matching_sentence():
    evidence = record([
        "PCT was elevated in neonatal sepsis.",
        "Lactate predicted PICU admission in children.",
    ])

    answer, notes = resolve_plain_citations(
        "Lactate was associated with PICU admission [1].",
        [evidence],
    )

    assert answer == "Lactate was associated with PICU admission [1:abstract-2]."
    assert notes


def test_ambiguous_plain_citation_is_not_guessed():
    evidence = record([
        "PCT was elevated in sepsis.",
        "PCT may be useful in sepsis.",
    ])

    answer, notes = resolve_plain_citations("PCT was reported in sepsis [1].", [evidence])

    assert answer == "PCT was reported in sepsis [1]."
    assert notes == []


def test_single_abstract_span_can_be_resolved_without_guessing():
    evidence = record(["Only one abstract sentence is available."])

    answer, _ = resolve_plain_citations("One sentence is available [1].", [evidence])

    assert answer.endswith("[1:abstract-1].")


def test_uncited_claim_gets_unique_sentence_level_evidence():
    evidence = record([
        "Progression-free survival improved compared with control.",
        "Overall survival was not reported.",
    ])

    answer, notes = attach_unique_evidence_to_uncited_claims(
        "Progression-free survival improved compared with control.",
        [evidence],
    )

    assert answer == "Progression-free survival improved compared with control [1:abstract-1]."
    assert notes


def test_uncited_claim_is_left_unchanged_when_match_is_ambiguous():
    evidence = record([
        "PCT was elevated in sepsis.",
        "PCT may be useful in sepsis.",
    ])

    answer, notes = attach_unique_evidence_to_uncited_claims(
        "PCT was reported in sepsis.",
        [evidence],
    )

    assert answer == "PCT was reported in sepsis."
    assert notes == []


def test_one_claim_can_bind_two_supporting_sentences_in_same_paper():
    evidence = record([
        "PCT was elevated in early-onset neonatal sepsis.",
        "IL-18 was also elevated in early-onset neonatal sepsis.",
    ])

    answer, notes = attach_unique_evidence_to_uncited_claims(
        "PCT and IL-18 were elevated in early-onset neonatal sepsis.",
        [evidence],
    )

    assert "[1:abstract-1]" in answer
    assert "[1:abstract-2]" in answer
    assert notes


def test_uncited_tail_without_sentence_punctuation_gets_evidence():
    evidence = record(["PICU admission had AUC = 0.641."])

    answer, notes = attach_unique_evidence_to_uncited_claims(
        "PICU admission had AUC = 0.641",
        [evidence],
    )

    assert answer.endswith("[1:abstract-1]")
    assert notes


def test_numeric_claim_prefers_span_containing_the_claimed_number():
    evidence = record([
        "A total of 149 pediatric patients were included, of whom 90 underwent testing.",
        "The study included children presenting to the emergency department.",
    ])

    answer, _ = attach_unique_evidence_to_uncited_claims(
        "149 children were included.",
        [evidence],
    )

    assert answer == "149 children were included [1:abstract-1]."
