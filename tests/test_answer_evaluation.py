from pathlib import Path

from app.answer_evaluation import (
    AnswerEvaluationCase,
    AnswerObservation,
    ExpectedClaim,
    evaluate_answer_case,
    load_answer_evaluation_cases,
    summarize_answer_evaluation,
)
from app.medical_models import CitationBinding


def _binding(text: str, *, source_section: str = "abstract") -> CitationBinding:
    return CitationBinding(
        citation_number=1,
        pmid="123456",
        evidence_id="abstract-1" if source_section == "abstract" else "title-1",
        source_section=source_section,
        exact_text=text,
        source_url="https://pubmed.ncbi.nlm.nih.gov/123456/",
    )


def _answer_case() -> AnswerEvaluationCase:
    return AnswerEvaluationCase(
        id="a01",
        question="What did the study report about lactate and PICU admission?",
        scenario="answerable",
        expected_behavior="answer",
        expected_claims=[
            ExpectedClaim(id="c1", support_terms=["lactate", "PICU"]),
        ],
    )


def test_supported_citation_gets_high_precision_recall_and_faithfulness():
    result = evaluate_answer_case(
        _answer_case(),
        AnswerObservation(
            answer="Elevated lactate was associated with PICU admission [1:abstract-1].",
            citations=[1],
            citation_bindings=[
                _binding("Elevated lactate was significantly associated with PICU admission.")
            ],
        ),
    )

    assert result.citation_precision == 1.0
    assert result.citation_recall == 1.0
    assert result.answer_faithfulness == 1.0
    assert result.unsupported_claim_count == 0
    assert result.abstention_correct is True


def test_mismatched_span_is_not_counted_as_semantic_support():
    result = evaluate_answer_case(
        _answer_case(),
        AnswerObservation(
            answer="Elevated lactate was associated with PICU admission [1:abstract-1].",
            citations=[1],
            citation_bindings=[_binding("The study measured inflammatory cytokines.")],
        ),
    )

    assert result.citation_precision == 0.0
    assert result.citation_recall == 0.0
    assert result.unsupported_claim_count == 1
    assert result.citation_traceable is True


def test_association_evidence_flags_causal_or_benefit_overclaim():
    case = AnswerEvaluationCase(
        id="a02",
        question="What was observed?",
        scenario="answerable",
        expected_behavior="answer",
        expected_claims=[ExpectedClaim(id="c1", support_terms=["treatment", "prognosis"])],
    )
    result = evaluate_answer_case(
        case,
        AnswerObservation(
            answer="The treatment improved prognosis [1:abstract-1].",
            citations=[1],
            citation_bindings=[
                _binding("Treatment was associated with prognosis in an observational cohort.")
            ],
        ),
    )

    assert result.scope_warnings
    assert result.answer_faithfulness < 1.0
    assert result.unsupported_claim_count >= 1


def test_expected_abstention_is_scored_correctly():
    case = AnswerEvaluationCase(
        id="a03",
        question="What is the exact unpublished dose?",
        scenario="no_answer",
        expected_behavior="abstain",
        expected_abstention_reasons=["no_retrieval_results"],
    )
    result = evaluate_answer_case(
        case,
        AnswerObservation(
            answer="当前固定语料没有足够证据回答该问题。",
            abstained=True,
            abstention_reason="no_retrieval_results",
        ),
    )

    assert result.abstention_correct is True
    assert result.citation_recall == 0.0
    assert result.unsupported_claim_count == 0


def test_dataset_has_twenty_fixed_cases_and_expected_scenarios():
    cases = load_answer_evaluation_cases()

    assert len(cases) == 20
    assert len({case.id for case in cases}) == 20
    assert sum(case.expected_behavior == "answer" for case in cases) == 12
    assert sum(case.expected_behavior == "abstain" for case in cases) == 8


def test_summary_reports_all_answer_quality_metrics():
    case = _answer_case()
    observation = AnswerObservation(
        answer="Elevated lactate was associated with PICU admission [1:abstract-1].",
        citations=[1],
        citation_bindings=[
            _binding("Elevated lactate was significantly associated with PICU admission.")
        ],
    )
    summary = summarize_answer_evaluation([case], {case.id: observation})

    assert summary.total_cases == 1
    assert summary.citation_precision == 1.0
    assert summary.citation_recall == 1.0
    assert summary.abstention_accuracy == 1.0
    assert summary.metric_note


def test_explicit_unknown_span_does_not_fall_back_to_other_span():
    result = evaluate_answer_case(_answer_case(), {
        'answer': 'Lactate was associated with PICU admission [1:missing].',
        'citation_bindings': [_binding('Lactate was associated with PICU admission.').model_dump()],
    })
    assert result.citation_traceable is False
    assert result.citation_precision == 0


def test_missing_response_cannot_pass_answer_decision():
    summary = summarize_answer_evaluation([_answer_case()], {})
    assert summary.abstention_accuracy == 0
    assert summary.cases[0].errors == ['missing_observation']


def test_refusal_does_not_inflate_citation_metrics():
    case = AnswerEvaluationCase(id='refuse', question='No evidence?',
        scenario='no_answer', expected_behavior='abstain')
    summary = summarize_answer_evaluation([case], {
        'refuse': {'answer': 'Insufficient evidence.', 'abstained': True}
    })
    assert summary.abstention_accuracy == 1
    assert summary.citation_precision == 0
    assert summary.citation_coverage == 0


def test_multiple_citations_are_scored_individually():
    good = _binding('Lactate was associated with PICU admission.')
    bad = good.model_copy(update={'citation_number': 2, 'pmid': '999',
                                 'exact_text': 'Unrelated molecular investigation.'})
    result = evaluate_answer_case(_answer_case(), {
        'answer': 'Lactate was associated with PICU admission [1:abstract-1][2:abstract-1].',
        'citation_bindings': [good.model_dump(), bad.model_dump()],
    })
    assert result.citation_precision == 0.5


def test_wrong_pmid_cannot_cover_expected_claim():
    case = _answer_case()
    case.expected_claims[0].allowed_pmids = ['correct-pmid']
    result = evaluate_answer_case(case, {
        'answer': 'Lactate was associated with PICU admission [1:abstract-1].',
        'citation_bindings': [_binding('Lactate was associated with PICU admission.').model_dump()],
    })
    assert result.citation_recall == 0


def test_expected_claim_can_be_supported_across_two_sentences():
    case = AnswerEvaluationCase(
        id="multi",
        question="design and age?",
        scenario="answerable",
        expected_behavior="answer",
        expected_claims=[
            ExpectedClaim(
                id="c",
                support_terms=["retrospective", "3 months", "14 years"],
            )
        ],
    )
    binding = _binding(
        "METHODS: This was a retrospective observational study of children aged "
        "3 months to 14 years."
    )
    result = evaluate_answer_case(case, {
        "answer": (
            "The design was retrospective [1:abstract-1]. "
            "The age range was 3 months to 14 years [1:abstract-1]."
        ),
        "citation_bindings": [binding.model_dump()],
    })
    assert result.citation_recall == 1


def test_missing_numeric_result_is_not_supported_by_a_related_span():
    case = AnswerEvaluationCase(
        id="numeric",
        question="AUC?",
        scenario="answerable",
        expected_behavior="answer",
        expected_claims=[ExpectedClaim(id="c", support_terms=["0.641"])],
    )
    result = evaluate_answer_case(case, {
        "answer": "The AUC was 0.641 [1:abstract-1].",
        "citation_bindings": [
            _binding(
                "The study aimed to evaluate predictive value for PICU admission."
            ).model_dump()
        ],
    })
    assert result.citation_precision == 0
    assert result.answer_faithfulness == 0


def test_no_auc_claim_is_not_supported_when_the_span_reports_an_auc_value():
    case = AnswerEvaluationCase(
        id="negative-number",
        question="AUC?",
        scenario="answerable",
        expected_behavior="answer",
        expected_claims=[ExpectedClaim(id="c", support_terms=["0.544"])],
    )
    result = evaluate_answer_case(case, {
        "answer": "No AUC value was reported for mortality [1:abstract-1].",
        "citation_bindings": [
            _binding(
                "No predictive value was observed for mortality (AUC = 0.544)."
            ).model_dump()
        ],
    })
    assert result.citation_precision == 0
    assert result.answer_faithfulness == 0


def test_no_predictive_value_with_reported_auc_is_supported():
    case = AnswerEvaluationCase(
        id="negative-predictive-value",
        question="What AUC values were reported?",
        scenario="answerable",
        expected_behavior="answer",
        expected_claims=[
            ExpectedClaim(id="c", support_terms=["0.641", "0.544"]),
        ],
    )
    result = evaluate_answer_case(case, {
        "answer": (
            "PICU admission had modest predictive performance (AUC = 0.641), "
            "whereas no predictive value was observed for mortality (AUC = 0.544) "
            "[1:abstract-1]."
        ),
        "citation_bindings": [
            _binding(
                "Receiver operating characteristic analysis demonstrated predictive "
                "performance for PICU admission (AUC = 0.641), whereas no predictive "
                "value was observed for mortality (AUC = 0.544)."
            ).model_dump()
        ],
    })
    assert result.citation_precision == 1
    assert result.citation_recall == 1
    assert result.answer_faithfulness == 1


def test_missing_observation_note_is_not_counted_as_unsupported_claim():
    case = AnswerEvaluationCase(
        id="missing-observation-note",
        question="Did the study investigate vaccine perceptions or measure clinical protection?",
        scenario="answerable",
        expected_behavior="answer",
        expected_claims=[
            ExpectedClaim(id="c", support_terms=["perceptions", "effectiveness"]),
        ],
    )
    result = evaluate_answer_case(case, {
        "answer": (
            "The study investigated perceptions of COVID-19 vaccine effectiveness "
            "among older adults [1:abstract-1]. "
            "The abstract does not report clinical-protection measurements."
        ),
        "citation_bindings": [
            _binding(
                "This study investigates perceptions of COVID-19 vaccine effectiveness "
                "among older adults by racial background."
            ).model_dump()
        ],
    })
    assert result.citation_precision == 1
    assert result.citation_coverage == 1
    assert result.answer_faithfulness == 1
    assert result.unsupported_claim_count == 0
    assert result.citation_recall == 1
