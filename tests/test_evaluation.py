from app.evaluation import (
    EvaluationQuestion,
    load_evaluation_questions,
    run_fixed_evaluation,
)
from app.medical_models import MedicalDocument
from app.search import SearchHit


def test_fixed_evaluation_set_has_twelve_stable_questions():
    questions = load_evaluation_questions()

    assert len(questions) == 12
    assert len({question.id for question in questions}) == 12
    assert all(question.min_results >= 1 for question in questions)


class FakeEvaluationService:
    def search(self, query, limit, source, candidate_limit):
        document = MedicalDocument(
            id="pubmed:123456",
            source="pubmed",
            title=query,
            abstract=f"Abstract evidence for {query}.",
            url="https://pubmed.ncbi.nlm.nih.gov/123456/",
            metadata={"pmid": "123456"},
        )
        return [
            SearchHit(
                id=document.id,
                title=document.title,
                content=document.abstract,
                score=0.9,
                source="pubmed",
                url=document.url,
                authors=[],
                year=2026,
            )
        ][:limit], 1


def test_fixed_evaluation_reports_retrieval_json_citation_and_cost_metrics():
    questions = [
        EvaluationQuestion(
            id="q1",
            question="lung cancer immunotherapy",
            required_terms=["lung cancer", "immunotherapy"],
            min_results=1,
        ),
        EvaluationQuestion(
            id="q2",
            question="breast cancer screening",
            required_terms=["breast cancer"],
            min_results=1,
        ),
    ]

    summary = run_fixed_evaluation(
        FakeEvaluationService(),
        questions=questions,
        top_k=2,
        candidate_limit=20,
        mode="metadata_only",
    )

    assert summary.total_cases == 2
    assert summary.retrieval_success_rate == 1.0
    assert summary.topic_match_rate == 1.0
    assert summary.json_complete_rate == 1.0
    assert summary.citation_valid_rate == 1.0
    assert summary.abstract_evidence_coverage == 0.0
    assert summary.llm_calls == 0
    assert summary.estimated_cost_usd == 0.0
    assert summary.average_latency_ms >= 0
    assert all(case.errors == [] for case in summary.cases)
