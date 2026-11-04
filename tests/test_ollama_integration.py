import os

import pytest

from app.evidence_gate import evaluate_evidence
from app.grounded_llm import GroundedAnswerer, OpenAICompatibleProvider
from app.llm_cache import LLMResponseCache
from app.medical_models import MedicalDocument, RetrievalHit


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_INTEGRATION") != "1",
    reason="set RUN_OLLAMA_INTEGRATION=1 to call the local Ollama service",
)
def test_local_ollama_returns_a_traceable_grounded_answer():
    hit = RetrievalHit(
        document=MedicalDocument(
            id="pubmed:123456",
            source="pubmed",
            title="Example immunotherapy study",
            abstract=(
                "In this fictional test abstract, immunotherapy improved progression-free "
                "survival compared with the control group. No overall survival result was reported."
            ),
            url="https://pubmed.ncbi.nlm.nih.gov/123456/",
            metadata={"pmid": "123456"},
        ),
        score=1.0,
        retrieval_method="integration_fixture",
    )
    decision = evaluate_evidence([hit])
    provider = OpenAICompatibleProvider(
        api_key="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5:3b",
        timeout=180,
        max_retries=1,
        max_tokens=128,
        cache=LLMResponseCache(enabled=False),
    )

    answer = GroundedAnswerer(provider, max_citation_repairs=1).answer(
        "What outcome did the abstract report?",
        [hit],
        evidence_records=decision.records,
    )

    assert answer.abstained is False
    assert answer.model is not None
    assert answer.citations == [1]
    assert answer.citation_bindings[0].pmid == "123456"
    assert answer.citation_bindings[0].source_section == "abstract"
    assert answer.estimated_cost_usd == 0
