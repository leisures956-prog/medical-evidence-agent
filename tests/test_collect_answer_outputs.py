import json
from pathlib import Path

import pytest

from app.answer_evaluation import load_answer_evaluation_cases
from app.grounded_llm import GroundedAnswerer
from scripts.collect_answer_outputs import (
    SNAPSHOT, checked_output, collect_one, load_contexts,
)


def test_frozen_context_and_reference_spans_match():
    cases = load_answer_evaluation_cases()
    contexts, digest = load_contexts(cases, SNAPSHOT)
    assert len(contexts) == 20
    assert all(c.corpus_sha256 == digest for c in cases)
    assert len(contexts['a01']) == 1
    assert not contexts['a20']
    for case in cases:
        for claim in case.expected_claims:
            assert claim.exact_text
            assert claim.allowed_pmids
            assert any(ref['exact_text'] == claim.exact_text for ref in case.reference_spans)


def test_changed_corpus_hash_is_rejected(tmp_path):
    path = tmp_path / 'changed.jsonl'
    path.write_text('{}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_contexts(load_answer_evaluation_cases(), path)


def test_empty_context_abstains_without_model_call():
    class NeverCall:
        def answer(self, *args, **kwargs):
            raise AssertionError('Must not call model')
    case = load_answer_evaluation_cases()[-1]
    result = collect_one(case, [], NeverCall())
    assert result['abstained'] is True
    assert result['abstention_reason'] == 'no_retrieval_results'
    assert 'error' not in result


def test_generation_uses_existing_answerer_without_gold_leakage():
    class Provider:
        def complete(self, messages):
            text = json.dumps(messages)
            assert 'support_terms' not in text
            assert 'expected_behavior' not in text
            return 'The retrospective study included children aged 3 months to 14 years [1:abstract-5].'
    case = load_answer_evaluation_cases()[0]
    contexts, _ = load_contexts([case], SNAPSHOT)
    result = collect_one(case, contexts[case.id], GroundedAnswerer(Provider()))
    assert 'error' not in result
    assert result['citation_bindings'][0]['pmid'] == '42856985'


def test_generation_error_is_saved_not_disguised_as_refusal():
    class Broken:
        def answer(self, *args, **kwargs):
            raise RuntimeError('model unavailable')
    case = load_answer_evaluation_cases()[0]
    contexts, _ = load_contexts([case], SNAPSHOT)
    result = collect_one(case, contexts[case.id], Broken())
    assert 'model unavailable' in result['error']
    assert result['abstained'] is False


def test_c_drive_output_is_rejected():
    with pytest.raises(ValueError, match='D-drive project'):
        checked_output(Path('C:/not-allowed/output.json'))
