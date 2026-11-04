import pytest

from app.query_rewriter import MedicalQueryRewriter


def test_rewriter_keeps_original_and_adds_medical_variants():
    result = MedicalQueryRewriter(max_queries=4).rewrite(
        "lung cancer immunotherapy"
    )

    assert result.queries[0] == "lung cancer immunotherapy"
    assert len(result.queries) == 4
    assert any("non-small cell lung cancer" in item for item in result.queries)
    assert any("immune checkpoint inhibitors" in item for item in result.queries)


def test_rewriter_expands_acronym_without_duplicates():
    result = MedicalQueryRewriter().rewrite("NSCLC immunotherapy")

    assert len(result.queries) == len(set(item.casefold() for item in result.queries))
    assert any("non-small cell lung cancer" in item for item in result.queries)


def test_rewriter_rejects_blank_query():
    with pytest.raises(ValueError, match="不能为空"):
        MedicalQueryRewriter().rewrite("   ")
