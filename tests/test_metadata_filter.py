import pytest
from pydantic import ValidationError

from app.main import SearchRequest
from app.medical_models import MedicalDocument, RetrievalHit
from app.metadata_filter import RetrievalFilters, filter_hits


def _hit(pmid: str, title: str, abstract: str, year: int | None):
    return RetrievalHit(
        document=MedicalDocument(
            id=f"pubmed:{pmid}",
            source="pubmed",
            title=title,
            abstract=abstract,
            year=year,
            metadata={"pmid": pmid},
        ),
        score=1.0,
        retrieval_method="test",
    )


def test_filters_report_each_first_exclusion_reason():
    hits = [
        _hit("1", "Clinical trial", "Results", 2024),
        _hit("2", "Old clinical trial", "Results", 2010),
        _hit("3", "No abstract", "", 2025),
        _hit("4", "Editorial: treatment update", "Comment", 2025),
    ]

    result = filter_hits(
        hits,
        RetrievalFilters(
            year_from=2020,
            year_to=2026,
            require_abstract=True,
            exclude_article_types=("editorial", "correction"),
        ),
    )

    assert [hit.document.id for hit in result.hits] == ["pubmed:1"]
    assert result.diagnostics.input_count == 4
    assert result.diagnostics.output_count == 1
    assert result.diagnostics.excluded_by_year == 1
    assert result.diagnostics.excluded_by_abstract == 1
    assert result.diagnostics.excluded_by_article_type == 1


def test_year_filter_excludes_unknown_year():
    result = filter_hits(
        [_hit("1", "Unknown year", "Results", None)],
        RetrievalFilters(year_from=2020),
    )

    assert not result.hits
    assert result.diagnostics.excluded_by_year == 1


def test_retrieval_filters_validate_year_range():
    with pytest.raises(ValueError, match="year_from"):
        RetrievalFilters(year_from=2026, year_to=2020)


def test_search_request_rejects_invalid_year_order():
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(
            {
                "query": "lung cancer",
                "source": "advanced_persisted",
                "filters": {"year_from": 2026, "year_to": 2020},
            }
        )
