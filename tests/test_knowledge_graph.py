import json

import pytest

from app.knowledge_graph import (
    KnowledgeGraphError,
    build_evidence_graph,
    write_graph_json,
)
from app.medical_models import EvidenceSpan, EvidenceValue, StructuredEvidence


def _reported(value, evidence_id):
    return EvidenceValue(value=value, status="reported", evidence_ids=[evidence_id])


def _empty():
    return EvidenceValue(value=None, status="not_reported_in_abstract", evidence_ids=[])


def _record(
    pmid="123",
    title="Immunotherapy in lung cancer",
    disease="lung cancer",
    method="cohort study",
    dataset="public clinical dataset",
):
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    spans = [
        EvidenceSpan(
            evidence_id="title-1",
            pmid=pmid,
            source_section="title",
            exact_text=title,
            source_url=url,
        )
    ]
    values = {}
    for field_name, value, prefix in (
        ("disease", disease, "disease"),
        ("method", method, "method"),
        ("dataset", dataset, "dataset"),
    ):
        if value is None:
            values[field_name] = _empty()
            continue
        evidence_id = f"{prefix}-1"
        spans.append(
            EvidenceSpan(
                evidence_id=evidence_id,
                pmid=pmid,
                source_section="abstract",
                exact_text=value,
                source_url=url,
            )
        )
        values[field_name] = _reported(value, evidence_id)

    return StructuredEvidence(
        pmid=pmid,
        source_url=url,
        title=title,
        title_evidence_ids=["title-1"],
        research_object=_empty(),
        disease=values["disease"],
        method=values["method"],
        dataset=values["dataset"],
        main_findings=_empty(),
        limitations=_empty(),
        evidence_spans=spans,
    )


def test_graph_contains_typed_nodes_and_provenance_edges():
    graph = build_evidence_graph([_record()])

    assert graph.paper_count == 1
    assert {node.type for node in graph.nodes} == {"Paper", "Disease", "Method", "Dataset"}
    assert graph.relation_count == 3
    assert {edge.relation for edge in graph.edges} == {
        "studies_disease",
        "uses_method",
        "uses_dataset",
    }
    for edge in graph.edges:
        assert edge.pmid == "123"
        assert edge.source_section == "abstract"
        assert edge.evidence_text
        assert edge.source == "paper:123"


def test_graph_deduplicates_shared_entity_labels_but_keeps_edges():
    graph = build_evidence_graph(
        [
            _record(pmid="123"),
            _record(pmid="456", title="Another lung cancer study"),
        ]
    )

    assert graph.paper_count == 2
    assert sum(node.type == "Disease" for node in graph.nodes) == 1
    assert graph.relation_count == 6
    assert {edge.pmid for edge in graph.edges} == {"123", "456"}


def test_null_or_unreported_fields_do_not_create_entities():
    graph = build_evidence_graph([_record(disease=None, method=None, dataset=None)])

    assert graph.paper_count == 1
    assert len(graph.nodes) == 1
    assert graph.edges == []


def test_graph_rejects_missing_provenance_instead_of_dropping_edge():
    record = _record()
    record.method = EvidenceValue(
        value="cohort study", status="reported", evidence_ids=["missing-1"]
    )

    with pytest.raises(KnowledgeGraphError, match="不存在"):
        build_evidence_graph([record])


def test_graph_json_export_is_reloadable(tmp_path):
    graph = build_evidence_graph([_record()])
    output = write_graph_json(graph, tmp_path / "graph.json")

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["source"] == "structured_evidence"
    assert len(payload["nodes"]) == 4
    assert payload["edges"][0]["evidence_id"]
