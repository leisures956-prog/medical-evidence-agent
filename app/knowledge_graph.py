"""Lightweight, provenance-preserving graph built from structured evidence.

The graph deliberately consumes only :class:`StructuredEvidence` records. It
does not infer entities from free text and it does not call an LLM. Every
non-paper edge points back to a PMID, the title/abstract section, and the exact
evidence span that justified the relation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.medical_models import EvidenceSection, EvidenceSpan, StructuredEvidence


GraphEntityType = Literal["Paper", "Method", "Disease", "Dataset"]
GraphRelation = Literal["uses_method", "studies_disease", "uses_dataset"]


class KnowledgeGraphError(ValueError):
    """Raised when structured evidence cannot be represented safely."""


class KnowledgeGraphNode(BaseModel):
    """One graph entity.

    Paper nodes keep their PMID and source URL. Method, disease, and dataset
    nodes are normalized labels shared by papers that mention the same label.
    """

    id: str = Field(min_length=1)
    type: GraphEntityType
    label: str = Field(min_length=1)
    pmid: str | None = None
    source_url: str | None = None


class KnowledgeGraphEdge(BaseModel):
    """A relation with the exact source fragment that supports it."""

    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    relation: GraphRelation
    pmid: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    source_section: EvidenceSection
    evidence_text: str = Field(min_length=1)
    source_url: str = Field(min_length=1)


class EvidenceKnowledgeGraph(BaseModel):
    """JSON-serializable graph generated only from structured evidence."""

    graph_type: Literal["summary_evidence_graph"] = "summary_evidence_graph"
    source: Literal["structured_evidence"] = "structured_evidence"
    nodes: list[KnowledgeGraphNode] = Field(default_factory=list)
    edges: list[KnowledgeGraphEdge] = Field(default_factory=list)

    @property
    def paper_count(self) -> int:
        return sum(node.type == "Paper" for node in self.nodes)

    @property
    def relation_count(self) -> int:
        return len(self.edges)


_FIELD_SPECS: tuple[tuple[str, GraphEntityType, GraphRelation], ...] = (
    ("method", "Method", "uses_method"),
    ("disease", "Disease", "studies_disease"),
    ("dataset", "Dataset", "uses_dataset"),
)


def _normalized_label(value: str) -> str:
    return " ".join(value.split()).casefold()


def _entity_id(entity_type: GraphEntityType, label: str) -> str:
    """Return a readable, deterministic ID without relying on a database."""

    normalized = _normalized_label(label)
    slug = re.sub(r"[^0-9a-z]+", "-", normalized).strip("-")
    if not slug:
        slug = "entity"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{entity_type.lower()}:{slug}-{digest}"


def _paper_id(pmid: str) -> str:
    return f"paper:{pmid}"


def _field_values(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _validated_span_map(record: StructuredEvidence) -> dict[str, EvidenceSpan]:
    spans = {span.evidence_id: span for span in record.evidence_spans}
    if len(spans) != len(record.evidence_spans):
        raise KnowledgeGraphError(f"PMID {record.pmid} 的 evidence_id 不唯一。")
    return spans


def _add_node(
    nodes: dict[str, KnowledgeGraphNode],
    *,
    entity_type: GraphEntityType,
    label: str,
    pmid: str | None = None,
    source_url: str | None = None,
) -> str:
    node_id = (
        _paper_id(pmid)
        if entity_type == "Paper" and pmid
        else _entity_id(entity_type, label)
    )
    if node_id not in nodes:
        nodes[node_id] = KnowledgeGraphNode(
            id=node_id,
            type=entity_type,
            label=label,
            pmid=pmid,
            source_url=source_url,
        )
    return node_id


def build_evidence_graph(records: Iterable[StructuredEvidence]) -> EvidenceKnowledgeGraph:
    """Build a small graph while preserving provenance for every relation.

    Only ``reported`` values with evidence IDs become Method, Disease, or
    Dataset nodes. Null fields and fields marked as not reported are omitted.
    If an evidence ID cannot be found, this function raises instead of silently
    creating an unsupported graph edge.
    """

    nodes: dict[str, KnowledgeGraphNode] = {}
    edges: dict[str, KnowledgeGraphEdge] = {}

    for record in records:
        spans_by_id = _validated_span_map(record)
        paper_id = _add_node(
            nodes,
            entity_type="Paper",
            label=record.title,
            pmid=record.pmid,
            source_url=record.source_url,
        )

        for field_name, entity_type, relation in _FIELD_SPECS:
            field = getattr(record, field_name)
            if field.status != "reported" or field.value is None:
                continue
            if not field.evidence_ids:
                raise KnowledgeGraphError(
                    f"PMID {record.pmid} 的 {field_name} 缺少 evidence_ids。"
                )

            labels = _field_values(field.value)
            for label in labels:
                target_id = _add_node(
                    nodes,
                    entity_type=entity_type,
                    label=label,
                )
                for evidence_id in field.evidence_ids:
                    span = spans_by_id.get(evidence_id)
                    if span is None:
                        raise KnowledgeGraphError(
                            f"PMID {record.pmid} 的 {field_name} 引用了不存在的 {evidence_id}。"
                        )
                    if span.pmid != record.pmid:
                        raise KnowledgeGraphError(
                            f"PMID {record.pmid} 的 {evidence_id} PMID 不匹配。"
                        )
                    if span.source_url != record.source_url:
                        raise KnowledgeGraphError(
                            f"PMID {record.pmid} 的 {evidence_id} source_url 不匹配。"
                        )
                    edge_key = "|".join((paper_id, target_id, relation, evidence_id))
                    edge_id = "edge:" + hashlib.sha1(edge_key.encode("utf-8")).hexdigest()[:16]
                    edges.setdefault(
                        edge_id,
                        KnowledgeGraphEdge(
                            id=edge_id,
                            source=paper_id,
                            target=target_id,
                            relation=relation,
                            pmid=record.pmid,
                            evidence_id=evidence_id,
                            source_section=span.source_section,
                            evidence_text=span.exact_text,
                            source_url=span.source_url,
                        ),
                    )

    return EvidenceKnowledgeGraph(
        nodes=list(nodes.values()),
        edges=list(edges.values()),
    )


def write_graph_json(
    graph: EvidenceKnowledgeGraph,
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """Write a graph artifact to JSON and return the output path."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(graph.model_dump(mode="json"), ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )
    return output_path


__all__ = [
    "EvidenceKnowledgeGraph",
    "KnowledgeGraphEdge",
    "KnowledgeGraphError",
    "KnowledgeGraphNode",
    "build_evidence_graph",
    "write_graph_json",
]
