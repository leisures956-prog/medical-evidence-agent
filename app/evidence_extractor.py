from __future__ import annotations

import json
import re
from typing import Literal

from app.evidence_validator import EvidenceValidationError, validate_evidence
from app.grounded_llm import LLMProvider, LLMProviderError
from app.medical_models import (
    EvidenceValue,
    EvidenceSpan,
    MedicalDocument,
    StructuredEvidence,
)


EvidenceMode = Literal["metadata_only", "strict_llm"]


class EvidenceExtractionError(RuntimeError):
    """Raised when structured evidence cannot be safely produced."""


def _pmid_for(document: MedicalDocument) -> str:
    value = document.metadata.get("pmid")
    if value is not None:
        return str(value)
    if document.id.startswith("pubmed:"):
        return document.id.removeprefix("pubmed:")
    raise EvidenceExtractionError(f"文献 {document.id} 缺少 PMID，不能提取证据。")


def _url_for(document: MedicalDocument, pmid: str) -> str:
    return document.url or f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


class EvidenceExtractor:
    """Extract summary-level evidence while preserving exact source spans."""

    def __init__(self, provider: LLMProvider | None = None):
        self.provider = provider

    @staticmethod
    def _empty_value(document: MedicalDocument) -> EvidenceValue:
        status = "abstract_unavailable" if not document.abstract.strip() else "not_reported_in_abstract"
        return EvidenceValue(value=None, status=status, evidence_ids=[])

    def _metadata_only(self, document: MedicalDocument) -> StructuredEvidence:
        pmid = _pmid_for(document)
        source_url = _url_for(document, pmid)
        title_span = EvidenceSpan(
            evidence_id="title-1",
            pmid=pmid,
            source_section="title",
            exact_text=document.title,
            source_url=source_url,
        )
        evidence = StructuredEvidence(
            pmid=pmid,
            source_url=source_url,
            title=document.title,
            title_evidence_ids=[title_span.evidence_id],
            research_object=self._empty_value(document),
            disease=self._empty_value(document),
            method=self._empty_value(document),
            dataset=self._empty_value(document),
            main_findings=self._empty_value(document),
            limitations=self._empty_value(document),
            evidence_spans=[title_span],
        )
        try:
            return validate_evidence(document, evidence)
        except EvidenceValidationError as exc:
            raise EvidenceExtractionError(str(exc)) from exc

    @staticmethod
    def _prompt(document: MedicalDocument) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "你是摘要级医学证据整理器。只能使用用户提供的 PubMed 标题和摘要。"
                    "摘要没有提到的字段必须 value=null、status=not_reported_in_abstract；"
                    "没有摘要时使用 abstract_unavailable。禁止补造全文细节。"
                    "每个非空字段必须绑定 evidence_ids，并且 evidence_spans 中的 exact_text "
                    "必须逐字复制标题或摘要原文。只返回 JSON，不要 Markdown。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请按以下结构返回 JSON："
                    "{pmid,source_url,title,title_evidence_ids," 
                    "research_object,disease,method,dataset,main_findings,limitations,evidence_spans}。"
                    "每个字段值格式为 {value,status,evidence_ids}；"
                    "每个 evidence_span 格式为 {evidence_id,pmid,source_section,exact_text,source_url}，"
                    "source_section 只能是 title 或 abstract。\n\n"
                    f"PMID：{_pmid_for(document)}\n"
                    f"标题（title）：{document.title}\n"
                    f"摘要（abstract）：{document.abstract or '[无摘要]'}\n"
                    f"来源链接：{_url_for(document, _pmid_for(document))}"
                ),
            },
        ]

    @staticmethod
    def _parse_json(raw: str) -> dict:
        text = raw.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise EvidenceExtractionError("模型没有返回合法的证据 JSON。") from exc
        if not isinstance(payload, dict):
            raise EvidenceExtractionError("模型返回的证据 JSON 必须是对象。")
        return payload

    def _strict_llm(self, document: MedicalDocument) -> StructuredEvidence:
        if self.provider is None:
            raise EvidenceExtractionError(
                "strict_llm 模式需要配置证据提取模型；可先使用 metadata_only 模式。"
            )
        try:
            raw = self.provider.complete(self._prompt(document))
        except LLMProviderError as exc:
            raise EvidenceExtractionError(str(exc)) from exc
        except Exception as exc:  # custom providers should not leak implementation errors
            raise EvidenceExtractionError("证据提取模型调用失败。") from exc

        payload = self._parse_json(raw)
        pmid = _pmid_for(document)
        source_url = _url_for(document, pmid)
        payload["pmid"] = pmid
        payload["source_url"] = source_url
        payload["title"] = document.title
        try:
            evidence = StructuredEvidence.model_validate(payload)
        except Exception as exc:
            raise EvidenceExtractionError("模型返回的证据 JSON 不符合结构化字段要求。") from exc
        try:
            return validate_evidence(document, evidence)
        except EvidenceValidationError as exc:
            raise EvidenceExtractionError(str(exc)) from exc

    def extract(self, document: MedicalDocument, mode: EvidenceMode = "metadata_only") -> StructuredEvidence:
        if mode == "metadata_only":
            return self._metadata_only(document)
        if mode == "strict_llm":
            return self._strict_llm(document)
        raise EvidenceExtractionError(f"不支持的证据提取模式：{mode}")
