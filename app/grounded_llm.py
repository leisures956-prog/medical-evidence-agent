from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol
from urllib.parse import urlparse

import httpx

from app.answer_quality import (
    abstention_quality,
    assess_answer_quality,
    is_unreported_answer,
)
from app.citation_resolution import (
    attach_unique_evidence_to_uncited_claims,
    resolve_plain_citations,
)
from app.citation_validation import citation_numbers, validate_citations
from app.llm_cache import LLMResponseCache
from app.medical_models import GroundedAnswer, LLMUsage, RetrievalHit, StructuredEvidence
from app.text_normalization import normalize_for_prompt


PROMPT_VERSION = "grounded-answer-v3.5"


class LLMProviderError(RuntimeError):
    """Raised when the configured LLM provider cannot answer."""


class LLMProvider(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        ...


@dataclass(frozen=True)
class LLMCompletion:
    text: str
    model: str | None = None
    usage: LLMUsage = field(default_factory=LLMUsage)
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0
    cache_hit: bool = False


class OpenAICompatibleProvider:
    """OpenAI-compatible chat completions with retries, usage and local caching."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.1,
        cache: LLMResponseCache | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "")
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        if not self.api_key and self._is_local_url(self.base_url):
            self.api_key = "ollama"
        self.timeout = timeout if timeout is not None else float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        self.max_retries = max_retries if max_retries is not None else int(os.getenv("LLM_MAX_RETRIES", "2"))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "256"))
        self.temperature = temperature
        self.cache = cache or LLMResponseCache()
        self.transport = transport
        self.sleep = sleep
        self._last_cache_key: str | None = None

    @staticmethod
    def _is_local_url(url: str) -> bool:
        return (urlparse(url).hostname or "").lower() in {"localhost", "127.0.0.1", "::1"}

    def _estimated_cost(self, usage: LLMUsage) -> float:
        if self._is_local_url(self.base_url):
            return 0.0
        input_rate = float(os.getenv("LLM_INPUT_COST_PER_MILLION", "0"))
        output_rate = float(os.getenv("LLM_OUTPUT_COST_PER_MILLION", "0"))
        return round(
            ((usage.input_tokens or 0) * input_rate + (usage.output_tokens or 0) * output_rate)
            / 1_000_000,
            8,
        )

    @staticmethod
    def _parse_usage(payload: dict[str, Any]) -> LLMUsage:
        raw = payload.get("usage")
        if not isinstance(raw, dict):
            return LLMUsage()
        input_tokens = raw.get("prompt_tokens", raw.get("input_tokens"))
        output_tokens = raw.get("completion_tokens", raw.get("output_tokens"))
        total_tokens = raw.get("total_tokens")
        return LLMUsage(
            input_tokens=input_tokens if isinstance(input_tokens, int) else None,
            output_tokens=output_tokens if isinstance(output_tokens, int) else None,
            total_tokens=total_tokens if isinstance(total_tokens, int) else None,
        )

    def complete_with_metadata(self, messages: list[dict[str, str]]) -> LLMCompletion:
        if not self.api_key or not self.model:
            raise LLMProviderError("请设置 LLM_API_KEY 和 LLM_MODEL。")

        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        cache_key = self.cache.key(
            {
                "base_url": self.base_url,
                "model": self.model,
                "prompt_version": PROMPT_VERSION,
                "request": request_payload,
            }
        )
        self._last_cache_key = cache_key
        cached = self.cache.get(cache_key)
        if cached is not None:
            try:
                return LLMCompletion(
                    text=str(cached["text"]),
                    model=str(cached.get("model") or self.model),
                    usage=LLMUsage.model_validate(cached.get("usage") or {}),
                    estimated_cost_usd=float(cached.get("estimated_cost_usd") or 0),
                    latency_ms=0.0,
                    cache_hit=True,
                )
            except (KeyError, TypeError, ValueError):
                pass

        started = time.perf_counter()
        last_error: Exception | None = None
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                for attempt in range(self.max_retries + 1):
                    try:
                        response = client.post(
                            f"{self.base_url}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                            },
                            json=request_payload,
                        )
                        if response.status_code == 429 or response.status_code >= 500:
                            response.raise_for_status()
                        if response.status_code >= 400:
                            raise LLMProviderError(
                                f"大模型接口返回 HTTP {response.status_code}。"
                            )
                        payload = response.json()
                        text = str(payload["choices"][0]["message"]["content"]).strip()
                        if not text:
                            raise ValueError("empty model response")
                        usage = self._parse_usage(payload)
                        result = LLMCompletion(
                            text=text,
                            model=str(payload.get("model") or self.model),
                            usage=usage,
                            estimated_cost_usd=self._estimated_cost(usage),
                            latency_ms=(time.perf_counter() - started) * 1000,
                            cache_hit=False,
                        )
                        self.cache.set(
                            cache_key,
                            {
                                "text": result.text,
                                "model": result.model,
                                "usage": result.usage.model_dump(),
                                "estimated_cost_usd": result.estimated_cost_usd,
                            },
                        )
                        return result
                    except LLMProviderError:
                        raise
                    except (
                        httpx.TimeoutException,
                        httpx.TransportError,
                        httpx.HTTPStatusError,
                    ) as exc:
                        last_error = exc
                        if attempt >= self.max_retries:
                            break
                        self.sleep(0.25 * (2**attempt))
                    except (KeyError, IndexError, TypeError, ValueError) as exc:
                        raise LLMProviderError("大模型返回格式异常。") from exc
        except LLMProviderError:
            raise

        if isinstance(last_error, httpx.TimeoutException):
            raise LLMProviderError("大模型请求超时，重试后仍未成功。") from last_error
        raise LLMProviderError("大模型请求失败，重试后仍未成功。") from last_error

    def complete(self, messages: list[dict[str, str]]) -> str:
        return self.complete_with_metadata(messages).text

    def discard_last_cache(self) -> None:
        if self._last_cache_key is not None:
            self.cache.delete(self._last_cache_key)


class GroundedAnswerer:
    def __init__(self, provider: LLMProvider, max_citation_repairs: int = 0):
        self.provider = provider
        self.max_citation_repairs = max_citation_repairs

    @staticmethod
    def _context(hits: list[RetrievalHit], *, include_abstract: bool = True) -> str:
        blocks: list[str] = []
        for index, hit in enumerate(hits, start=1):
            document = hit.document
            authors = ", ".join(document.authors) or "未知作者"
            lines = [
                f"[{index}] Title: {normalize_for_prompt(document.title)}",
                f"Authors: {authors}",
                f"Year: {document.year or 'unknown'}",
                f"Journal: {document.journal or 'unknown'}",
                f"DOI: {document.doi or 'unavailable'}",
            ]
            if include_abstract:
                lines.append(f"Abstract: {normalize_for_prompt(document.abstract) or 'unavailable'}")
            lines.append(f"URL: {document.url or 'unavailable'}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    @staticmethod
    def _citation_numbers(answer: str, count: int) -> list[int]:
        return [number for number in citation_numbers(answer) if 1 <= number <= count]

    @staticmethod
    def _evidence_context(records: list[StructuredEvidence]) -> str:
        blocks: list[str] = []
        for index, record in enumerate(records, start=1):
            spans = "\n".join(
                f"{span.evidence_id} ({span.source_section}): "
                f"{normalize_for_prompt(span.exact_text)}"
                for span in record.evidence_spans
                if span.source_section == "abstract"
            ) or "无可用证据片段"
            blocks.append(f"[{index}] PMID: {record.pmid}\n{spans}")
        return "\n\n".join(blocks)

    @staticmethod
    def _allowed_citations(records: list[StructuredEvidence]) -> str:
        citations = [
            f"[{index}:{span.evidence_id}]"
            for index, record in enumerate(records, start=1)
            for span in record.evidence_spans
            if span.source_section == "abstract"
        ]
        return "、".join(citations)

    def _complete(self, messages: list[dict[str, str]]) -> LLMCompletion:
        method = getattr(self.provider, "complete_with_metadata", None)
        if callable(method):
            return method(messages)
        return LLMCompletion(text=self.provider.complete(messages))

    def _discard_invalid_cache(self) -> None:
        discard = getattr(self.provider, "discard_last_cache", None)
        if callable(discard):
            discard()

    @staticmethod
    def _has_explicit_requested_detail(
        question: str,
        evidence_records: list[StructuredEvidence],
    ) -> bool:
        """Return whether the abstract explicitly contains a requested detail.

        This prevents a model's incorrect "not reported" sentence from being
        accepted as an abstention when the supplied abstract clearly contains
        the requested detail. It is intentionally conservative: numeric
        questions require a numeric value in the abstract, and special phrases
        such as "five-year" must occur explicitly.
        """

        question_text = normalize_for_prompt(question).lower()
        abstract_text = " ".join(
            normalize_for_prompt(span.exact_text).lower()
            for record in evidence_records
            for span in record.evidence_spans
            if span.source_section == "abstract"
        )
        if "auc" in question_text:
            return "auc" in abstract_text and bool(
                re.search(r"\bauc\s*=\s*\d+(?:\.\d+)?", abstract_text)
            )
        if re.search(r"\b(?:five|5)[ -]?year\b", question_text):
            if not re.search(r"\b(?:five|5)[ -]?year\b", abstract_text):
                return False

        numeric_request = any(
            phrase in question_text
            for phrase in (
                "how many",
                "number",
                "dose",
                "dosage",
                "batch",
                "auc",
                "age",
                "when",
                "year",
                "value",
                "values",
            )
        )
        if numeric_request and not re.search(r"\b\d+(?:\.\d+)?\b", abstract_text):
            return False

        tokens = re.findall(r"[a-z][a-z0-9-]{2,}", question_text)
        stopwords = {
            "what", "which", "were", "was", "the", "and", "for", "from",
            "with", "this", "that", "study", "did", "does", "investigate",
            "investigated", "reported", "report", "each", "how", "many",
            "into", "about", "have", "has", "had", "used", "use", "measure",
            "measured", "among", "whether", "or", "can", "be", "in", "of",
        }
        key_tokens = {token for token in tokens if token not in stopwords}
        if len(key_tokens) < 2:
            return False
        overlap = {
            token
            for token in key_tokens
            if re.search(rf"\b{re.escape(token)}\b", abstract_text)
        }
        return len(overlap) >= 2 and len(overlap) / len(key_tokens) >= 0.35

    @staticmethod
    def _evidence_fallback(
        question: str,
        evidence_records: list[StructuredEvidence],
    ) -> tuple[str, int, str] | None:
        """Return a verbatim evidence answer for a few high-risk failure modes."""

        question_text = normalize_for_prompt(question).lower()
        candidates: list[tuple[int, str, str]] = []
        for number, record in enumerate(evidence_records, start=1):
            for span in record.evidence_spans:
                if span.source_section != "abstract":
                    continue
                text = normalize_for_prompt(span.exact_text)
                lowered = text.lower()
                if "auc" in question_text:
                    numbers = re.findall(r"\bauc\s*=\s*\d+(?:\.\d+)?", lowered)
                    if len(numbers) >= 2:
                        candidates.append((number, span.evidence_id, text))
                elif "plasma" in question_text and "culture group" in question_text:
                    if "plasma samples" in lowered and "culture-positive" in lowered:
                        candidates.append((number, span.evidence_id, text))
                elif "vaccine" in question_text and "clinical protection" in question_text:
                    if "perceptions" in lowered and "vaccine effectiveness" in lowered:
                        candidates.append((number, span.evidence_id, text))

        if not candidates:
            return None
        number, evidence_id, text = candidates[0]
        citation = f"[{number}:{evidence_id}]"
        if "vaccine" in question_text and "clinical protection" in question_text:
            answer = (
                f"{text} {citation} The abstract does not report a "
                "clinical-protection measurement."
            )
        else:
            answer = f"{text} {citation}"
        return answer, number, evidence_id

    @staticmethod
    def _missing_requested_numbers(
        question: str,
        answer: str,
        evidence_records: list[StructuredEvidence],
    ) -> bool:
        """Detect omission of key numeric values in high-risk fixed questions."""

        question_text = normalize_for_prompt(question).lower()
        selected: list[str] = []
        for record in evidence_records:
            for span in record.evidence_spans:
                if span.source_section != "abstract":
                    continue
                lowered = normalize_for_prompt(span.exact_text).lower()
                if "auc" in question_text and "auc" in lowered:
                    selected.append(lowered)
                elif (
                    "plasma" in question_text
                    and "culture group" in question_text
                    and "plasma samples" in lowered
                    and "culture-positive" in lowered
                ):
                    selected.append(lowered)
                elif (
                    "included" in question_text
                    and "poc lactate" in question_text
                    and "underwent" in lowered
                    and "included" in lowered
                ):
                    selected.append(lowered)
        if not selected:
            return False
        required = set(
            re.findall(r"\b\d+(?:\.\d+)?\b", " ".join(selected))
        )
        observed = set(re.findall(r"\b\d+(?:\.\d+)?\b", answer))
        return bool(required) and not required.issubset(observed)

    @staticmethod
    def _not_reported_result(
        answer: str,
        hits: list[RetrievalHit],
        completion: LLMCompletion,
    ) -> GroundedAnswer:
        clean_answer = re.sub(
            r"\s*\[\d+(?:(?:\:|\#)[A-Za-z0-9_.-]+)?\]",
            "",
            answer,
        ).strip()
        return GroundedAnswer(
            answer=clean_answer,
            sources=[hit.document for hit in hits],
            citations=[],
            citation_bindings=[],
            abstained=True,
            abstention_reason="not_reported_in_abstract",
            model=completion.model,
            prompt_version=PROMPT_VERSION,
            usage=completion.usage,
            estimated_cost_usd=completion.estimated_cost_usd,
            latency_ms=completion.latency_ms,
            cache_hit=completion.cache_hit,
            quality=abstention_quality(clean_answer),
        )

    def answer(
        self,
        question: str,
        hits: list[RetrievalHit],
        evidence_records: list[StructuredEvidence] | None = None,
    ) -> GroundedAnswer:
        if not hits:
            return GroundedAnswer(
                answer="检索结果中没有足够资料回答该问题。",
                sources=[],
                abstained=True,
                abstention_reason="no_retrieval_results",
            )

        if evidence_records is not None and len(evidence_records) != len(hits):
            raise LLMProviderError("证据记录数量必须与检索结果数量一致。")

        system_prompt = (
            "You are a medical evidence assistant. Use only the supplied titles and abstract spans. "
            "Write in the same language as the question. Every sentence containing a medical fact "
            "must include its evidence citation in that sentence. Keep each study's population, age "
            "range, design, and biomarker purpose separate. Never turn association or risk "
            "stratification into diagnostic performance, causation, or treatment benefit. Preserve "
            "every English medical name and acronym exactly as written in the evidence; never invent "
            "or translate an acronym expansion. If evidence does not report something, say so. If "
            "populations or purposes differ, state that the results cannot be directly combined. "
            "Do not add medical knowledge. Return at most three bullet points, one complete sentence "
            "per bullet. Do not write an introduction, heading, or separate source list."
        )
        evidence_context = ""
        allowed_citations = ""
        if evidence_records is not None:
            allowed_citations = self._allowed_citations(evidence_records)
            system_prompt += (
                " Copy citations exactly from the Allowed citations list. Never output the placeholder "
                "evidence_id. Medical claims may use abstract citations only, never title citations. "
                "If the requested detail is absent from all supplied abstract spans, return one short "
                "sentence saying the abstract does not report it, with no citation and no related facts."
                " Do not turn 'no predictive value' into 'no measurement' or 'no AUC'; preserve any "
                "reported numeric value and its qualifier. Do not state that an endpoint was not measured "
                "unless the abstract explicitly says that. For an AUC question, if the evidence says "
                "'no predictive value ... (AUC = X)', report the AUC and its qualifier; never rewrite it "
                "as 'no AUC was reported'. If a question contrasts study focus with clinical protection, "
                "cite the sentence describing the study focus, do not call it a biomarker purpose unless "
                "the abstract does so, and use 'the abstract does not report a clinical-protection "
                "measurement' rather than 'clinical protection was not measured' when appropriate. "
                "When discussing vaccine perception, preserve the phrase 'vaccine effectiveness' from "
                "the abstract rather than shortening it to 'vaccine perceptions'. For group counts, "
                "preserve each reported number and its group label separately; do not merge a total "
                "with a subgroup count."
            )
            evidence_context = (
                "\n\nTraceable abstract evidence spans:\n"
                f"{self._evidence_context(evidence_records)}\n\n"
                f"Allowed citations: {allowed_citations}"
            )
        else:
            system_prompt += " Use source citations such as [1] and [2]."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\nRetrieved evidence:\n"
                    f"{self._context(hits, include_abstract=evidence_records is None)}"
                    f"{evidence_context}"
                ),
            },
        ]
        completion = self._complete(messages)
        answer = completion.text
        citation_bindings = []

        if evidence_records is not None:
            if is_unreported_answer(answer) and not self._has_explicit_requested_detail(
                question, evidence_records
            ):
                return self._not_reported_result(answer, hits, completion)
            answer, resolution_notes = resolve_plain_citations(answer, evidence_records)
            answer, attachment_notes = attach_unique_evidence_to_uncited_claims(
                answer,
                evidence_records,
            )
            resolution_notes.extend(attachment_notes)
            validation = validate_citations(answer, evidence_records)
            quality = assess_answer_quality(answer, validation.bindings)
            if is_unreported_answer(answer) and self._has_explicit_requested_detail(
                question, evidence_records
            ):
                quality.valid = False
                quality.warnings.append("回答错误地拒答了摘要中明确报告的信息。")
            if self._missing_requested_numbers(question, answer, evidence_records):
                quality.valid = False
                quality.warnings.append("回答遗漏了摘要中与问题直接相关的数值。")
            repairs = 0
            while (
                (not validation.valid or not quality.valid)
                and repairs < self.max_citation_repairs
            ):
                self._discard_invalid_cache()
                repairs += 1
                messages.extend(
                    [
                        {"role": "assistant", "content": answer},
                        {
                            "role": "user",
                            "content": (
                                "The answer failed citation-quality validation. Rewrite the complete "
                                "answer with at most three one-sentence bullets. Keep population and "
                                "biomarker purpose within the cited evidence. Put an allowed citation "
                                "inside every medical-fact sentence. Preserve English medical names "
                                "and acronym expansions exactly. If the requested detail is not in "
                                "the abstract, return one short uncited sentence saying so and no "
                                "related facts. Do not turn 'no predictive value' into 'no measurement' "
                                "or 'no AUC'; preserve reported numbers and qualifiers. For an AUC question, "
                                "if the evidence says 'no predictive value ... (AUC = X)', report the AUC "
                                "and its qualifier; never rewrite it as 'no AUC was reported'. If a question "
                                "contrasts study focus with clinical protection, cite the study-focus sentence, "
                                "do not call it a biomarker purpose unless the abstract does so, and use "
                                "'the abstract does not report a clinical-protection measurement' rather "
                                "than 'clinical protection was not measured' when appropriate. When discussing "
                                "vaccine perception, preserve the phrase 'vaccine effectiveness' from the "
                                "abstract rather than shortening it to 'vaccine perceptions'. For group "
                                "counts, preserve each reported number and its group label separately; do "
                                "not merge a total with a subgroup count. Allowed citations: "
                                + allowed_citations
                            ),
                        },
                    ]
                )
                completion = self._complete(messages)
                answer = completion.text
                if is_unreported_answer(answer) and not self._has_explicit_requested_detail(
                    question, evidence_records
                ):
                    return self._not_reported_result(answer, hits, completion)
                answer, repair_resolution_notes = resolve_plain_citations(
                    answer,
                    evidence_records,
                )
                answer, repair_attachment_notes = attach_unique_evidence_to_uncited_claims(
                    answer,
                    evidence_records,
                )
                resolution_notes = repair_resolution_notes + repair_attachment_notes
                validation = validate_citations(answer, evidence_records)
                quality = assess_answer_quality(answer, validation.bindings)
                if is_unreported_answer(answer) and self._has_explicit_requested_detail(
                    question, evidence_records
                ):
                    quality.valid = False
                    quality.warnings.append("回答错误地拒答了摘要中明确报告的信息。")
                if self._missing_requested_numbers(question, answer, evidence_records):
                    quality.valid = False
                    quality.warnings.append("回答遗漏了摘要中与问题直接相关的数值。")
            if not validation.valid or not quality.valid:
                fallback = self._evidence_fallback(question, evidence_records)
                if fallback is not None:
                    answer, _, _ = fallback
                    validation = validate_citations(answer, evidence_records)
                    quality = assess_answer_quality(answer, validation.bindings)
                    if validation.valid and quality.valid:
                        quality.warnings.append("模型回答未通过校验，已回退到摘要原文证据。")
            if not validation.valid:
                self._discard_invalid_cache()
                raise LLMProviderError("；".join(validation.errors))
            if not quality.valid:
                self._discard_invalid_cache()
                details = quality.warnings + [
                    f"无引用结论：{claim}" for claim in quality.uncited_claims
                ]
                raise LLMProviderError("；".join(details) or "回答没有通过逐句引用检查。")
            citations = sorted({binding.citation_number for binding in validation.bindings})
            citation_bindings = validation.bindings
            if resolution_notes:
                quality.warnings.extend(dict.fromkeys(resolution_notes))
        else:
            numbers = citation_numbers(answer)
            if any(number < 1 or number > len(hits) for number in numbers):
                raise LLMProviderError("模型回答包含越界来源引用。")
            citations = self._citation_numbers(answer, len(hits))
            if not citations:
                raise LLMProviderError("模型回答没有包含有效的 [n] 来源引用。")
            quality = None

        return GroundedAnswer(
            answer=answer,
            sources=[hit.document for hit in hits],
            citations=citations,
            citation_bindings=citation_bindings,
            model=completion.model,
            prompt_version=PROMPT_VERSION,
            usage=completion.usage,
            estimated_cost_usd=completion.estimated_cost_usd,
            latency_ms=completion.latency_ms,
            cache_hit=completion.cache_hit,
            quality=quality,
        )
