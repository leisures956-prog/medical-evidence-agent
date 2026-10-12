from __future__ import annotations

import os
import re
from typing import Any, Protocol

import httpx

from app.medical_models import GroundedAnswer, RetrievalHit


class LLMProviderError(RuntimeError):
    """Raised when the configured LLM provider cannot answer."""


class LLMProvider(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str:
        ...


class OpenAICompatibleProvider:
    """Works with OpenAI or another OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "")
        self.timeout = timeout

    def complete(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key or not self.model:
            raise LLMProviderError("请设置 LLM_API_KEY 和 LLM_MODEL。")

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={"model": self.model, "messages": messages, "temperature": 0},
                )
                response.raise_for_status()
                payload: dict[str, Any] = response.json()
                return str(payload["choices"][0]["message"]["content"]).strip()
        except (httpx.TimeoutException, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMProviderError("大模型请求失败或返回格式异常。") from exc


class GroundedAnswerer:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    @staticmethod
    def _context(hits: list[RetrievalHit]) -> str:
        blocks: list[str] = []
        for index, hit in enumerate(hits, start=1):
            document = hit.document
            authors = ", ".join(document.authors) or "未知作者"
            blocks.append(
                "\n".join(
                    [
                        f"[{index}] 标题：{document.title}",
                        f"作者：{authors}",
                        f"年份：{document.year or '未知'}",
                        f"期刊：{document.journal or '未知'}",
                        f"DOI：{document.doi or '无'}",
                        f"摘要：{document.abstract or '无摘要'}",
                        f"链接：{document.url or '无'}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _citation_numbers(answer: str, count: int) -> list[int]:
        numbers = sorted({int(item) for item in re.findall(r"\[(\d+)\]", answer)})
        return [number for number in numbers if 1 <= number <= count]

    def answer(self, question: str, hits: list[RetrievalHit]) -> GroundedAnswer:
        if not hits:
            return GroundedAnswer(answer="检索结果中没有足够资料回答该问题。", sources=[])

        messages = [
            {
                "role": "system",
                "content": (
                    "你是医学证据问答助手。只能依据用户提供的检索资料回答。"
                    "每个重要结论后必须使用 [1]、[2] 形式引用对应资料；"
                    "资料不足时明确说明，不得凭空补充。"
                ),
            },
            {
                "role": "user",
                "content": f"问题：{question}\n\n检索资料：\n{self._context(hits)}",
            },
        ]
        answer = self.provider.complete(messages)
        numbers = {int(item) for item in re.findall(r"\[(\d+)\]", answer)}
        if any(number < 1 or number > len(hits) for number in numbers):
            raise LLMProviderError("模型回答包含越界来源引用。")
        citations = self._citation_numbers(answer, len(hits))
        if not citations:
            raise LLMProviderError("模型回答没有包含有效的 [n] 来源引用。")
        return GroundedAnswer(
            answer=answer,
            sources=[hit.document for hit in hits],
            citations=citations,
        )
