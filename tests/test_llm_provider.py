import httpx
import pytest

from app.grounded_llm import LLMProviderError, OpenAICompatibleProvider
from app.llm_cache import LLMResponseCache


MESSAGES = [{"role": "user", "content": "test"}]


def test_provider_parses_usage_and_local_cost_is_zero(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5:3b",
                "choices": [{"message": {"content": "answer [1]"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            },
            request=request,
        )

    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5:3b",
        cache=LLMResponseCache(tmp_path),
        transport=httpx.MockTransport(handler),
    )

    result = provider.complete_with_metadata(MESSAGES)

    assert result.text == "answer [1]"
    assert result.usage.total_tokens == 14
    assert result.estimated_cost_usd == 0
    assert result.cache_hit is False


def test_provider_uses_d_drive_project_cache_without_storing_api_key(tmp_path):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "cached answer [1]"}}]},
            request=request,
        )

    provider = OpenAICompatibleProvider(
        api_key="secret-value",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5:3b",
        cache=LLMResponseCache(tmp_path),
        transport=httpx.MockTransport(handler),
    )

    first = provider.complete_with_metadata(MESSAGES)
    second = provider.complete_with_metadata(MESSAGES)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert calls == 1
    assert "secret-value" not in next(tmp_path.glob("*.json")).read_text(encoding="utf-8")


def test_provider_retries_timeouts_then_returns_clear_error(tmp_path):
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("timed out", request=request)

    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5:3b",
        max_retries=2,
        cache=LLMResponseCache(tmp_path),
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )

    with pytest.raises(LLMProviderError, match="超时"):
        provider.complete_with_metadata(MESSAGES)

    assert attempts == 3


def test_provider_requires_model_configuration(tmp_path, monkeypatch):
    # Make the missing-configuration case independent of the user's shell
    # environment (for example, an Ollama LLM_MODEL setting).
    monkeypatch.delenv("LLM_MODEL", raising=False)
    provider = OpenAICompatibleProvider(
        api_key="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="",
        cache=LLMResponseCache(tmp_path),
    )

    with pytest.raises(LLMProviderError, match="LLM_MODEL"):
        provider.complete_with_metadata(MESSAGES)


def test_invalid_cache_entry_can_be_removed(tmp_path):
    cache = LLMResponseCache(tmp_path)
    cache.set("invalid", {"text": "answer without citation"})

    assert cache.get("invalid") is not None
    cache.delete("invalid")
    assert cache.get("invalid") is None
