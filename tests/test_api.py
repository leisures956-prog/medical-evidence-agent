import httpx
from fastapi.testclient import TestClient

from app.main import app, get_search_service
from app.search import PubMedClient, SearchHit, SearchService


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_search_success():
    response = client.post("/search", json={"query": "Python"})

    assert response.status_code == 200
    assert response.json()["source"] == "local"
    assert response.json()["total"] >= 1
    assert len(response.json()["results"]) >= 1


def test_search_no_result():
    response = client.post("/search", json={"query": "不存在的内容"})

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_search_invalid_limit():
    response = client.post("/search", json={"query": "Python", "limit": 0})

    assert response.status_code == 422


def test_answer_success():
    response = client.post("/answer", json={"question": "什么是 Python 列表？"})

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "confidence" in data
    assert "[1]" in data["answer"]


def test_blank_query():
    response = client.post("/search", json={"query": " "})

    assert response.status_code == 400


def _pubmed_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/esearch.fcgi"):
        return httpx.Response(
            200,
            json={"esearchresult": {"idlist": ["123456"]}},
            request=request,
        )
    return httpx.Response(
        200,
        json={
            "result": {
                "uids": ["123456"],
                "123456": {
                    "title": "A tested PubMed article",
                    "fulljournalname": "Example Journal",
                    "pubdate": "2026",
                },
            }
        },
        request=request,
    )


def test_pubmed_search_with_mocked_external_api():
    service = SearchService(
        pubmed_client=PubMedClient(transport=httpx.MockTransport(_pubmed_handler))
    )
    app.dependency_overrides[get_search_service] = lambda: service

    try:
        response = client.post(
            "/search",
            json={"query": "medical evidence", "source": "pubmed"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "pubmed:123456"
    assert response.json()["results"][0]["source"] == "pubmed"


def test_pubmed_timeout_returns_502():
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    service = SearchService(
        pubmed_client=PubMedClient(transport=httpx.MockTransport(timeout_handler))
    )
    app.dependency_overrides[get_search_service] = lambda: service

    try:
        response = client.post(
            "/search",
            json={"query": "medical evidence", "source": "pubmed"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert "超时" in response.json()["detail"]


class FakeVectorIndex:
    def search(self, query: str, limit: int) -> list[SearchHit]:
        return [
            SearchHit(
                id="doc-vector",
                title="向量检索测试",
                content=f"query={query}",
                score=0.91,
                source="vector",
            )
        ][:limit]


def test_vector_search_with_injected_index():
    service = SearchService(vector_index=FakeVectorIndex())
    app.dependency_overrides[get_search_service] = lambda: service

    try:
        response = client.post(
            "/search",
            json={"query": "列表结构", "source": "vector"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["results"][0]["source"] == "vector"
