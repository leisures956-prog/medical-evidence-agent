import httpx
import pytest

from app.main import app
from app.medical_models import MedicalDocument, RetrievalHit
from app.pubmed_client_v2 import PubMedClientError, PubMedClientV2
from app.pubmed_hybrid import PubMedHybridRetriever


def _pubmed_xml() -> str:
    return """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>123</PMID>
          <Article>
            <ArticleTitle>Alpha study</ArticleTitle>
            <Abstract>
              <AbstractText>Alpha abstract evidence.</AbstractText>
            </Abstract>
            <Journal>
              <Title>Example Journal</Title>
              <JournalIssue><PubDate><Year>2026</Year></PubDate></JournalIssue>
            </Journal>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """


def test_pubmed_client_retries_transient_error_and_caches_result():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if len(calls) == 1:
            return httpx.Response(503, request=request)
        if request.url.path.endswith("/esearch.fcgi"):
            return httpx.Response(
                200,
                json={"esearchresult": {"idlist": ["123"]}},
                request=request,
            )
        return httpx.Response(200, text=_pubmed_xml(), request=request)

    client = PubMedClientV2(
        transport=httpx.MockTransport(handler),
        max_retries=1,
        retry_backoff_seconds=0,
        cache_enabled=True,
    )

    first = client.search("Alpha", limit=1)
    second = client.search("alpha", limit=1)

    assert [document.metadata["pmid"] for document in first] == ["123"]
    assert [document.metadata["pmid"] for document in second] == ["123"]
    # One transient failure, one successful ESearch, and one EFetch. The
    # second call is served from the case-insensitive in-memory cache.
    assert len(calls) == 3


def test_pubmed_client_retry_limit_is_respected():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(503, request=request)

    client = PubMedClientV2(
        transport=httpx.MockTransport(handler),
        max_retries=2,
        retry_backoff_seconds=0,
        cache_enabled=False,
    )

    try:
        client.search("always fails", limit=1)
    except Exception as exc:  # noqa: BLE001 - assert the public error below
        assert "PubMed 请求失败" in str(exc) or "503" in str(exc)
    else:
        raise AssertionError("transient PubMed failures should eventually fail")

    assert len(calls) == 3


def test_pubmed_client_timeout_is_exposed_as_public_error():
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    client = PubMedClientV2(
        transport=httpx.MockTransport(timeout_handler),
        max_retries=0,
        cache_enabled=False,
    )

    with pytest.raises(PubMedClientError, match="超时"):
        client.search("timeout", limit=1)


def _document(pmid: str = "1") -> MedicalDocument:
    return MedicalDocument(
        id=f"pubmed:{pmid}",
        source="pubmed",
        title="Alpha study",
        abstract="Alpha abstract evidence.",
        metadata={"pmid": pmid},
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    )


class CountingPubMedClient:
    def __init__(self):
        self.calls = 0

    def search(self, query, limit):
        self.calls += 1
        return [_document()]


class FakeSemanticRetriever:
    factory_calls = 0

    def __init__(self, documents, model_name, embedder):
        type(self).factory_calls += 1
        self.documents = documents
        self.embedder = object()

    def search(self, query, top_k=5):
        return [
            RetrievalHit(
                document=document,
                score=0.9,
                semantic_score=0.9,
                retrieval_method="semantic",
            )
            for document in self.documents[:top_k]
        ]


def test_pubmed_hybrid_cache_avoids_reembedding_and_refetching():
    FakeSemanticRetriever.factory_calls = 0
    client = CountingPubMedClient()
    retriever = PubMedHybridRetriever(
        pubmed_client=client,
        semantic_factory=FakeSemanticRetriever,
        cache_enabled=True,
        cache_ttl_seconds=60,
    )

    first = retriever.search("alpha", candidate_limit=20, top_k=1)
    second = retriever.search("alpha", candidate_limit=20, top_k=1)

    assert len(first) == len(second) == 1
    assert client.calls == 1
    assert FakeSemanticRetriever.factory_calls == 1
    assert first[0].document.id == second[0].document.id


def test_request_id_middleware_is_registered():
    dispatch_names = {
        getattr(middleware.kwargs.get("dispatch"), "__name__", "")
        for middleware in app.user_middleware
    }
    assert "request_context_middleware" in dispatch_names
