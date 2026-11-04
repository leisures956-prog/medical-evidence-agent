from concurrent.futures import ThreadPoolExecutor
import time

import httpx

from app.medical_models import MedicalDocument, RetrievalHit
from app.pubmed_client_v2 import PubMedClientV2
from app.pubmed_hybrid import PubMedHybridRetriever
from app.request_context import get_request_id, request_id_context


def _xml_for(pmid: str = "123") -> str:
    return f"""
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>{pmid}</PMID>
          <Article>
            <ArticleTitle>Stage 8C study</ArticleTitle>
            <Abstract><AbstractText>Stable evidence.</AbstractText></Abstract>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """


def test_pubmed_cache_expires_after_ttl():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("esearch.fcgi"):
            return httpx.Response(
                200,
                json={"esearchresult": {"idlist": ["123"]}},
                request=request,
            )
        return httpx.Response(200, text=_xml_for(), request=request)

    client = PubMedClientV2(
        transport=httpx.MockTransport(handler),
        max_retries=0,
        cache_ttl_seconds=0.01,
    )
    client.search("ttl", limit=1)
    time.sleep(0.02)
    client.search("ttl", limit=1)

    assert len(calls) == 4


def test_pubmed_429_is_retried_once_then_succeeds():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if len(calls) == 1:
            return httpx.Response(429, request=request)
        if request.url.path.endswith("esearch.fcgi"):
            return httpx.Response(
                200,
                json={"esearchresult": {"idlist": ["123"]}},
                request=request,
            )
        return httpx.Response(200, text=_xml_for(), request=request)

    client = PubMedClientV2(
        transport=httpx.MockTransport(handler),
        max_retries=1,
        retry_backoff_seconds=0,
        cache_enabled=False,
    )

    assert client.search("rate limited", limit=1)[0].metadata["pmid"] == "123"
    assert len(calls) == 3


class ConcurrentPubMedClient:
    def __init__(self):
        self.calls = []

    def search(self, query, limit):
        self.calls.append(query)
        return [
            MedicalDocument(
                id=f"pubmed:{query}",
                source="pubmed",
                title=f"Study {query}",
                abstract=f"Evidence for {query}.",
                metadata={"pmid": query},
            )
        ]


class ConcurrentSemanticRetriever:
    def __init__(self, documents, model_name, embedder):
        self.documents = documents
        self.embedder = embedder or object()

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


def test_concurrent_hybrid_queries_keep_results_isolated():
    client = ConcurrentPubMedClient()
    retriever = PubMedHybridRetriever(
        pubmed_client=client,
        semantic_factory=ConcurrentSemanticRetriever,
        cache_enabled=True,
        cache_ttl_seconds=60,
    )
    queries = ["alpha", "beta", "alpha", "beta"]

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda query: retriever.search(query, candidate_limit=20, top_k=1),
                queries,
            )
        )

    assert [result[0].document.metadata["pmid"] for result in results] == queries


def test_request_id_context_is_scoped_and_restored():
    assert get_request_id() == "-"
    token = request_id_context.set("stage8c-test")
    try:
        assert get_request_id() == "stage8c-test"
    finally:
        request_id_context.reset(token)
    assert get_request_id() == "-"
