import httpx
import pytest

from app.agent_nodes import citation_check_node
from app.grounded_llm import GroundedAnswerer
from app.medical_models import MedicalDocument, RetrievalHit
from app.pubmed_client_v2 import PubMedClientV2


PUBMED_XML = """
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>123456</PMID>
      <Article>
        <ArticleTitle>Example medical evidence</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Background text.</AbstractText>
          <AbstractText>Result text.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Zhang</LastName><ForeName>Wei</ForeName></Author>
        </AuthorList>
        <Journal>
          <Title>Example Medicine</Title>
          <JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1000/example</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


def pubmed_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("esearch.fcgi"):
        return httpx.Response(
            200,
            json={"esearchresult": {"idlist": ["123456"]}},
            request=request,
        )
    return httpx.Response(200, text=PUBMED_XML, request=request)


def test_pubmed_returns_full_metadata():
    client = PubMedClientV2(transport=httpx.MockTransport(pubmed_handler))
    documents = client.search("medical evidence", limit=1)

    assert documents[0].abstract.startswith("BACKGROUND:")
    assert documents[0].authors == ["Wei Zhang"]
    assert documents[0].year == 2024
    assert documents[0].doi == "10.1000/example"


def test_citation_check_rejects_missing_and_invalid_citations():
    document = MedicalDocument(id="doc-1", title="Evidence")
    state = {
        "question": "question",
        "hits": [
            RetrievalHit(document=document, score=0.9, retrieval_method="hybrid")
        ],
        "answer": "A claim [3]",
    }

    checked = citation_check_node(state)
    assert checked["citations"] == []
    assert checked["errors"]


class FakeProvider:
    def complete(self, messages):
        return "结论来自资料。[1]"


def test_grounded_answer_requires_citation():
    document = MedicalDocument(id="doc-1", title="Evidence", abstract="Abstract")
    hit = RetrievalHit(document=document, score=0.9, retrieval_method="hybrid")
    result = GroundedAnswerer(FakeProvider()).answer("question", [hit])

    assert result.citations == [1]
    assert result.sources[0].id == "doc-1"
