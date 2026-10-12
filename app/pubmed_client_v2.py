from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from app.medical_models import MedicalDocument


class PubMedClientError(RuntimeError):
    """Raised when PubMed cannot be reached or its response is invalid."""


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def _first_year(*values: str) -> int | None:
    for value in values:
        match = re.search(r"\b(19|20)\d{2}\b", value or "")
        if match:
            return int(match.group(0))
    return None


class PubMedClientV2:
    """Fetch PubMed records with title, abstract, authors, year and DOI."""

    def __init__(
        self,
        base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        timeout: float | None = None,
        email: str | None = None,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout or float(os.getenv("PUBMED_TIMEOUT_SECONDS", "15"))
        self.email = email if email is not None else os.getenv("NCBI_EMAIL")
        self.api_key = api_key if api_key is not None else os.getenv("NCBI_API_KEY")
        self.transport = transport

    def _common_params(self) -> dict[str, str]:
        params = {"retmode": "json"}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _get_json(self, client: httpx.Client, endpoint: str, params: dict[str, str]):
        response = client.get(f"{self.base_url}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()

    def _get_xml(self, client: httpx.Client, endpoint: str, params: dict[str, str]) -> str:
        response = client.get(f"{self.base_url}/{endpoint}", params=params)
        response.raise_for_status()
        return response.text

    @staticmethod
    def _parse_article(article: ET.Element) -> MedicalDocument | None:
        pmid = _element_text(article.find(".//PMID"))
        if not pmid:
            return None

        title = _element_text(article.find(".//ArticleTitle"))
        abstract_parts: list[str] = []
        for node in article.findall(".//Abstract/AbstractText"):
            text = _element_text(node)
            if not text:
                continue
            label = node.attrib.get("Label")
            abstract_parts.append(f"{label}: {text}" if label else text)

        authors: list[str] = []
        for author in article.findall(".//AuthorList/Author"):
            collective = _element_text(author.find("CollectiveName"))
            if collective:
                authors.append(collective)
                continue
            family = _element_text(author.find("LastName"))
            fore = _element_text(author.find("ForeName"))
            initials = _element_text(author.find("Initials"))
            name = " ".join(part for part in (fore or initials, family) if part)
            if name:
                authors.append(name)

        journal = _element_text(article.find(".//Journal/Title")) or None
        pub_date = article.find(".//JournalIssue/PubDate")
        article_date = article.find(".//ArticleDate")
        year = _first_year(
            _element_text(article_date.find("Year") if article_date is not None else None),
            _element_text(pub_date.find("Year") if pub_date is not None else None),
            _element_text(pub_date.find("MedlineDate") if pub_date is not None else None),
        )

        doi: str | None = None
        for article_id in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if article_id.attrib.get("IdType", "").lower() == "doi":
                doi = _element_text(article_id) or None
                break

        keywords = [
            _element_text(node)
            for node in article.findall(".//KeywordList/Keyword")
            if _element_text(node)
        ]

        return MedicalDocument(
            id=f"pubmed:{pmid}",
            source="pubmed",
            title=title or f"PubMed article {pmid}",
            abstract="\n".join(abstract_parts),
            authors=authors,
            year=year,
            doi=doi,
            journal=journal,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            keywords=keywords,
            metadata={"pmid": pmid},
        )

    def search(self, query: str, limit: int = 10) -> list[MedicalDocument]:
        if not query.strip():
            return []

        try:
            with httpx.Client(
                timeout=self.timeout,
                transport=self.transport,
                headers={"User-Agent": "medical-evidence-agent/0.2"},
            ) as client:
                search_params = self._common_params()
                search_params.update(
                    {"db": "pubmed", "term": query.strip(), "retmax": str(limit)}
                )
                payload: dict[str, Any] = self._get_json(client, "esearch.fcgi", search_params)
                ids = payload.get("esearchresult", {}).get("idlist", [])[:limit]
                ids = [str(item) for item in ids]
                if not ids:
                    return []

                fetch_params = self._common_params()
                fetch_params.update({"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
                xml_text = self._get_xml(client, "efetch.fcgi", fetch_params)

        except httpx.TimeoutException as exc:
            raise PubMedClientError("PubMed 请求超时。") from exc
        except httpx.HTTPError as exc:
            raise PubMedClientError("PubMed 请求失败。") from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise PubMedClientError("PubMed 搜索结果格式异常。") from exc

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise PubMedClientError("PubMed 返回的 XML 无法解析。") from exc

        parsed: dict[str, MedicalDocument] = {}
        for article in root.findall(".//PubmedArticle"):
            document = self._parse_article(article)
            if document:
                parsed[document.metadata["pmid"]] = document

        return [parsed[pmid] for pmid in ids if pmid in parsed]
