from __future__ import annotations

import os
import re
import logging
import time
from threading import Lock
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from app.medical_models import MedicalDocument
from app.request_context import get_request_id


class PubMedClientError(RuntimeError):
    """Raised when PubMed cannot be reached or its response is invalid."""


logger = logging.getLogger(__name__)


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
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
        cache_enabled: bool | None = None,
        cache_ttl_seconds: float | None = None,
        cache_max_entries: int | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = (
            float(os.getenv("PUBMED_TIMEOUT_SECONDS", "15"))
            if timeout is None
            else float(timeout)
        )
        self.max_retries = (
            int(os.getenv("PUBMED_MAX_RETRIES", "2"))
            if max_retries is None
            else int(max_retries)
        )
        self.retry_backoff_seconds = (
            float(os.getenv("PUBMED_RETRY_BACKOFF_SECONDS", "0.25"))
            if retry_backoff_seconds is None
            else float(retry_backoff_seconds)
        )
        self.retry_max_backoff_seconds = float(
            os.getenv("PUBMED_RETRY_MAX_BACKOFF_SECONDS", "2")
        )
        self.cache_enabled = (
            os.getenv("PUBMED_CACHE_ENABLED", "1").strip().lower()
            not in {"0", "false", "no"}
            if cache_enabled is None
            else bool(cache_enabled)
        )
        self.cache_ttl_seconds = (
            float(os.getenv("PUBMED_CACHE_TTL_SECONDS", "300"))
            if cache_ttl_seconds is None
            else float(cache_ttl_seconds)
        )
        self.cache_max_entries = (
            int(os.getenv("PUBMED_CACHE_MAX_ENTRIES", "128"))
            if cache_max_entries is None
            else int(cache_max_entries)
        )
        if self.timeout <= 0:
            raise ValueError("timeout 必须大于 0。")
        if self.max_retries < 0:
            raise ValueError("max_retries 不能小于 0。")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds 不能小于 0。")
        if self.retry_max_backoff_seconds < 0:
            raise ValueError("retry_max_backoff_seconds 不能小于 0。")
        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds 不能小于 0。")
        if self.cache_max_entries < 1:
            raise ValueError("cache_max_entries 必须大于 0。")
        self.email = email if email is not None else os.getenv("NCBI_EMAIL")
        self.api_key = api_key if api_key is not None else os.getenv("NCBI_API_KEY")
        self.transport = transport
        self._cache: dict[str, tuple[float, list[MedicalDocument]]] = {}
        self._cache_lock = Lock()

    def _common_params(self) -> dict[str, str]:
        params = {"retmode": "json"}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _request(
        self,
        client: httpx.Client,
        endpoint: str,
        params: dict[str, str],
    ) -> httpx.Response:
        """Send one request with bounded retries for transient failures."""

        url = f"{self.base_url}/{endpoint}"
        last_error: httpx.HTTPError | None = None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = client.get(url, params=params)
                response.raise_for_status()
                logger.info(
                    "pubmed_request_succeeded",
                    extra={
                        "request_id": get_request_id(),
                        "endpoint": endpoint,
                        "attempt": attempt + 1,
                        "status_code": response.status_code,
                        "duration_ms": round(
                            (time.perf_counter() - started) * 1000, 3
                        ),
                    },
                )
                return response
            except httpx.TimeoutException as exc:
                last_error = exc
                retryable = True
                error_kind = "timeout"
                status_code = None
            except httpx.HTTPStatusError as exc:
                last_error = exc
                status_code = exc.response.status_code
                retryable = status_code == 429 or status_code >= 500
                error_kind = f"http_{status_code}"
            except httpx.RequestError as exc:
                last_error = exc
                retryable = True
                error_kind = "request_error"
                status_code = None
            except httpx.HTTPError as exc:
                last_error = exc
                retryable = True
                error_kind = "http_error"
                status_code = None

            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            if not retryable or attempt >= self.max_retries:
                logger.warning(
                    "pubmed_request_failed",
                    extra={
                        "request_id": get_request_id(),
                        "endpoint": endpoint,
                        "attempt": attempt + 1,
                        "status_code": status_code,
                        "error_kind": error_kind,
                        "duration_ms": elapsed_ms,
                    },
                )
                raise last_error

            delay = min(
                self.retry_backoff_seconds * (2**attempt),
                self.retry_max_backoff_seconds,
            )
            logger.warning(
                "pubmed_request_retry",
                extra={
                    "request_id": get_request_id(),
                    "endpoint": endpoint,
                    "attempt": attempt + 1,
                    "next_attempt": attempt + 2,
                    "status_code": status_code,
                    "error_kind": error_kind,
                    "duration_ms": elapsed_ms,
                    "backoff_seconds": delay,
                },
            )
            time.sleep(delay)

        # The loop always returns or raises.  This keeps type checkers aware of
        # the failure contract if the implementation changes later.
        raise PubMedClientError("PubMed 请求失败。")

    def _get_json(self, client: httpx.Client, endpoint: str, params: dict[str, str]):
        return self._request(client, endpoint, params).json()

    def _get_xml(self, client: httpx.Client, endpoint: str, params: dict[str, str]) -> str:
        return self._request(client, endpoint, params).text

    @staticmethod
    def _cache_key(query: str, limit: int) -> str:
        return f"{query.strip().casefold()}\x00{limit}"

    @staticmethod
    def _copy_documents(documents: list[MedicalDocument]) -> list[MedicalDocument]:
        return [document.model_copy(deep=True) for document in documents]

    def _get_cached(self, key: str) -> list[MedicalDocument] | None:
        if not self.cache_enabled or self.cache_ttl_seconds == 0:
            return None
        now = time.monotonic()
        with self._cache_lock:
            item = self._cache.get(key)
            if item is None:
                return None
            created, documents = item
            if now - created > self.cache_ttl_seconds:
                self._cache.pop(key, None)
                return None
            return self._copy_documents(documents)

    def _set_cached(self, key: str, documents: list[MedicalDocument]) -> None:
        if not self.cache_enabled or self.cache_ttl_seconds == 0:
            return
        with self._cache_lock:
            self._cache[key] = (time.monotonic(), self._copy_documents(documents))
            while len(self._cache) > self.cache_max_entries:
                oldest_key = min(self._cache, key=lambda item: self._cache[item][0])
                self._cache.pop(oldest_key, None)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

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
        normalized_query = query.strip()
        if not normalized_query:
            return []

        cache_key = self._cache_key(normalized_query, limit)
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.info(
                "pubmed_search_cache_hit",
                extra={
                    "request_id": get_request_id(),
                    "limit": limit,
                    "result_count": len(cached),
                },
            )
            return cached

        try:
            with httpx.Client(
                timeout=self.timeout,
                transport=self.transport,
                headers={"User-Agent": "medical-evidence-agent/0.2"},
            ) as client:
                search_params = self._common_params()
                search_params.update(
                    {"db": "pubmed", "term": normalized_query, "retmax": str(limit)}
                )
                payload: dict[str, Any] = self._get_json(client, "esearch.fcgi", search_params)
                ids = payload.get("esearchresult", {}).get("idlist", [])[:limit]
                ids = [str(item) for item in ids]
                if not ids:
                    self._set_cached(cache_key, [])
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

        documents = [parsed[pmid] for pmid in ids if pmid in parsed]
        self._set_cached(cache_key, documents)
        return self._copy_documents(documents)
