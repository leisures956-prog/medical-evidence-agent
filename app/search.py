from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


class ExternalServiceError(RuntimeError):
    """Raised when an external evidence provider cannot be reached or parsed."""


class VectorSearchError(RuntimeError):
    """Raised when the optional local embedding stack is unavailable."""


@dataclass(frozen=True)
class SearchHit:
    id: str
    title: str
    content: str
    score: float
    source: str = "local"
    url: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    doi: str | None = None
    journal: str | None = None


class DocumentStore:
    def __init__(self, path: Path | None = None):
        self.path = path or Path(__file__).resolve().parents[1] / "data" / "documents.json"

    def load(self) -> list[dict[str, Any]]:
        try:
            with self.path.open("r", encoding="utf-8") as file:
                documents = json.load(file)
        except FileNotFoundError as exc:
            raise RuntimeError(f"找不到本地文档文件：{self.path}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"本地文档 JSON 格式错误：{self.path}") from exc
        except OSError as exc:
            raise RuntimeError(f"读取本地文档失败：{self.path}") from exc

        if not isinstance(documents, list):
            raise RuntimeError("本地文档文件必须是 JSON 数组。")

        return documents


class PubMedClient:
    """Small client for NCBI E-utilities ESearch + ESummary."""

    def __init__(
        self,
        base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        timeout: float | None = None,
        email: str | None = None,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout or float(os.getenv("PUBMED_TIMEOUT_SECONDS", "10"))
        self.email = email if email is not None else os.getenv("NCBI_EMAIL")
        self.api_key = api_key if api_key is not None else os.getenv("NCBI_API_KEY")
        self.transport = transport

    def _common_params(self) -> dict[str, str]:
        params: dict[str, str] = {"retmode": "json"}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _get_json(
        self,
        client: httpx.Client,
        endpoint: str,
        params: dict[str, str],
    ) -> dict[str, Any]:
        response = client.get(f"{self.base_url}/{endpoint}", params=params)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("外部接口返回的数据不是 JSON 对象")
        return payload

    def search(self, query: str, limit: int) -> list[SearchHit]:
        if not query.strip():
            return []

        try:
            with httpx.Client(
                timeout=self.timeout,
                transport=self.transport,
                headers={"User-Agent": "medical-evidence-agent/0.1"},
            ) as client:
                search_params = self._common_params()
                search_params.update(
                    {"db": "pubmed", "term": query.strip(), "retmax": str(limit)}
                )
                search_payload = self._get_json(client, "esearch.fcgi", search_params)
                ids = search_payload["esearchresult"]["idlist"]
                ids = [str(item) for item in ids[:limit]]

                if not ids:
                    return []

                summary_params = self._common_params()
                summary_params.update({"db": "pubmed", "id": ",".join(ids)})
                summary_payload = self._get_json(client, "esummary.fcgi", summary_params)

        except httpx.TimeoutException as exc:
            raise ExternalServiceError("PubMed 请求超时，请稍后重试。") from exc
        except httpx.HTTPError as exc:
            raise ExternalServiceError("PubMed 请求失败，请稍后重试。") from exc
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ExternalServiceError("PubMed 返回结果格式异常。") from exc

        result = summary_payload.get("result", {})
        hits: list[SearchHit] = []
        for pmid in ids:
            item = result.get(pmid, {})
            if not isinstance(item, dict):
                continue

            title = str(item.get("title") or f"PubMed article {pmid}")
            details = [title]
            for key in ("fulljournalname", "source", "pubdate", "sortfirstauthor"):
                value = item.get(key)
                if value:
                    details.append(str(value))

            hits.append(
                SearchHit(
                    id=f"pubmed:{pmid}",
                    title=title,
                    content="；".join(details),
                    score=1.0,
                    source="pubmed",
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                )
            )

        return hits


class LocalVectorIndex:
    """Lazy local embedding index backed by sentence-transformers and NumPy."""

    def __init__(self, documents: list[dict[str, Any]], model_name: str | None = None):
        self.documents = documents
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        self._model = None
        self._matrix = None

    def _ensure_index(self):
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise VectorSearchError(
                "本地向量检索尚未安装依赖，请运行："
                "python -m pip install -r requirements-rag.txt"
            ) from exc

        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # model download/configuration errors
                raise VectorSearchError(
                    f"Embedding 模型加载失败：{self.model_name}。"
                ) from exc

        if self._matrix is None:
            texts = [
                f"{document.get('title', '')}\n{document.get('content', '')}"
                for document in self.documents
            ]
            self._matrix = np.asarray(
                self._model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )

        return np

    def search(self, query: str, limit: int) -> list[SearchHit]:
        if not query.strip():
            return []

        np = self._ensure_index()
        query_vector = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        scores = self._matrix @ query_vector
        indices = np.argsort(-scores)[:limit]

        results: list[SearchHit] = []
        for index in indices:
            document = self.documents[int(index)]
            results.append(
                SearchHit(
                    id=str(document["id"]),
                    title=str(document.get("title", "")),
                    content=str(document.get("content", "")),
                    score=float(scores[index]),
                    source="vector",
                    url=document.get("url"),
                )
            )
        return results


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)


class SearchService:
    def __init__(
        self,
        document_store: DocumentStore | None = None,
        pubmed_client: PubMedClient | None = None,
        vector_index: LocalVectorIndex | None = None,
    ):
        self.documents = (document_store or DocumentStore()).load()
        self.pubmed_client = pubmed_client or RichPubMedAdapter()
        self.vector_index = vector_index or LocalVectorIndex(self.documents)

    def _search_local(self, query: str, limit: int) -> list[SearchHit]:
        terms = _tokens(query)
        results: list[SearchHit] = []

        for document in self.documents:
            title = str(document.get("title", ""))
            content = str(document.get("content", ""))
            searchable = f"{title} {content}".lower()
            score = sum(searchable.count(term) for term in terms)

            if score > 0:
                results.append(
                    SearchHit(
                        id=str(document["id"]),
                        title=title,
                        content=content,
                        score=float(score),
                        source=str(document.get("source", "local")),
                        url=document.get("url"),
                    )
                )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def search(self, query: str, limit: int, source: str = "local") -> tuple[list[SearchHit], int]:
        normalized_source = getattr(source, "value", source)
        normalized_source = str(normalized_source).lower()
        if not query.strip():
            raise ValueError("query 不能为空")
        if normalized_source == "hybrid":
            from app.hybrid_retriever import HybridRetriever
            from app.semantic_retriever import SemanticRetriever, EmbeddingError
            documents = [medical_document(item) for item in self.documents]
            if not hasattr(self, "_hybrid"):
                self._hybrid = HybridRetriever(documents, SemanticRetriever(documents))
            try:
                hits = self._hybrid.search(query, top_k=limit)
            except EmbeddingError as exc:
                raise VectorSearchError(str(exc)) from exc
            results = [document_hit(hit.document, hit.score) for hit in hits]
            return results, len(results)
        if normalized_source not in {"local", "pubmed", "vector", "all"}:
            raise ValueError("source 必须是 local、pubmed、vector 或 all。")

        if not query.strip():
            raise ValueError("query 不能为空")

        results: list[SearchHit] = []
        if normalized_source in {"local", "all"}:
            results.extend(self._search_local(query, limit))
        if normalized_source in {"pubmed", "all"}:
            results.extend(self.pubmed_client.search(query, limit))
        if normalized_source in {"vector", "all"}:
            results.extend(self.vector_index.search(query, limit))

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit], len(results)


def medical_document(item):
    from app.medical_models import MedicalDocument
    if isinstance(item, dict):
        payload = dict(item)
        payload.setdefault("abstract", payload.get("content", ""))
    else:
        from dataclasses import asdict
        payload = asdict(item)
        payload["abstract"] = payload.pop("content", "")
        payload["authors"] = payload.get("authors") or []
    return MedicalDocument(**payload)


def document_hit(document, score=1.0):
    return SearchHit(id=document.id, title=document.title, content=document.abstract,
                     score=score, source=document.source, url=document.url,
                     authors=document.authors, year=document.year, doi=document.doi,
                     journal=document.journal)


class RichPubMedAdapter:
    def __init__(self):
        from app.pubmed_client_v2 import PubMedClientV2
        self.client = PubMedClientV2()

    def search(self, query, limit):
        from app.pubmed_client_v2 import PubMedClientError
        try:
            return [document_hit(doc) for doc in self.client.search(query, limit)]
        except PubMedClientError as exc:
            raise ExternalServiceError(str(exc)) from exc
