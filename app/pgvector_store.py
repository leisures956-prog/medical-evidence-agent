"""PostgreSQL + pgvector persistence for versioned document embeddings.

The adapter is intentionally optional at import time.  The ordinary offline
test suite can import the project without installing ``psycopg`` or running a
database; those dependencies are required only when a persistent search is
actually used.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from app.medical_models import MedicalDocument, RetrievalHit


class PgVectorError(RuntimeError):
    """Base error for persistent vector storage operations."""


class PgVectorUnavailable(PgVectorError):
    """Raised when psycopg, the DSN, or the database is unavailable."""


@dataclass(frozen=True)
class VectorRecord:
    """One deterministic document chunk and its normalized embedding."""

    document: MedicalDocument
    chunk_id: str
    chunk_index: int
    chunk_text: str
    embedding: tuple[float, ...]
    content_sha256: str


def content_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def document_pmid(document: MedicalDocument) -> str | None:
    value = document.metadata.get("pmid")
    if value:
        return str(value)
    if document.id.startswith("pubmed:"):
        return document.id.split(":", 1)[1]
    return None


def vector_literal(values: Iterable[float]) -> str:
    """Format a vector safely for a ``%s::vector`` PostgreSQL parameter."""

    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("embedding 不能为空。")
    if any(not math.isfinite(value) for value in numbers):
        raise ValueError("embedding 不能包含 NaN 或无穷大。")
    return "[" + ",".join(f"{value:.9g}" for value in numbers) + "]"


def schema_statements(dimension: int) -> tuple[str, ...]:
    """Return idempotent schema SQL for one embedding dimension."""

    if dimension < 1:
        raise ValueError("embedding dimension 必须大于 0。")
    return (
        "CREATE EXTENSION IF NOT EXISTS vector",
        """
        CREATE TABLE IF NOT EXISTS rag_vector_store_meta (
            store_name TEXT PRIMARY KEY,
            embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
            embedding_model TEXT NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS rag_corpus_versions (
            corpus_version TEXT PRIMARY KEY,
            embedding_model TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
            document_count INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS rag_documents (
            document_id TEXT NOT NULL,
            pmid TEXT,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            abstract TEXT NOT NULL DEFAULT '',
            authors JSONB NOT NULL DEFAULT '[]'::jsonb,
            year INTEGER,
            doi TEXT,
            journal TEXT,
            source_url TEXT,
            content_sha256 TEXT NOT NULL,
            corpus_version TEXT NOT NULL REFERENCES rag_corpus_versions(corpus_version)
                ON DELETE CASCADE,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (document_id, corpus_version)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS rag_document_chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            corpus_version TEXT NOT NULL REFERENCES rag_corpus_versions(corpus_version)
                ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
            chunk_text TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            UNIQUE(document_id, corpus_version, chunk_index),
            FOREIGN KEY (document_id, corpus_version)
                REFERENCES rag_documents(document_id, corpus_version)
                ON DELETE CASCADE
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS rag_embeddings (
            chunk_id TEXT PRIMARY KEY REFERENCES rag_document_chunks(chunk_id)
                ON DELETE CASCADE,
            corpus_version TEXT NOT NULL REFERENCES rag_corpus_versions(corpus_version)
                ON DELETE CASCADE,
            model_name TEXT NOT NULL,
            embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension = {dimension}),
            embedding vector({dimension}) NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        "CREATE INDEX IF NOT EXISTS rag_documents_corpus_idx "
        "ON rag_documents(corpus_version)",
        "CREATE INDEX IF NOT EXISTS rag_documents_pmid_idx ON rag_documents(pmid)",
        "CREATE INDEX IF NOT EXISTS rag_chunks_corpus_idx "
        "ON rag_document_chunks(corpus_version)",
        "CREATE INDEX IF NOT EXISTS rag_embeddings_corpus_idx "
        "ON rag_embeddings(corpus_version, model_name)",
        "CREATE INDEX IF NOT EXISTS rag_embeddings_vector_hnsw_idx "
        "ON rag_embeddings USING hnsw (embedding vector_cosine_ops)",
    )


class PgVectorStore:
    """Small transaction-safe pgvector repository.

    ``connection_factory`` is injectable for tests and must accept the DSN.
    In production the default factory lazily imports ``psycopg``.
    """

    def __init__(
        self,
        dsn: str | None = None,
        connection_factory: Callable[[str], Any] | None = None,
    ):
        self.dsn = (
            dsn
            or os.getenv("VECTOR_DATABASE_URL")
            or os.getenv("PGVECTOR_DATABASE_URL")
            or os.getenv("DATABASE_URL")
        )
        self.connection_factory = connection_factory

    def _connect(self):
        if not self.dsn:
            raise PgVectorUnavailable(
                "未配置 VECTOR_DATABASE_URL；请提供 PostgreSQL + pgvector 的连接地址。"
            )
        if self.connection_factory is not None:
            return self.connection_factory(self.dsn)
        try:
            import psycopg
        except ImportError as exc:
            raise PgVectorUnavailable(
                "缺少 psycopg，请安装 requirements-vector.txt。"
            ) from exc
        try:
            return psycopg.connect(self.dsn)
        except Exception as exc:  # psycopg's concrete error types vary by version
            raise PgVectorUnavailable("无法连接 PostgreSQL + pgvector。") from exc

    @staticmethod
    def _check_dimension(embedding: Iterable[float], dimension: int) -> tuple[float, ...]:
        values = tuple(float(value) for value in embedding)
        if len(values) != dimension:
            raise ValueError(
                f"embedding 维度不一致：收到 {len(values)}，预期 {dimension}。"
            )
        # Validate finite values and keep a tuple for deterministic reuse.
        vector_literal(values)
        return values

    def initialize(self, corpus_version: str, model_name: str, dimension: int) -> None:
        """Create the schema and record the single store-wide dimension."""

        if not corpus_version.strip() or not model_name.strip():
            raise ValueError("corpus_version 和 model_name 不能为空。")
        statements = schema_statements(dimension)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(statements[0])
                cursor.execute(statements[1])
                cursor.execute(
                    "SELECT embedding_dimension, embedding_model "
                    "FROM rag_vector_store_meta WHERE store_name = %s",
                    ("default",),
                )
                existing = cursor.fetchone()
                if existing and int(existing[0]) != dimension:
                    raise PgVectorError(
                        "当前向量库已经固定为维度 "
                        f"{existing[0]}，不能写入维度 {dimension}。请使用新的数据库或重建索引。"
                    )
                for statement in statements[2:]:
                    cursor.execute(statement)
                cursor.execute(
                    """
                    INSERT INTO rag_vector_store_meta
                        (store_name, embedding_dimension, embedding_model, updated_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (store_name) DO UPDATE SET
                        embedding_dimension = EXCLUDED.embedding_dimension,
                        embedding_model = EXCLUDED.embedding_model,
                        updated_at = NOW()
                    """,
                    ("default", dimension, model_name),
                )
                cursor.execute(
                    """
                    INSERT INTO rag_corpus_versions
                        (corpus_version, embedding_model, embedding_dimension, updated_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (corpus_version) DO UPDATE SET
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_dimension = EXCLUDED.embedding_dimension
                    """,
                    (corpus_version, model_name, dimension),
                )
            connection.commit()

    def upsert(
        self,
        records: Iterable[VectorRecord],
        *,
        corpus_version: str,
        model_name: str,
    ) -> int:
        """Insert or replace documents, chunks and embeddings atomically."""

        batch = list(records)
        if not batch:
            return 0
        dimension = len(batch[0].embedding)
        for record in batch:
            self._check_dimension(record.embedding, dimension)
        self.initialize(corpus_version, model_name, dimension)

        with self._connect() as connection:
            with connection.cursor() as cursor:
                seen_documents: set[str] = set()
                for record in batch:
                    document = record.document
                    if document.id not in seen_documents:
                        seen_documents.add(document.id)
                        document_hash = str(
                            document.metadata.get("content_sha256")
                            or content_sha256(document.retrieval_text())
                        )
                        cursor.execute(
                            """
                            INSERT INTO rag_documents
                                (document_id, pmid, source, title, abstract, authors,
                                 year, doi, journal, source_url, content_sha256,
                                 corpus_version, metadata, updated_at)
                            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s,
                                    %s, %s, %s::jsonb, NOW())
                            ON CONFLICT (document_id, corpus_version) DO UPDATE SET
                                pmid = EXCLUDED.pmid,
                                source = EXCLUDED.source,
                                title = EXCLUDED.title,
                                abstract = EXCLUDED.abstract,
                                authors = EXCLUDED.authors,
                                year = EXCLUDED.year,
                                doi = EXCLUDED.doi,
                                journal = EXCLUDED.journal,
                                source_url = EXCLUDED.source_url,
                                content_sha256 = EXCLUDED.content_sha256,
                                corpus_version = EXCLUDED.corpus_version,
                                metadata = EXCLUDED.metadata,
                                updated_at = NOW()
                            """,
                            (
                                document.id,
                                document_pmid(document),
                                document.source,
                                document.title,
                                document.abstract,
                                json.dumps(document.authors, ensure_ascii=False),
                                document.year,
                                document.doi,
                                document.journal,
                                document.url,
                                document_hash,
                                corpus_version,
                                json.dumps(document.metadata, ensure_ascii=False),
                            ),
                        )
                        cursor.execute(
                            """
                            DELETE FROM rag_document_chunks
                            WHERE document_id = %s AND corpus_version = %s
                            """,
                            (document.id, corpus_version),
                        )
                    cursor.execute(
                        """
                        INSERT INTO rag_document_chunks
                            (chunk_id, document_id, corpus_version, chunk_index,
                             chunk_text, content_sha256)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            corpus_version = EXCLUDED.corpus_version,
                            chunk_index = EXCLUDED.chunk_index,
                            chunk_text = EXCLUDED.chunk_text,
                            content_sha256 = EXCLUDED.content_sha256
                        """,
                        (
                            record.chunk_id,
                            document.id,
                            corpus_version,
                            record.chunk_index,
                            record.chunk_text,
                            record.content_sha256,
                        ),
                    )
                    cursor.execute(
                        """
                        INSERT INTO rag_embeddings
                            (chunk_id, corpus_version, model_name,
                             embedding_dimension, embedding, updated_at)
                        VALUES (%s, %s, %s, %s, %s::vector, NOW())
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            corpus_version = EXCLUDED.corpus_version,
                            model_name = EXCLUDED.model_name,
                            embedding_dimension = EXCLUDED.embedding_dimension,
                            embedding = EXCLUDED.embedding,
                            updated_at = NOW()
                        """,
                        (
                            record.chunk_id,
                            corpus_version,
                            model_name,
                            dimension,
                            vector_literal(record.embedding),
                        ),
                    )

                cursor.execute(
                    "SELECT COUNT(*) FROM rag_documents WHERE corpus_version = %s",
                    (corpus_version,),
                )
                total_documents = int(cursor.fetchone()[0])
                cursor.execute(
                    """
                    UPDATE rag_corpus_versions
                    SET document_count = %s, updated_at = NOW()
                    WHERE corpus_version = %s
                    """,
                    (total_documents, corpus_version),
                )
            connection.commit()
        return total_documents

    def list_documents(self, *, corpus_version: str) -> list[MedicalDocument]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT document_id, source, title, abstract, authors, year,
                           doi, journal, source_url, metadata
                    FROM rag_documents
                    WHERE corpus_version = %s
                    ORDER BY document_id
                    """,
                    (corpus_version,),
                )
                rows = cursor.fetchall()

        documents: list[MedicalDocument] = []
        for row in rows:
            authors = row[4] if isinstance(row[4], list) else json.loads(row[4] or "[]")
            metadata = row[9] if isinstance(row[9], dict) else json.loads(row[9] or "{}")
            documents.append(
                MedicalDocument(
                    id=str(row[0]),
                    source=str(row[1]),
                    title=str(row[2]),
                    abstract=str(row[3] or ""),
                    authors=[str(item) for item in authors],
                    year=int(row[5]) if row[5] is not None else None,
                    doi=str(row[6]) if row[6] is not None else None,
                    journal=str(row[7]) if row[7] is not None else None,
                    url=str(row[8]) if row[8] is not None else None,
                    metadata=metadata,
                )
            )
        return documents

    def search(
        self,
        query_embedding: Iterable[float],
        *,
        corpus_version: str,
        model_name: str,
        limit: int,
    ) -> list[RetrievalHit]:
        if limit < 1:
            raise ValueError("limit 必须大于 0。")
        values = tuple(float(value) for value in query_embedding)
        vector = vector_literal(values)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT d.document_id, d.source, d.title, d.abstract, d.authors,
                           d.year, d.doi, d.journal, d.source_url, d.metadata,
                           c.chunk_text, (e.embedding <=> %s::vector) AS distance
                    FROM rag_embeddings AS e
                    JOIN rag_document_chunks AS c ON c.chunk_id = e.chunk_id
                    JOIN rag_documents AS d
                        ON d.document_id = c.document_id
                       AND d.corpus_version = c.corpus_version
                    WHERE e.corpus_version = %s AND e.model_name = %s
                    ORDER BY e.embedding <=> %s::vector, d.document_id, c.chunk_index
                    LIMIT %s
                    """,
                    (vector, corpus_version, model_name, vector, limit),
                )
                rows = cursor.fetchall()

        hits: list[RetrievalHit] = []
        for row in rows:
            authors = row[4] if isinstance(row[4], list) else json.loads(row[4] or "[]")
            metadata = row[9] if isinstance(row[9], dict) else json.loads(row[9] or "{}")
            distance = float(row[11])
            document = MedicalDocument(
                id=str(row[0]),
                source=str(row[1]),
                title=str(row[2]),
                abstract=str(row[3] or ""),
                authors=[str(item) for item in authors],
                year=int(row[5]) if row[5] is not None else None,
                doi=str(row[6]) if row[6] is not None else None,
                journal=str(row[7]) if row[7] is not None else None,
                url=str(row[8]) if row[8] is not None else None,
                metadata=metadata,
            )
            hits.append(
                RetrievalHit(
                    document=document,
                    score=1.0 - distance,
                    semantic_score=1.0 - distance,
                    retrieval_method="vector_pg",
                )
            )
        return hits

    def delete_corpus(self, corpus_version: str) -> int:
        """Delete one corpus version; foreign keys cascade to chunks/vectors."""

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM rag_corpus_versions WHERE corpus_version = %s",
                    (corpus_version,),
                )
                deleted = int(cursor.rowcount or 0)
            connection.commit()
        return deleted
