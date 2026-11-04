"""D盘本地向量索引的持久化存储。

这个模块不依赖 PostgreSQL、Docker 或 WSL。它把归一化后的向量保存为
``.npy``，把文档和分块元数据保存为 ``.jsonl``，并用一个 manifest 记录
语料版本、Embedding 模型和向量维度。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from app.medical_models import MedicalDocument


class LocalVectorStoreError(RuntimeError):
    """本地向量索引不存在、损坏或元数据不一致。"""


@dataclass(frozen=True)
class LocalVectorRecord:
    """一个文档分块及其可追溯元数据。"""

    document: MedicalDocument
    chunk_id: str
    chunk_index: int
    chunk_text: str
    content_sha256: str


@dataclass(frozen=True)
class LocalVectorIndexData:
    records: list[LocalVectorRecord]
    embeddings: np.ndarray
    manifest: dict[str, Any]


def content_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def split_text(text: str, *, max_chars: int = 1200, overlap: int = 120) -> list[str]:
    """确定性切分文本，保留少量上下文重叠。"""

    if max_chars < 1:
        raise ValueError("max_chars 必须大于 0。")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap 必须大于等于 0 且小于 max_chars。")
    value = (text or "").strip()
    if not value:
        return []

    chunks: list[str] = []
    step = max_chars - overlap
    start = 0
    while start < len(value):
        end = min(len(value), start + max_chars)
        chunks.append(value[start:end])
        if end == len(value):
            break
        start += step
    return chunks


def chunk_documents(
    documents: Iterable[MedicalDocument],
    *,
    corpus_version: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 120,
) -> list[LocalVectorRecord]:
    """将文档转换为稳定的分块记录。"""

    if not corpus_version.strip():
        raise ValueError("corpus_version 不能为空。")

    records: list[LocalVectorRecord] = []
    for document in documents:
        text = document.retrieval_text() or document.title
        chunks = split_text(text, max_chars=chunk_size, overlap=chunk_overlap)
        for chunk_index, chunk_text in enumerate(chunks):
            records.append(
                LocalVectorRecord(
                    document=document,
                    chunk_id=f"{corpus_version}:{document.id}:{chunk_index}",
                    chunk_index=chunk_index,
                    chunk_text=chunk_text,
                    content_sha256=content_sha256(chunk_text),
                )
            )
    return records


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalVectorStore:
    """基于本地文件的版本化向量仓库。"""

    schema_version = "local-vector-index-v1"

    def __init__(self, root: Path | str):
        self.root = Path(root)

    @property
    def vectors_path(self) -> Path:
        return self.root / "vectors.npy"

    @property
    def records_path(self) -> Path:
        return self.root / "records.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def exists(self) -> bool:
        return all(
            path.exists()
            for path in (self.vectors_path, self.records_path, self.manifest_path)
        )

    def _read_records(self) -> list[LocalVectorRecord]:
        records: list[LocalVectorRecord] = []
        try:
            with self.records_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                        records.append(
                            LocalVectorRecord(
                                document=MedicalDocument.model_validate(
                                    payload["document"]
                                ),
                                chunk_id=str(payload["chunk_id"]),
                                chunk_index=int(payload["chunk_index"]),
                                chunk_text=str(payload["chunk_text"]),
                                content_sha256=str(payload["content_sha256"]),
                            )
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise LocalVectorStoreError(
                            f"索引记录格式错误：{self.records_path} 第 {line_number} 行。"
                        ) from exc
        except FileNotFoundError as exc:
            raise LocalVectorStoreError(f"找不到索引记录：{self.records_path}") from exc
        except OSError as exc:
            raise LocalVectorStoreError(f"读取索引记录失败：{self.records_path}") from exc
        return records

    def load(self) -> LocalVectorIndexData:
        """读取并校验 manifest、记录和向量矩阵。"""

        if not self.exists:
            raise LocalVectorStoreError(
                f"本地向量索引不完整：{self.root}。请先运行构建脚本。"
            )
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LocalVectorStoreError(f"读取索引 manifest 失败：{self.manifest_path}") from exc

        if not isinstance(manifest, dict):
            raise LocalVectorStoreError("索引 manifest 必须是 JSON 对象。")
        if manifest.get("schema_version") != self.schema_version:
            raise LocalVectorStoreError(
                f"索引版本不兼容：{manifest.get('schema_version')!r}。"
            )

        records = self._read_records()
        try:
            embeddings = np.load(self.vectors_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise LocalVectorStoreError(f"读取向量矩阵失败：{self.vectors_path}") from exc

        if embeddings.ndim != 2:
            raise LocalVectorStoreError("向量矩阵必须是二维数组。")
        if len(records) != embeddings.shape[0]:
            raise LocalVectorStoreError(
                f"记录数与向量行数不一致：{len(records)} != {embeddings.shape[0]}。"
            )
        dimension = int(manifest.get("embedding_dimension", 0))
        if dimension < 1 or embeddings.shape[1] != dimension:
            raise LocalVectorStoreError("manifest 中的向量维度与矩阵不一致。")
        if not np.isfinite(embeddings).all():
            raise LocalVectorStoreError("向量矩阵包含 NaN 或无穷大。")

        return LocalVectorIndexData(
            records=records,
            embeddings=np.asarray(embeddings, dtype=np.float32),
            manifest=manifest,
        )

    @staticmethod
    def _validate_records(records: list[LocalVectorRecord]) -> None:
        chunk_ids = [record.chunk_id for record in records]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk_id 必须唯一。")
        document_chunks = [
            (record.document.id, record.chunk_index) for record in records
        ]
        if len(document_chunks) != len(set(document_chunks)):
            raise ValueError("同一文档的 chunk_index 必须唯一。")

    def _write(
        self,
        records: list[LocalVectorRecord],
        embeddings: np.ndarray,
        *,
        corpus_version: str,
        model_name: str,
        source_path: str | None,
        chunk_size: int,
        chunk_overlap: int,
        embedding_dimension: int | None = None,
        created_at: str | None = None,
    ) -> None:
        if not corpus_version.strip() or not model_name.strip():
            raise ValueError("corpus_version 和 model_name 不能为空。")
        self._validate_records(records)

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings 必须是二维数组。")
        if matrix.shape[0] != len(records):
            raise ValueError("记录数与 embedding 行数不一致。")
        dimension = int(embedding_dimension or (matrix.shape[1] if matrix.ndim == 2 else 0))
        if dimension < 1:
            raise ValueError("embedding dimension 必须大于 0。")
        if matrix.shape[1] != dimension:
            raise ValueError("embedding dimension 与矩阵不一致。")
        if not np.isfinite(matrix).all():
            raise ValueError("embedding 不能包含 NaN 或无穷大。")

        self.root.mkdir(parents=True, exist_ok=True)
        previous_created_at = None
        if self.manifest_path.exists():
            try:
                previous = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                previous_created_at = previous.get("created_at")
            except (OSError, json.JSONDecodeError):
                previous_created_at = None

        manifest = {
            "schema_version": self.schema_version,
            "corpus_version": corpus_version,
            "model_name": model_name,
            "embedding_dimension": dimension,
            "document_count": len({record.document.id for record in records}),
            "chunk_count": len(records),
            "source_path": source_path,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "created_at": created_at or previous_created_at or _utc_now(),
            "updated_at": _utc_now(),
        }

        temporary_paths: list[Path] = []
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=self.root,
                prefix=".records.", suffix=".tmp", delete=False,
            ) as handle:
                records_tmp = Path(handle.name)
                temporary_paths.append(records_tmp)
                for record in records:
                    handle.write(
                        json.dumps(
                            {
                                "chunk_id": record.chunk_id,
                                "chunk_index": record.chunk_index,
                                "chunk_text": record.chunk_text,
                                "content_sha256": record.content_sha256,
                                "document": record.document.model_dump(mode="json"),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())

            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self.root, prefix=".vectors.", suffix=".tmp", delete=False,
            ) as handle:
                vectors_tmp = Path(handle.name)
                temporary_paths.append(vectors_tmp)
                np.save(handle, matrix)
                handle.flush()
                os.fsync(handle.fileno())

            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=self.root,
                prefix=".manifest.", suffix=".tmp", delete=False,
            ) as handle:
                manifest_tmp = Path(handle.name)
                temporary_paths.append(manifest_tmp)
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(records_tmp, self.records_path)
            os.replace(vectors_tmp, self.vectors_path)
            os.replace(manifest_tmp, self.manifest_path)
            temporary_paths.clear()
        finally:
            for path in temporary_paths:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass

    def replace_all(
        self,
        records: Iterable[LocalVectorRecord],
        embeddings: Any,
        *,
        corpus_version: str,
        model_name: str,
        source_path: str | None = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 120,
    ) -> int:
        """完整重建索引，返回文档数量。"""

        batch = list(records)
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("embeddings 必须是二维数组。")
        self._write(
            batch,
            matrix,
            corpus_version=corpus_version,
            model_name=model_name,
            source_path=source_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return len({record.document.id for record in batch})

    rebuild = replace_all

    def upsert(
        self,
        records: Iterable[LocalVectorRecord],
        embeddings: Any,
        *,
        corpus_version: str | None = None,
        model_name: str | None = None,
        source_path: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> int:
        """按文档 ID 增量替换记录，不产生重复分块。"""

        incoming = list(records)
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(incoming):
            raise ValueError("增量记录与 embedding 行数不一致。")
        self._validate_records(incoming)

        if not self.exists:
            if not corpus_version or not model_name:
                raise ValueError("首次写入必须提供 corpus_version 和 model_name。")
            return self.replace_all(
                incoming,
                matrix,
                corpus_version=corpus_version,
                model_name=model_name,
                source_path=source_path,
                chunk_size=chunk_size or 1200,
                chunk_overlap=chunk_overlap or 120,
            )

        existing = self.load()
        dimension = existing.embeddings.shape[1]
        if matrix.shape[1] != dimension:
            raise ValueError("增量 embedding 维度与已有索引不一致。")
        incoming_ids = {record.document.id for record in incoming}
        keep_indices = [
            index
            for index, record in enumerate(existing.records)
            if record.document.id not in incoming_ids
        ]
        kept_records = [existing.records[index] for index in keep_indices]
        kept_vectors = existing.embeddings[keep_indices]
        merged_records = kept_records + incoming
        merged_vectors = np.vstack([kept_vectors, matrix])
        manifest = existing.manifest
        self._write(
            merged_records,
            merged_vectors,
            corpus_version=corpus_version or str(manifest["corpus_version"]),
            model_name=model_name or str(manifest["model_name"]),
            source_path=source_path if source_path is not None else manifest.get("source_path"),
            chunk_size=chunk_size or int(manifest.get("chunk_size", 1200)),
            chunk_overlap=chunk_overlap or int(manifest.get("chunk_overlap", 120)),
            embedding_dimension=dimension,
            created_at=str(manifest.get("created_at") or _utc_now()),
        )
        return len({record.document.id for record in merged_records})

    def delete_documents(self, document_ids: Iterable[str]) -> int:
        """删除指定文档及其所有分块，返回实际删除的文档数。"""

        if not self.exists:
            return 0
        data = self.load()
        target_ids = {str(document_id) for document_id in document_ids}
        existing_ids = {record.document.id for record in data.records}
        deleted_ids = target_ids & existing_ids
        if not deleted_ids:
            return 0

        keep_indices = [
            index
            for index, record in enumerate(data.records)
            if record.document.id not in deleted_ids
        ]
        kept_records = [data.records[index] for index in keep_indices]
        if keep_indices:
            kept_vectors = data.embeddings[keep_indices]
        else:
            kept_vectors = np.empty((0, data.embeddings.shape[1]), dtype=np.float32)
        self._write(
            kept_records,
            kept_vectors,
            corpus_version=str(data.manifest["corpus_version"]),
            model_name=str(data.manifest["model_name"]),
            source_path=data.manifest.get("source_path"),
            chunk_size=int(data.manifest.get("chunk_size", 1200)),
            chunk_overlap=int(data.manifest.get("chunk_overlap", 120)),
            embedding_dimension=data.embeddings.shape[1],
            created_at=str(data.manifest.get("created_at") or _utc_now()),
        )
        return len(deleted_ids)

    def list_documents(self) -> list[MedicalDocument]:
        """返回索引中的去重文档。"""

        data = self.load()
        documents: dict[str, MedicalDocument] = {}
        for record in data.records:
            documents.setdefault(record.document.id, record.document)
        return [documents[key] for key in sorted(documents)]
