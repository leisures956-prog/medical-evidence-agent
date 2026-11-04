"""Build a D-drive local persistent vector index from JSON or JSONL documents."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.local_vector_store import LocalVectorStore, chunk_documents
from app.medical_models import MedicalDocument
from app.persistent_retriever import LocalSentenceTransformerEmbedder


def _load_payload(path: Path) -> list[dict[str, Any]]:
    try:
        if path.suffix.lower() == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("documents", []) if isinstance(payload, dict) else payload
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取文档文件：{path}") from exc
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("文档文件必须是 JSON 数组、documents 数组或 JSONL。")
    return rows


def load_documents(path: Path) -> list[MedicalDocument]:
    documents: list[MedicalDocument] = []
    for row in _load_payload(path):
        document_id = row.get("id") or row.get("pmid")
        if not document_id:
            raise ValueError("每条文档必须包含 id 或 pmid。")
        abstract = row.get("abstract", row.get("content", "")) or ""
        metadata = dict(row.get("metadata") or {})
        if row.get("pmid") and "pmid" not in metadata:
            metadata["pmid"] = str(row["pmid"])
        for metadata_key in (
            "fetched_at",
            "corpus_version",
            "content_sha256",
            "candidate_question_ids",
        ):
            if row.get(metadata_key) is not None and metadata_key not in metadata:
                metadata[metadata_key] = row[metadata_key]
        source = row.get("source") or ("pubmed" if row.get("pmid") else "local")
        source_url = row.get("url") or row.get("source_url")
        documents.append(
            MedicalDocument(
                id=str(document_id),
                source=str(source),
                title=str(row.get("title", "")),
                abstract=str(abstract),
                authors=[str(item) for item in (row.get("authors") or [])],
                year=int(row["year"]) if row.get("year") is not None else None,
                doi=str(row["doi"]) if row.get("doi") else None,
                journal=str(row["journal"]) if row.get("journal") else None,
                url=str(source_url) if source_url else None,
                keywords=[str(item) for item in (row.get("keywords") or [])],
                metadata=metadata,
            )
        )
    return documents


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="构建 D 盘本地持久化向量索引。")
    parser.add_argument(
        "--documents",
        type=Path,
        default=project_root / "data" / "documents.json",
        help="JSON 或 JSONL 文档文件",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "artifacts" / "local_vector_index_v1",
        help="索引输出目录",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"),
        help="Embedding 模型路径或名称",
    )
    parser.add_argument("--corpus-version", default="local-documents-v1")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()

    documents = load_documents(args.documents)
    records = chunk_documents(
        documents,
        corpus_version=args.corpus_version,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    embedder = LocalSentenceTransformerEmbedder(args.model)
    embeddings = embedder.encode([record.chunk_text for record in records])
    store = LocalVectorStore(args.output)
    document_count = store.rebuild(
        records,
        embeddings,
        corpus_version=args.corpus_version,
        model_name=args.model,
        source_path=str(args.documents.resolve()),
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(
        json.dumps(
            {
                "status": "built",
                "index_dir": str(args.output.resolve()),
                "corpus_version": args.corpus_version,
                "model": args.model,
                "document_count": document_count,
                "chunk_count": len(records),
                "embedding_dimension": int(embeddings.shape[1]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
