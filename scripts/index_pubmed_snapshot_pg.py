"""Index a frozen PubMed snapshot into PostgreSQL + pgvector.

Example (after PostgreSQL + pgvector and the optional dependencies are ready):

    python scripts/index_pubmed_snapshot_pg.py \
      --snapshot data/benchmark/pubmed_snapshot_v1 \
      --database-url "$env:VECTOR_DATABASE_URL"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.corpus_snapshot import load_snapshot
from app.medical_models import MedicalDocument
from app.pgvector_retriever import PersistentVectorRetriever
from app.pgvector_store import PgVectorError, PgVectorStore


def _documents_from_snapshot(snapshot_dir: Path) -> tuple[str, list[MedicalDocument]]:
    bundle = load_snapshot(snapshot_dir)
    documents = [
        MedicalDocument(
            id=f"pubmed:{item.pmid}",
            source="pubmed",
            title=item.title,
            abstract=item.abstract,
            authors=item.authors,
            year=item.year,
            doi=item.doi,
            journal=item.journal,
            url=item.source_url,
            metadata={
                "pmid": item.pmid,
                "content_sha256": item.content_sha256,
                "corpus_version": item.corpus_version,
                "candidate_question_ids": item.candidate_question_ids,
            },
        )
        for item in bundle.documents
    ]
    return bundle.manifest.corpus_version, documents


def main() -> int:
    parser = argparse.ArgumentParser(description="将冻结 PubMed 快照写入 pgvector")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("data/benchmark/pubmed_snapshot_v1"),
    )
    parser.add_argument(
        "--database-url",
        required=True,
        help="PostgreSQL 连接地址，例如 postgresql://search:search@localhost:5433/search",
    )
    parser.add_argument("--model", help="Embedding 模型路径或名称，默认读取 EMBEDDING_MODEL")
    parser.add_argument("--corpus-version", help="覆盖快照中的 corpus_version")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()

    corpus_version, documents = _documents_from_snapshot(args.snapshot)
    corpus_version = args.corpus_version or corpus_version
    retriever = PersistentVectorRetriever(
        PgVectorStore(args.database_url),
        model_name=args.model,
        corpus_version=corpus_version,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    try:
        indexed = retriever.index_documents(documents)
    except (PgVectorError, ValueError) as exc:
        parser.error(str(exc))
        return 2

    print(
        json.dumps(
            {
                "corpus_version": corpus_version,
                "document_count": indexed,
                "model": retriever.model_name,
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
