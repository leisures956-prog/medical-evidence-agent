from __future__ import annotations

import numpy as np

from app.corpus_snapshot import BenchmarkQuestion, build_snapshot
from app.cross_encoder_reranker import CrossEncoderReranker
from app.local_vector_store import LocalVectorStore, chunk_documents
from app.medical_models import MedicalDocument
from app.metadata_filter import RetrievalFilters
from scripts.run_retrieval_benchmark import evaluate_retrieval_snapshot


class FakeEmbedder:
    """Small deterministic encoder so unit tests never download a model."""

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        rows = []
        for text in texts:
            value = text.lower()
            vector = np.array(
                [
                    value.count("lung"),
                    value.count("cancer"),
                    value.count("diabetes"),
                    1.0,
                ],
                dtype=np.float32,
            )
            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm:
                    vector = vector / norm
            rows.append(vector)
        return np.asarray(rows)


class FakeCrossEncoder:
    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        del batch_size, show_progress_bar
        return np.asarray(
            [
                4.0 if "lung cancer" in passage.lower() else -3.0
                for _, passage in pairs
            ],
            dtype=np.float32,
        )


def _document(pmid: str, title: str, abstract: str = "") -> MedicalDocument:
    return MedicalDocument(
        id=f"pubmed:{pmid}",
        source="pubmed",
        title=title,
        abstract=abstract,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        metadata={"pmid": pmid},
    )


def _bundle(complete_labels: bool = True):
    question = BenchmarkQuestion(id="q01", question="lung cancer")
    bundle = build_snapshot(
        [question],
        {
            "q01": [
                _document("1", "Lung cancer immunotherapy"),
                _document("2", "Diabetes treatment"),
            ]
        },
    )
    if complete_labels:
        bundle.labels.labels[0].relevance = 2
        bundle.labels.labels[1].relevance = 0
    return bundle


def test_benchmark_compares_all_methods_on_same_complete_case():
    result = evaluate_retrieval_snapshot(
        _bundle(),
        embedder=FakeEmbedder(),
    )

    assert set(result["methods"]) == {"keyword", "vector", "hybrid"}
    assert result["methods"]["keyword"]["completed_cases"] == 1
    assert result["methods"]["vector"]["metrics"]["recall"]["@1"] == 1.0
    assert result["methods"]["hybrid"]["metrics"]["mrr"] == 1.0
    assert result["comparison"]["recall@1"]["vector"] == 1.0
    assert result["timing"]["query_latency_excludes_model_load"] is True
    assert result["methods"]["vector"]["metrics"]["average_index_build_ms"] >= 0


def test_incomplete_labels_are_reported_without_metrics():
    result = evaluate_retrieval_snapshot(
        _bundle(complete_labels=False),
        methods=("keyword",),
    )

    method = result["methods"]["keyword"]
    assert method["completed_cases"] == 0
    assert method["incomplete_cases"] == 1
    assert method["metrics"]["mrr"] is None
    assert method["cases"][0]["status"] == "incomplete_labels"


def test_candidate_pool_is_explicit_and_rank_rows_are_traceable():
    result = evaluate_retrieval_snapshot(
        _bundle(),
        methods=("keyword",),
        candidate_scope="question_pool",
    )

    case = result["methods"]["keyword"]["cases"][0]
    assert case["candidate_count"] == 2
    assert case["ranked"][0]["pmid"] == "1"
    assert result["metric_note"]


def test_benchmark_can_compare_persistent_methods(tmp_path):
    bundle = _bundle()
    documents = [
        _document("1", "Lung cancer immunotherapy"),
        _document("2", "Diabetes treatment"),
    ]
    records = chunk_documents(
        documents,
        corpus_version="test-v1",
        chunk_size=1200,
        chunk_overlap=120,
    )
    fake = FakeEmbedder()
    store = LocalVectorStore(tmp_path / "persistent-index")
    store.rebuild(
        records,
        fake.encode([record.chunk_text for record in records]),
        corpus_version="test-v1",
        model_name="fake-model",
    )

    result = evaluate_retrieval_snapshot(
        bundle,
        methods=(
            "vector_persisted",
            "hybrid_persisted",
            "hybrid_persisted_reranked",
            "hybrid_persisted_cross_reranked",
            "hybrid_persisted_cross_quality_reranked",
            "advanced_persisted",
            "advanced_persisted_filtered",
        ),
        embedder=fake,
        persistent_index_dir=store.root,
        cross_encoder_reranker=CrossEncoderReranker(
            model=FakeCrossEncoder()
        ),
        retrieval_filters=RetrievalFilters(require_abstract=True),
    )

    assert result["methods"]["vector_persisted"]["completed_cases"] == 1
    assert result["methods"]["hybrid_persisted"]["completed_cases"] == 1
    assert result["methods"]["hybrid_persisted_reranked"]["completed_cases"] == 1
    assert result["methods"]["hybrid_persisted_cross_reranked"]["completed_cases"] == 1
    assert result["methods"]["hybrid_persisted_cross_quality_reranked"]["completed_cases"] == 1
    assert result["methods"]["advanced_persisted"]["completed_cases"] == 1
    assert result["methods"]["advanced_persisted_filtered"]["completed_cases"] == 1
    advanced_case = result["methods"]["advanced_persisted"]["cases"][0]
    assert advanced_case["retrieval_diagnostics"]["query_variant_count"] >= 2
    assert advanced_case["retrieval_diagnostics"]["duplicates_removed"] >= 1
    filtered_case = result["methods"]["advanced_persisted_filtered"]["cases"][0]
    assert filtered_case["retrieval_diagnostics"]["filtered_out_count"] == 2
    assert result["methods"]["vector_persisted"]["cases"][0]["ranked"][0]["pmid"] == "1"
    assert result["timing"]["persistent_index_loaded_once"] is True
    assert result["timing"]["persistent_index_load_ms"] >= 0
    assert result["timing"]["cross_encoder_loaded_once"] is True
    assert result["timing"]["cross_encoder_model_load_ms"] >= 0
    assert result["metric_scopes"]["relevance"] == "complete_labeled_cases_only"
    assert result["metric_scopes"]["quality"] == "all_cases"
    assert result["comparison"]["median_query_latency_ms"]["hybrid_persisted"] >= 0
    assert result["comparison"]["p95_query_latency_ms"]["hybrid_persisted"] >= 0
    assert "quality_flags" in result["methods"]["hybrid_persisted_reranked"]["cases"][0]["ranked"][0]
    assert "abstract_missing_rate@5" in result["comparison"]
