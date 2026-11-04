# Stage 4 Retrieval Ablation Report (Pilot)

## Experiment scope

- Corpus version: `pubmed-snapshot-v1`
- Annotation version: `labels-v1`
- Candidate scope: `question_pool`
- Relevance metrics: `complete_labeled_cases_only`
- Quality metrics: `all_cases`
- Latency metrics: `all_cases`
- Recall/MRR/nDCG are candidate-pool pilot metrics, not PubMed-wide recall.

## Filter configuration

```json
{
  "year_from": 2020,
  "year_to": 2026,
  "require_abstract": true,
  "exclude_article_types": [
    "editorial",
    "correction"
  ]
}
```

## Results

| Method | Labeled | Recall@5 | nDCG@5 | MRR | Missing abstract@5 | Editorial@5 | Avg ms | Median ms | P95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| keyword | 3/12 | 0.240 | 0.668 | 1.000 | 0.000 | 0.000 | 2.1 | 2.1 | 2.3 |
| vector | 3/12 | 0.279 | 1.000 | 1.000 | 0.050 | 0.017 | 16.6 | 15.2 | 31.8 |
| hybrid | 3/12 | 0.279 | 1.000 | 1.000 | 0.033 | 0.017 | 17.1 | 16.0 | 27.3 |
| vector_persisted | 3/12 | 0.279 | 1.000 | 1.000 | 0.050 | 0.017 | 17.0 | 15.7 | 26.7 |
| hybrid_persisted | 3/12 | 0.279 | 1.000 | 1.000 | 0.033 | 0.017 | 18.7 | 17.8 | 27.1 |
| hybrid_persisted_reranked | 3/12 | 0.279 | 0.968 | 1.000 | 0.000 | 0.000 | 19.1 | 18.6 | 24.6 |
| hybrid_persisted_cross_reranked | 3/12 | 0.279 | 1.000 | 1.000 | 0.033 | 0.017 | 1339.2 | 1276.0 | 1647.5 |
| hybrid_persisted_cross_quality_reranked | 3/12 | 0.279 | 0.971 | 1.000 | 0.017 | 0.000 | 1368.2 | 1304.7 | 1709.2 |
| advanced_persisted | 3/12 | 0.279 | 0.971 | 1.000 | 0.017 | 0.000 | 1365.0 | 1297.2 | 1669.8 |
| advanced_persisted_filtered | 3/12 | 0.279 | 0.938 | 1.000 | 0.000 | 0.000 | 1318.1 | 1240.9 | 1795.8 |

## Observed pilot findings

- Rule-based quality reranking kept Recall@5 at 0.279, reduced missing-abstract@5 from 0.033 to 0.000, and changed nDCG@5 from 1.000 to 0.968.
- Cross-Encoder reranking showed no observed Recall@5 gain in this pilot (0.279 vs 0.279) while average latency increased from 18.7 ms to 1339.2 ms.
- Multi-query retrieval and deduplication expanded candidate coverage, but no additional Recall@5 gain was observed over the quality-aware single-query pipeline (0.279).
- Hard filtering reduced missing-abstract@5 from 0.017 to 0.000, while nDCG@5 changed from 0.971 to 0.938.

## Recommended modes

- Balanced default: `hybrid_persisted` — strong pilot relevance with low local latency.
- Evidence-quality default: `hybrid_persisted_reranked` — removes most missing-abstract/editorial results without Cross-Encoder latency.
- Strict demonstration mode: `advanced_persisted` with explicit filters — use when traceable evidence completeness matters more than latency.

## Interpretation guardrails

- Only 3/12 questions have complete relevance labels.
- Quality and latency summaries use all questions.
- A method with more components is not automatically better.
- Multi-query coverage and deduplication are engineering outcomes; a Recall gain must be demonstrated separately.
- Hard filters can improve evidence completeness while lowering graded ranking quality.

## Current limitations

- Small candidate-pool annotation rather than exhaustive PubMed judgments.
- Query latency is measured locally and excludes model/index loading.
- The deterministic query expansion dictionary covers only selected medical concepts.
- Results should be described as pilot findings until more questions are fully labeled.
