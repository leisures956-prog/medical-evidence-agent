"""Collect fixed-context answers without a web server, PubMed, or embeddings."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.answer_evaluation import load_answer_evaluation_cases
from app.evidence_gate import evaluate_evidence
from app.grounded_llm import GroundedAnswerer, OpenAICompatibleProvider, PROMPT_VERSION
from app.llm_cache import LLMResponseCache
from app.medical_models import MedicalDocument, RetrievalHit

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / 'data/benchmark/answer_eval_v1.json'
SNAPSHOT = ROOT / 'data/benchmark/pubmed_snapshot_v1/documents.jsonl'
OUTPUT = ROOT / 'artifacts/evaluation/answer_outputs_v1.json'


def checked_output(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Output must stay inside the D-drive project.')
    return path


def load_contexts(cases, snapshot: Path):
    raw = snapshot.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if any(case.corpus_sha256 != digest for case in cases):
        raise ValueError('Frozen corpus hash mismatch. Do not run a different corpus with these labels.')
    documents = {str(row['pmid']): row for row in (
        json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()
    )}
    contexts = {}
    for case in cases:
        hits = []
        for pmid in case.context_pmids:
            if pmid not in documents:
                raise ValueError(f'Missing fixed PMID: {pmid}')
            doc = documents[pmid]
            hits.append(RetrievalHit(document=MedicalDocument(
                id=f'pubmed:{pmid}', source='pubmed', title=doc['title'],
                abstract=doc['abstract'], url=doc['source_url'], year=doc.get('year'),
                metadata={'pmid': pmid},
            ), score=1.0, retrieval_method='fixed_context_evaluation'))
        actual_spans = evaluate_evidence(hits).records
        available = {(r.pmid, s.source_section, s.exact_text)
                     for r in actual_spans for s in r.evidence_spans}
        for ref in case.reference_spans:
            if (ref['pmid'], ref['source_section'], ref['exact_text']) not in available:
                raise ValueError(f'Reference span not in frozen abstract: {case.id}')
        contexts[case.id] = hits
    return contexts, digest


def collect_one(case, hits, answerer):
    started = time.perf_counter()
    decision = evaluate_evidence(hits)
    row = {'case_id': case.id, 'question': case.question}
    try:
        if not decision.allowed:
            row.update(answer=decision.message, abstained=True,
                       abstention_reason=decision.reason, citation_bindings=[])
        else:
            # Gold labels and expected claims are never sent to the model.
            result = answerer.answer(case.question, hits, evidence_records=decision.records)
            row.update(result.model_dump())
    except Exception as exc:
        row.update(answer='', abstained=False, error=f'{type(exc).__name__}: {exc}')
    row['wall_latency_ms'] = round((time.perf_counter() - started) * 1000, 3)
    return row


def save_progress(path, payload):
    path = checked_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description='Collect local Ollama answers on frozen contexts')
    parser.add_argument('--dataset', type=Path, default=DATASET)
    parser.add_argument('--snapshot', type=Path, default=SNAPSHOT)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--limit', type=int, default=None, help='Smoke-test only the first N cases')
    parser.add_argument('--case-id', action='append', help='Run only this case; repeat for multiple cases')
    parser.add_argument('--resume', action='store_true', help='Keep successes and retry failed cases')
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    output = checked_output(args.output)
    if output.exists() and not args.resume:
        parser.error('Output exists. Use --resume or a different --output (no silent overwrite).')
    cases = load_answer_evaluation_cases(args.dataset)
    contexts, corpus_hash = load_contexts(cases, args.snapshot)
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    metadata = {'collection_mode': 'fixed_context_ollama', 'model': 'qwen2.5:3b',
                'base_url': 'http://127.0.0.1:11434/v1', 'prompt_version': PROMPT_VERSION,
                'dataset_sha256': dataset_hash, 'corpus_sha256': corpus_hash}
    payload = {**metadata, 'created_at': datetime.now(timezone.utc).isoformat(), 'answers': []}
    if args.resume and output.exists():
        payload = json.loads(output.read_text(encoding='utf-8'))
        if any(payload.get(key) != value for key, value in metadata.items()):
            parser.error('Resume metadata mismatch; use a new output filename.')
    prior = {row['case_id']: row for row in payload['answers']}
    provider = OpenAICompatibleProvider(
        api_key='ollama', base_url=metadata['base_url'], model=metadata['model'],
        timeout=args.timeout, max_retries=0, max_tokens=256, temperature=0,
        cache=LLMResponseCache(enabled=False),
    )
    answerer = GroundedAnswerer(provider, max_citation_repairs=1)
    if args.case_id:
        wanted = set(args.case_id)
        unknown = wanted - {case.id for case in cases}
        if unknown:
            parser.error(f'Unknown case id: {sorted(unknown)}')
        selected = [case for case in cases if case.id in wanted]
    else:
        selected = cases[:args.limit] if args.limit else cases
    try:
        for index, case in enumerate(selected, 1):
            if case.id in prior and not prior[case.id].get('error') and not args.case_id:
                print(f'[{index}/{len(selected)}] {case.id}: saved, skipped', flush=True)
                continue
            print(f'[{index}/{len(selected)}] {case.id}: running...', flush=True)
            row = collect_one(case, contexts[case.id], answerer)
            prior[case.id] = row
            payload['answers'] = [prior[c.id] for c in cases if c.id in prior]
            save_progress(output, payload)
            print(f"  {'ERROR' if row.get('error') else 'OK'} ({row['wall_latency_ms']:.0f} ms)", flush=True)
            if row.get('error'):
                print(f"  {row['error']}", flush=True)
    except KeyboardInterrupt:
        print('Stopped. Completed cases are saved; use --resume to continue.')
    print(f'Saved: {output}')
    print('This is generation-only evaluation on fixed contexts, not live retrieval accuracy.')


if __name__ == '__main__':
    main()
