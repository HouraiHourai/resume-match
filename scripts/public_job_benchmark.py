"""Public title retrieval benchmark; separate from full-JD production evaluation.

Run prepare -> embed -> dev -> test. Selection uses dev only. Test cannot overwrite.
pyarrow is an optional local preparation dependency, not a production dependency.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time

import httpx
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from resume_match.evaluation.metrics import score_query
from resume_match.rag.bm25_store import _tokenize

ROOT = Path('evaluation/local/public-jobs')
MODEL = 'Pro/BAAI/bge-m3'
RERANK_MODEL = 'Pro/BAAI/bge-reranker-v2-m3'
SEED = 20260918


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def tokens(text):
    """Chinese unigram/bigram + English terms; no benchmark-label dictionary."""
    units = re.findall(r'[a-z0-9]+(?:[+#]+)?|[\u4e00-\u9fff]', text.casefold())
    return units + [''.join(units[i:i+2]) for i in range(len(units)-1)
                    if all(re.fullmatch(r'[\u4e00-\u9fff]', c) for c in units[i:i+2])]


def prepare():
    required = ('corpus.parquet', 'queries.parquet', 'metadata.json')
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing benchmark inputs in {ROOT}: {', '.join(missing)}. "
            "See evaluation/README.md for input requirements."
        )
    sys.path.insert(0, str(ROOT/'vendor'))
    import pyarrow.parquet as pq
    docs = pq.read_table(ROOT/'corpus.parquet').to_pylist()
    queries = pq.read_table(ROOT/'queries.parquet').to_pylist()
    def norm(s):
        return ' '.join(s.casefold().split())
    assert len({norm(d['text']) for d in docs}) == len(docs)
    assert len({norm(q['text']) for q in queries}) == len(queries)
    overlap = set(norm(d['text']) for d in docs) & set(norm(q['text']) for q in queries)
    # The published benchmark can contain an identical surface form in the
    # corpus; it is part of the task and labels decide relevance. Record it
    # explicitly instead of silently removing the labelled target.
    assert all(q['labels'] and all(0 <= i < len(docs) for i in q['labels']) for q in queries)
    order = list(range(len(queries)))
    random.Random(SEED).shuffle(order)
    manifest = {
        'source': 'https://huggingface.co/datasets/Avature/Job-Title-Similarity',
        'revision': read(ROOT/'metadata.json')['sha'], 'license': 'Apache-2.0',
        'language': 'zh', 'task': 'job-title retrieval, NOT full-JD/resume matching',
        'labels': 'published binary labels, adjudicated English annotations + human Chinese translation',
        'seed': SEED, 'dev_ids': sorted(order[:33]), 'test_ids': sorted(order[33:]),
        'embedding': MODEL, 'reranker': RERANK_MODEL,
        'parameters': {'k': 3, 'rerank_candidates': 50, 'rrf_k': 60,
                       'vector_weights_dev_grid': [0, .25, .5, .75, 1]},
        'primary_metric': 'macro nDCG@3; dev chooses weighted RRF and final deployment candidate',
        'baseline': 'BGE-M3 cosine full-corpus ranking; strong lexical baseline Chinese unigram/bigram BM25',
        'scope': 'Plain titles encoded directly, no invented skills/company fields; no production fine_rank benchmark.',
        'exact_surface_overlap_count': len(overlap),
        'input_sha256': {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                         for n in ('queries.parquet', 'corpus.parquet')},
    }
    save_new(ROOT/'frozen.json', {'manifest': manifest, 'documents': docs, 'queries': queries})
    print(f'Frozen {len(docs)} docs, {len(order[:33])} dev, {len(order[33:])} test; {len(overlap)} official surface overlaps retained.', flush=True)


def request(endpoint, payload):
    load_dotenv('.env')
    base = (os.getenv('SILICONFLOW_BASE_URL') or 'https://api.siliconflow.cn/v1').rstrip('/')
    key = os.getenv('SILICONFLOW_API_KEY')
    if not key:
        raise RuntimeError('Missing SILICONFLOW_API_KEY')
    for attempt in range(3):
        try:
            response = httpx.post(base+'/'+endpoint, json=payload,
                headers={'Authorization': 'Bearer '+key}, timeout=45)
            if response.status_code not in (429, 500, 502, 503, 504):
                if response.status_code != 200:
                    raise RuntimeError(f'{endpoint} HTTP {response.status_code}; no body/credential logging')
                return response.json()
        except httpx.TransportError:
            if attempt == 2:
                raise RuntimeError(f'{endpoint} transport failure after 3 attempts') from None
        if attempt < 2:
            time.sleep(.5*2**attempt)
    raise RuntimeError(f'{endpoint} exhausted retry budget')


def embedding_batch(texts):
    path = ROOT/'embedding-cache'/(digest({'model': MODEL, 'texts': texts})+'.json')
    if path.exists():
        return read(path)
    start = time.perf_counter()
    response = request('embeddings', {'model': MODEL, 'input': texts, 'encoding_format': 'float'})
    rows = sorted(response['data'], key=lambda r: r['index'])
    assert [r['index'] for r in rows] == list(range(len(texts)))
    vectors = [r['embedding'] for r in rows]
    array = np.asarray(vectors)
    assert array.shape == (len(texts), 1024) and np.isfinite(array).all()
    assert (np.linalg.norm(array, axis=1) > 0).all()
    result = {'model': MODEL, 'texts': texts, 'vectors': vectors,
              'elapsed_ms': (time.perf_counter()-start)*1000, 'usage': response.get('usage')}
    save_new(path, result)
    return result


def embed():
    frozen = read(ROOT/'frozen.json')
    texts = list(dict.fromkeys(r['text'] for r in frozen['documents']+frozen['queries']))
    batches = [texts[i:i+32] for i in range(0, len(texts), 32)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(embedding_batch, batch) for batch in batches]
        for i, job in enumerate(as_completed(jobs), 1):
            job.result()
            print(f'Embedding batches {i}/{len(jobs)}', flush=True)


def order(scores):
    return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def fused_order(vector, lexical, weight, rrf_k=60):
    """Weighted RRF with equal ranks for equal scores and no absent BM25 evidence."""
    score = np.zeros(len(vector))
    for values, scale, positive_only in ((vector, weight, False), (lexical, 1-weight, True)):
        previous, rank = None, 0
        for position, i in enumerate(order(values), 1):
            if positive_only and values[i] <= 0:
                continue
            if previous is None or values[i] != previous:
                rank = position
            previous = values[i]
            score[i] += scale/(rrf_k+rank)
    return order(score)


def rerank(query, indices, docs):
    payload = {'model': RERANK_MODEL, 'query': query,
               'documents': [docs[i]['text'] for i in indices], 'top_n': len(indices),
               'return_documents': False}
    path = ROOT/'rerank-cache'/(digest(payload)+'.json')
    if path.exists():
        cached = read(path)
    else:
        started = time.perf_counter()
        response = request('rerank', payload)
        rows = response['results']
        assert sorted(r['index'] for r in rows) == list(range(len(indices)))
        assert all(math.isfinite(r['relevance_score']) for r in rows)
        cached = {'response': response, 'elapsed_ms': (time.perf_counter()-started)*1000,
                  'payload_sha256': digest(payload)}
        save_new(path, cached)
    rows = sorted(cached['response']['results'], key=lambda r: (-r['relevance_score'], r['index']))
    return [indices[r['index']] for r in rows], cached['elapsed_ms']


def metric(ranking, query, n_docs):
    return score_query([str(i) for i in ranking[:3]],
                       {str(i): int(i in query['labels']) for i in range(n_docs)}, 3)


def bootstrap_delta(before, after):
    diffs = np.asarray(after)-np.asarray(before)
    rng = np.random.default_rng(SEED)
    means = diffs[rng.integers(0, len(diffs), (10000, len(diffs)))].mean(axis=1)
    return {'mean_delta': float(diffs.mean()), 'paired_bootstrap_95ci': np.quantile(means,[.025,.975]).tolist(),
            'note': 'query-paired percentile bootstrap; narrow task, public benchmark may have model pretraining contamination'}


def evaluate(split):
    frozen = read(ROOT/'frozen.json')
    target = ROOT/(split+'-result.json')
    if target.exists():
        raise RuntimeError('Result already exists. Do not overwrite frozen evidence.')
    docs, queries, protocol = frozen['documents'], frozen['queries'], frozen['manifest']
    if split == 'test':
        selection = read(ROOT/'dev-result.json')['selection']
    texts = list(dict.fromkeys(r['text'] for r in docs+queries))
    vectors = {}
    for i in range(0, len(texts), 32):
        batch = texts[i:i+32]
        path = ROOT/'embedding-cache'/(digest({'model': MODEL, 'texts': batch})+'.json')
        cached = read(path)  # Missing embedding fails instead of calling hidden/fake backend.
        assert cached['model']==MODEL and cached['texts']==batch
        vectors.update(zip(batch,cached['vectors']))
    matrix = np.asarray([vectors[d['text']] for d in docs], dtype=np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    lexical = BM25Okapi([tokens(d['text']) for d in docs])
    legacy = BM25Okapi([_tokenize(d['text']) for d in docs])
    ids = protocol[split+'_ids']
    base, grid = {}, {str(w): [] for w in protocol['parameters']['vector_weights_dev_grid']}
    for qi in ids:
        q = queries[qi]
        vector = np.asarray(vectors[q['text']])
        vector /= np.linalg.norm(vector)
        similarity = matrix@vector
        bm25 = lexical.get_scores(tokens(q['text']))
        base[qi] = {'vector': order(similarity), 'bm25_zh': [i for i in order(bm25) if bm25[i]>0],
                    'bm25_legacy': [i for i in order(legacy.get_scores(_tokenize(q['text'])))
                                    if legacy.get_scores(_tokenize(q['text']))[i]>0],
                    'similarity': similarity, 'lexical': bm25}
        if split == 'dev':
            for weight in protocol['parameters']['vector_weights_dev_grid']:
                grid[str(weight)].append(metric(fused_order(similarity,bm25,weight),q,len(docs))['ndcg'])
    if split == 'dev':
        means = {w: statistics.mean(values) for w,values in grid.items()}
        # Deterministic tie preference closest to equal weighting.
        weight = sorted(means, key=lambda w: (-means[w], abs(float(w)-.5), float(w)))[0]
        selection = {'rrf_vector_weight': float(weight), 'dev_grid_ndcg': means}
    weight = selection['rrf_vector_weight']
    rows = []
    def one(qi):
        q, b = queries[qi], base[qi]
        fused = fused_order(b['similarity'],b['lexical'],weight)
        re_ranked, elapsed = rerank(q['text'],fused[:50],docs)
        methods = {m:b[m] for m in ('vector','bm25_zh','bm25_legacy')}
        methods.update(weighted_rrf=fused, rrf_reranker=re_ranked)
        results = []
        for name, ranked in methods.items():
            results.append({'query_id':qi,'query':q['text'],'method':name,'top3':ranked[:3],
                'metrics':metric(ranked,q,len(docs)),
                'rerank_remote_ms':elapsed if name=='rrf_reranker' else None,
                'candidate_recall_at_50':len(set(fused[:50])&set(q['labels']))/len(q['labels'])})
        return results
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(one,qi):qi for qi in ids}
        for i, job in enumerate(as_completed(jobs),1):
            rows.extend(job.result())
            print(f'{split} rerank {i}/{len(ids)}', flush=True)
    rows.sort(key=lambda r:(r['query_id'],r['method']))
    summaries = {method:{key:statistics.mean(r['metrics'][key] for r in rows if r['method']==method)
                         for key in ('hit','precision','recall','mrr','ndcg')}
                 for method in ('vector','bm25_zh','bm25_legacy','weighted_rrf','rrf_reranker')}
    if split=='dev':
        eligible = ['vector','bm25_zh','weighted_rrf','rrf_reranker']
        selection['method'] = max(eligible,key=lambda m:summaries[m]['ndcg'])
    comparisons={}
    if split=='test':
        for candidate in ('weighted_rrf','rrf_reranker',selection['method']):
            for baseline in ('vector','bm25_zh'):
                before=[r['metrics']['ndcg'] for r in rows if r['method']==baseline]
                after=[r['metrics']['ndcg'] for r in rows if r['method']==candidate]
                comparisons[candidate+'_vs_'+baseline]=bootstrap_delta(before,after)
    save_new(target, {'split':split,'frozen_sha256':digest(frozen),'protocol':protocol,'selection':selection,
                      'summary':summaries,'rows':rows,'comparisons':comparisons,
                      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({'selection':selection,'summary':summaries,'comparisons':comparisons},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','embed','dev','test'])
    args=parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'embed':
        embed()
    else:
        evaluate(args.action)

