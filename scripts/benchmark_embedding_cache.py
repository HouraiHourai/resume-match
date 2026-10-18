"""Real A/B retrieval latency with fixed repeat rates; isolated historical indexes.

Invokes production find_similar + embed_text, toggling only the process-local
EmbeddingCache. Real HTTP measured by transport, not estimated from hit ratios.
Historical vectors reuse previously saved BGE responses; query misses always
make fresh API calls. No private/default history store is opened.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import statistics
import tempfile
import time
from unittest.mock import patch

import httpx
from langchain_openai import OpenAIEmbeddings

from resume_match.evaluation.dataset import dataset_digest, load_dataset
from resume_match.evaluation.metrics import percentile
from resume_match.history.models import HistoryEntry
from resume_match.rag import embedder, retriever, coarse_ranker
from resume_match.rag.bm25_store import BM25Store
from resume_match.rag.embedding_cache import EmbeddingCache, valid_vector
from resume_match.rag.vector_store import VectorStore


def input_text(record):
    return embedder.build_embedding_text_from_dict(record.jd.model_dump())


def make_workload(query_count, repeats, seed=20260919):
    # The repetition pattern and order are frozen before requests are executed.
    if query_count < 1 or repeats < 1:
        raise ValueError('positive query_count and repeats required')
    indices = list(range(query_count)) * repeats
    random.Random(seed).shuffle(indices)
    return indices


def history_vectors(documents):
    saved = {}
    old = Path('evaluation/local/jd-pilot/bge-responses.json')
    if old.exists():
        for text, row in json.loads(old.read_text(encoding='utf-8')).items():
            if row['model'] == 'Pro/BAAI/bge-m3':
                saved[text] = row['vector']
    for directory in [Path('evaluation/local/jd-pilot-v2/embedding-cache')]:
        for path in directory.glob('*.json'):
            row = json.loads(path.read_text(encoding='utf-8'))
            if row['model'] == 'Pro/BAAI/bge-m3':
                saved.update(zip(row['texts'], row['vectors']))
    missing = sum(input_text(d) not in saved for d in documents)
    if missing:
        raise ValueError(
            f"Missing saved BGE vectors for {missing} records. Prepare the local "
            "embedding inputs described in evaluation/README.md before running this benchmark."
        )
    # Only expose history embeddings to index initialization. Query cache starts empty.
    result = {input_text(d): list(valid_vector(saved[input_text(d)])) for d in documents}
    if any(len(vector) != 1024 for vector in result.values()):
        raise ValueError('Historical embeddings must be real 1024-dimensional BGE responses')
    return result


class MeteredTransport(httpx.BaseTransport):
    def __init__(self, output):
        self.transport = httpx.HTTPTransport(retries=0)
        self.output = output
        self.calls = []
        self.context = {'scenario':'setup', 'arm':'setup', 'request_index':-1}

    def handle_request(self, request):
        body = json.loads(request.content)
        row = {**self.context, 'model':body.get('model'), 'status':None,
               'input_sha256':hashlib.sha256(json.dumps(body.get('input'), ensure_ascii=False).encode()).hexdigest()}
        self.calls.append(row)
        started = time.perf_counter()
        try:
            response = self._send(request)
            row['status'] = response.status_code
            response.read()
            if response.status_code == 200:
                row['usage'] = response.json().get('usage')
            return response
        except Exception as exc:
            row['error_type'] = type(exc).__name__
            raise
        finally:
            row['elapsed_ms'] = (time.perf_counter()-started)*1000
            with (self.output/'http_calls.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row, ensure_ascii=False)+'\n')

    def close(self):
        self.transport.close()

    def _send(self, request):
        return self.transport.handle_request(request)


class ReplayTransport(MeteredTransport):
    """Network-free correctness check using existing API response vectors only."""
    def __init__(self, output, vectors):
        super().__init__(output)
        self.vectors = vectors

    def _send(self, request):
        payload = json.loads(request.content)
        texts = payload['input']
        if isinstance(texts, str):
            texts = [texts]
        return httpx.Response(200, json={'object':'list','model':payload['model'],
            'data':[{'object':'embedding','index':i,'embedding':self.vectors[text]}
                    for i,text in enumerate(texts)]})


def summarize(rows):
    times = [r['retrieval_ms'] for r in rows]
    usage = [r['embedding_tokens'] for r in rows if r['embedding_tokens'] is not None]
    return {'requests':len(rows), 'http_calls':sum(r['http_calls'] for r in rows),
            'cache_hits':sum(r['cache_hit'] for r in rows),
            'reported_embedding_tokens':sum(usage),
            'requests_with_missing_usage':sum(r['embedding_tokens'] is None for r in rows),
            'mean_ms':statistics.mean(times), 'p50_ms':percentile(times,.5), 'p95_ms':percentile(times,.95)}


def run(dataset_path, output, *, offline=False):
    if not dataset_path.is_file():
        raise FileNotFoundError(
            "Benchmark dataset is not included in the public source snapshot. "
            "Supply a prepared dataset with --dataset; see evaluation/README.md."
        )
    data = load_dataset(dataset_path)
    if len(data.queries) != 10:
        raise ValueError('Protocol requires exactly 10 distinct queries')
    stored = history_vectors(data.documents)
    output.mkdir(parents=True, exist_ok=False)
    sequence = {name:make_workload(10, repeats) for name,repeats in [('unique',1),('repeat80',5)]}
    protocol = {'backend':'offline_response_replay' if offline else 'live_siliconflow',
        'dataset_sha256':dataset_digest(data), 'documents':len(data.documents),
        'queries':10, 'workloads':sequence, 'repeated_request_fractions':{'unique':0,'repeat80':.8},
        'cache':{'max_entries':256,'ttl_seconds':3600,'initial':'empty for each scenario'},
        'concurrency':1,'seed':20260919,'k':3,'http_retries':0,'timeout_seconds':20,
        'arm_order':'alternates by request index; each pair runs the same JD, shared warmed HTTP connection',
        'metric_scope':'find_similar wall time INCLUDING remote query embedding, stores count/query, dedup, RRF and fine_rank; excludes JD extraction, index build and LLM generation',
        'history_initialization':'reuse saved real BGE history vectors only; never prepopulate query cache',
        'network_warmup':'one real embedding request before scenarios, outside measurement; no cache populated',
        'early_stop':'any API/recall error aborts rather than silently benchmarking degraded results',
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in list(Path('src/resume_match/rag').glob('*.py'))+[Path(__file__)]}}
    (output/'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    transport = ReplayTransport(output,history_vectors(data.documents+data.queries)) if offline else MeteredTransport(output)
    with httpx.Client(transport=transport) as http:
        source_model = None if offline else embedder._get_model()
        model = OpenAIEmbeddings(model='Pro/BAAI/bge-m3',
            api_key='offline-replay-not-real' if offline else source_model.openai_api_key,
            base_url='https://offline.invalid/v1' if offline else source_model.openai_api_base,
            check_embedding_ctx_length=False,
            http_client=http, max_retries=0, request_timeout=20)
        with tempfile.TemporaryDirectory(prefix='resume-match-cache-bench-') as directory:
            root = Path(directory)
            vector, bm25 = VectorStore(root/'chroma'), BM25Store(root/'bm25.pkl')
            try:
                entries = []
                for i,d in enumerate(data.documents,1):
                    entry = HistoryEntry(id=i,created_at='cache-benchmark',jd_text=d.text,
                        jd_summary=d.jd.role_title,resume_filename='none',full_result={'jd':d.jd.model_dump()})
                    entries.append(entry)
                    vector.add(str(i),input_text(d),{'history_id':i,'jd_summary':d.jd.role_title,
                        'role_title':d.jd.role_title,'company':d.jd.company or '',
                        'hard_skills':','.join(d.jd.hard_skills),'domain':d.jd.domain or '',
                        'seniority':d.jd.seniority,'created_at':'cache-benchmark'},stored[input_text(d)])
                bm25.replace_all(entries)
                vector.query(stored[input_text(data.documents[0])],3)
                bm25.query('Python',3)
                transport.context = {'scenario':'warmup','arm':'outside_measurement','request_index':-1}
                model.embed_query(input_text(data.queries[0]))
                with ExitStack() as stack:
                    stack.enter_context(patch.object(embedder,'_model',model))
                    for module in (retriever,coarse_ranker):
                        stack.enter_context(patch.object(module,'get_vector_store',lambda:vector))
                        stack.enter_context(patch.object(module,'get_bm25_store',lambda:bm25))
                    production_coarse = coarse_ranker.coarse_rank
                    def strict_coarse(*args, **kwargs):
                        return production_coarse(*args, **kwargs, strict=True)
                    stack.enter_context(patch.object(retriever,'coarse_rank',strict_coarse))
                    for scenario, workload in sequence.items():
                        caches = {'disabled':EmbeddingCache(0), 'enabled':EmbeddingCache(256,3600)}
                        for position, query_index in enumerate(workload):
                            query = data.queries[query_index]
                            pair = {}
                            for arm in (('disabled','enabled') if position%2==0 else ('enabled','disabled')):
                                transport.context={'scenario':scenario,'arm':arm,'request_index':position}
                                start_calls = len(transport.calls)
                                before = caches[arm].info()
                                with patch.object(embedder,'_cache',caches[arm]):
                                    started = time.perf_counter()
                                    result = retriever.find_similar(query.jd,top_k=3)
                                    elapsed = (time.perf_counter()-started)*1000
                                actual = transport.calls[start_calls:]
                                after = caches[arm].info()
                                record = {'scenario':scenario,'arm':arm,'request_index':position,
                                    'query_id':query.id,'retrieval_ms':elapsed,'http_calls':len(actual),
                                    'cache_hit':after['hits']>before['hits'],
                                    'embedding_tokens':sum(c['usage'].get('total_tokens',0) for c in actual)
                                        if all(c.get('usage') is not None for c in actual) else None,
                                    'result':[r.model_dump() for r in result]}
                                rows.append(record)
                                pair[arm]=record
                                with (output/'requests.jsonl').open('a',encoding='utf-8') as f:
                                    f.write(json.dumps(record,ensure_ascii=False)+'\n')
                            # Full structured result equality, not just number of hits.
                            if pair['enabled']['result'] != pair['disabled']['result']:
                                raise RuntimeError(f'Retrieval changed for {scenario}/{query.id}; inspect evidence')
                            if (position+1)%10==0:
                                print(f'{scenario}: {position+1}/{len(workload)} paired requests; results identical',flush=True)
            finally:
                from chromadb.api.client import SharedSystemClient
                for identifier,system in list(SharedSystemClient._identifier_to_system.items()):
                    if system.settings.persist_directory == str(root/'chroma'):
                        system.stop()
                        SharedSystemClient._identifier_to_system.pop(identifier,None)
    summaries = {scenario:{arm:summarize([r for r in rows if r['scenario']==scenario and r['arm']==arm])
                           for arm in ('disabled','enabled')} for scenario in sequence}
    hits = [r for r in rows if r['scenario']=='repeat80' and r['arm']=='enabled' and r['cache_hit']]
    misses = [r for r in rows if r['scenario']=='repeat80' and r['arm']=='enabled' and not r['cache_hit']]
    for scenario,expected_n in [('unique',10),('repeat80',50)]:
        assert summaries[scenario]['disabled']['http_calls']==expected_n
        assert summaries[scenario]['enabled']['http_calls']==10
    assert len(hits)==40 and len(misses)==10
    result={'protocol':protocol,'summary':summaries,'repeat80_hits':summarize(hits),
            'repeat80_misses':summarize(misses),'all_60_paired_results_identical':True,
            'http_calls_including_warmup':len(transport.calls),
            'reported_tokens_including_warmup':sum((c.get('usage') or {}).get('total_tokens',0) for c in transport.calls)}
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    write_report(result,output)
    print(json.dumps(result['summary'],ensure_ascii=False,indent=2),flush=True)


def write_report(result,output):
    if result['protocol']['backend']=='offline_response_replay':
        lines=['# Embedding缓存离线验证（不是API性能实测）','',
               '使用以前保存的真实向量作为HTTP响应回放，不发送任何网络请求、不消耗API Token。',
               '检查的是实际SDK序列化、生产find_similar链路、缓存调用次数和返回结果一致性。', '',
               '| 固定负载 | 关闭缓存：回放次数 | 开启缓存：回放次数 | 命中次数 |',
               '|---|---:|---:|---:|']
        for name,arms in result['summary'].items():
            lines.append(f"| {name} | {arms['disabled']['http_calls']} | {arms['enabled']['http_calls']} | {arms['enabled']['cache_hits']} |")
        lines+=['','60组配对请求的完整检索结果全部一致。',
                '上述是离线回放调用次数。真实API的P50/P95、Token与账单收益尚未测量，回放耗时仅用于功能验证。',
                'result.json中的本地执行耗时仅供调试；真实测试需去掉--offline，使用全新的output目录。']
        (output/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return
    a,b=result['summary']['repeat80']['disabled'],result['summary']['repeat80']['enabled']
    p50=(a['p50_ms']-b['p50_ms'])/a['p50_ms']*100
    p95=(a['p95_ms']-b['p95_ms'])/a['p95_ms']*100
    lines=['# Embedding缓存真实API对照测试','',
        f"数据：{result['protocol']['queries']}条查询、{result['protocol']['documents']}条历史JD。测量实际find_similar，包括查询Embedding API耗时与检索排序，不包括JD提取、索引构建和LLM建议生成。",
        '测试负载是人为固定的重复率，不能声称是用户真实流量分布。未根据结果修改请求顺序或重复率。','',
        '| 场景 | 缓存 | 请求数 | 实际HTTP调用 | Embedding Token | P50 ms | P95 ms |',
        '|---|---|---:|---:|---:|---:|---:|']
    for scenario,arms in result['summary'].items():
        for arm,s in arms.items():
            lines.append(f"| {scenario} | {arm} | {s['requests']} | {s['http_calls']} | {s['reported_embedding_tokens']} | {s['p50_ms']:.2f} | {s['p95_ms']:.2f} |")
    lines+=['','## 80%重复请求场景','',
        f"10个不同JD各请求5次，总计50次，两个方案初始缓存均为空。HTTP调用由{a['http_calls']}次减为{b['http_calls']}次，减少80%；40次命中、10次首次未命中。",
        f"全负载P50由{a['p50_ms']:.2f}ms到{b['p50_ms']:.2f}ms，降低{p50:.2f}%；P95由{a['p95_ms']:.2f}ms到{b['p95_ms']:.2f}ms，变化按降低计为{p95:+.2f}%。",
        'P95包含未命中请求，仍会受网络影响，不只统计缓存命中。不要把命中子集的延迟代替整个负载P95。',
        f"命中40次的P95={result['repeat80_hits']['p95_ms']:.2f}ms；未命中10次的P95={result['repeat80_misses']['p95_ms']:.2f}ms。",
        '', '## 校验和限制','',
        '- 60组配对请求（10组无重复、50组80%重复），两侧返回的完整结构化检索结果全部一致，不只核对Top-3数量。',
        '- 缓存关闭仍调用相同生产embed_text/find_similar，只禁用缓存容量；候选库、索引、阈值、精排和连接相同。SDK重试关闭，两侧相同，API或召回错误立即终止。',
        '- 交替先后执行两侧，使用同一预热HTTP连接；测量串行请求，非并发压力测试。每次请求单独测量，没有挑最快的一次。',
        '- 无重复场景两侧均10次真实HTTP调用；缓存不降低首次生成费用。',
        '- 本地进程内LRU容量256，TTL一小时；重启/过期/淘汰/配置或输入变化会重新调用。多worker不共享缓存，不保证跨进程消重。',
        '- 输入键为实际传入embedding的结构化文本哈希，不是原始JD哈希。LLM抽取结果有变化（含技能顺序变化）可能不命中。',
        '- 缓存仅复用向量，不缓存检索结果；历史新增/删除仍重新查询当前索引。缓存不会提升匹配准确率。',
        '- 模型同名但服务端版本更换不能自动识别，TTL限制陈旧时间；需要主动clear_embedding_cache或重启。',
        '- Token是服务响应usage的Embedding Token，不是DeepSeek/通义Token，也不直接等于账单节省比例。',
        '', '时延数字仅限当前机器和固定测试负载；报告时需保留重复率、样本量、P50/P95名称和“包含查询Embedding的检索耗时”范围。',
        '', '逐HTTP请求及usage： http_calls.jsonl；逐检索耗时/命中/结果：requests.jsonl；冻结协议与源码哈希：protocol.json。']
    (output/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=Path('evaluation/local/jd-pilot-v2/dataset.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--offline',action='store_true',help='Replay stored vectors; no HTTP/cost/latency claims')
    args=parser.parse_args()
    run(args.dataset,args.output,offline=args.offline)

