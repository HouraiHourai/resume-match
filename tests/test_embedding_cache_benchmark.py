import importlib.util
from pathlib import Path
from types import SimpleNamespace

import httpx
from langchain_openai import OpenAIEmbeddings
from resume_match.rag import embedder, coarse_ranker, retriever
from resume_match.rag.embedding_cache import EmbeddingCache
from resume_match.state import JDRequirements

spec = importlib.util.spec_from_file_location('cache_benchmark',
    Path(__file__).resolve().parents[1]/'scripts/benchmark_embedding_cache.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_workload_repetition_is_fixed_not_selected_after_run():
    assert len(set(benchmark.make_workload(10, 1))) == 10
    repeated = benchmark.make_workload(10, 5)
    assert len(repeated) == 50
    assert all(repeated.count(i) == 5 for i in range(10))
    assert repeated == benchmark.make_workload(10, 5)


def test_sdk_replay_cache_changes_http_count_not_vectors(tmp_path, monkeypatch):
    transport = benchmark.ReplayTransport(tmp_path, {'Python': [1., 2.]})
    with httpx.Client(transport=transport) as http:
        model = OpenAIEmbeddings(model='Pro/BAAI/bge-m3', api_key='test',
            http_client=http, base_url='https://offline.invalid/v1',
            check_embedding_ctx_length=False, max_retries=0)
        monkeypatch.setattr(embedder, '_model', model)
        monkeypatch.setattr(embedder, '_cache', EmbeddingCache(0))
        assert embedder.embed_text('Python') == embedder.embed_text('Python') == [1., 2.]
        assert len(transport.calls) == 2
        monkeypatch.setattr(embedder, '_cache', EmbeddingCache(3))
        assert embedder.embed_text('Python') == embedder.embed_text('Python') == [1., 2.]
        assert len(transport.calls) == 3


def test_cached_embedding_does_not_cache_retrieval_results(monkeypatch):
    from unittest.mock import Mock
    model = SimpleNamespace(model='test', embed_query=Mock(return_value=[1., 0.]))
    monkeypatch.setattr(embedder, '_model', model)
    monkeypatch.setattr(embedder, '_cache', EmbeddingCache())
    query = JDRequirements(role_title='Agent',hard_skills=['Python'],seniority='intern',key_responsibilities=[])

    class Store:
        current_id = '1'
        def count(self):
            return 1
        def query(self, embedding, top_k):
            return {'ids':[[self.current_id]], 'distances':[[0.]],
                    'metadatas':[[{'hard_skills':'Python','seniority':'intern','role_title':'Agent'}]]}

    vector = Store()
    for module in (retriever, coarse_ranker):
        monkeypatch.setattr(module, 'get_vector_store', lambda: vector)
        monkeypatch.setattr(module, 'get_bm25_store', lambda: SimpleNamespace(count=lambda:0))
    monkeypatch.setattr(coarse_ranker,'embed_text',embedder.embed_text)
    assert retriever.find_similar(query)[0].history_id == 1
    vector.current_id = '2'  # Index changed; same query embedding may still be reused.
    assert retriever.find_similar(query)[0].history_id == 2
    assert model.embed_query.call_count == 1
