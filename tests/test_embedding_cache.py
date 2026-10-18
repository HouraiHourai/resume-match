from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from resume_match.rag.embedding_cache import EmbeddingCache
from resume_match.rag import embedder


def test_hit_returns_independent_copy_and_lru_eviction():
    cache = EmbeddingCache(2)
    loader = Mock(return_value=[1., 2.])
    first = cache.get_or_compute('a', loader)
    first[0] = 99
    cache.get_or_compute('b', loader)
    assert cache.get_or_compute('a', loader) == [1., 2.]
    cache.get_or_compute('c', loader)  # b is least recently used
    cache.get_or_compute('b', loader)
    assert loader.call_count == 4
    assert cache.info()['entries'] == 2


def test_ttl_expires_from_completion_and_zero_disables_cache():
    now = [0.]
    cache = EmbeddingCache(3, 10, clock=lambda: now[0])
    loader = Mock(return_value=[1.])
    cache.get_or_compute('a', loader)
    now[0] = 9.
    cache.get_or_compute('a', loader)
    now[0] = 10.
    cache.get_or_compute('a', loader)
    assert loader.call_count == 2
    disabled = EmbeddingCache(0)
    disabled.get_or_compute('a', loader)
    disabled.get_or_compute('a', loader)
    assert loader.call_count == 4 and disabled.info()['entries'] == 0


@pytest.mark.parametrize('bad', [[], [0., 0.], [float('nan')], [float('inf')]])
def test_invalid_response_not_cached(bad):
    cache = EmbeddingCache()
    loader = Mock(side_effect=[bad, [1.]])
    with pytest.raises(ValueError):
        cache.get_or_compute('a', loader)
    assert cache.get_or_compute('a', loader) == [1.]
    assert loader.call_count == 2


def test_failure_is_retryable_and_does_not_poison_key():
    cache = EmbeddingCache()
    loader = Mock(side_effect=[TimeoutError('timeout'), [1.]])
    with pytest.raises(TimeoutError):
        cache.get_or_compute('a', loader)
    assert cache.info()['inflight'] == 0
    assert cache.get_or_compute('a', loader) == [1.]


def test_same_key_threads_share_request_but_other_keys_can_progress():
    cache = EmbeddingCache()
    entered, release, joined = Event(), Event(), Event()

    def slow():
        entered.set()
        assert release.wait(3)
        return [1.]

    with ThreadPoolExecutor(max_workers=3) as pool:
        one = pool.submit(cache.get_or_compute, 'a', slow)
        assert entered.wait(3)
        # Observe Future.result entry to deterministically establish a waiter.
        with cache._lock:
            pending = next(iter(cache._inflight.values()))
            original = pending.result

            def waiting(*args, **kwargs):
                joined.set()
                return original(*args, **kwargs)

            pending.result = waiting
        duplicate_loader = Mock(side_effect=AssertionError('duplicate API call'))
        two = pool.submit(cache.get_or_compute, 'a', duplicate_loader)
        try:
            assert joined.wait(3)
            assert pool.submit(cache.get_or_compute, 'b', lambda: [2.]).result(3) == [2.]
        finally:
            release.set()
        assert one.result(3) == two.result(3) == [1.]
        assert not duplicate_loader.called
        assert cache.info()['waits'] == 1


def test_clear_during_inflight_does_not_reinsert_stale_entry():
    cache = EmbeddingCache()
    entered, release = Event(), Event()

    def slow():
        entered.set()
        assert release.wait(3)
        return [1.]

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(cache.get_or_compute, 'a', slow)
        assert entered.wait(3)
        cache.clear()
        release.set()
        assert result.result(3) == [1.]
    assert cache.info()['entries'] == 0
    assert cache.get_or_compute('a', lambda: [2.]) == [2.]


def test_client_namespace_and_input_changes_invalidate(monkeypatch):
    model = SimpleNamespace(model='m1', openai_api_base='https://one', dimensions=2,
                            openai_api_key='key1', check_embedding_ctx_length=False,
                            model_kwargs={}, embed_query=Mock(return_value=[1., 2.]))
    monkeypatch.setattr(embedder, '_model', model)
    monkeypatch.setattr(embedder, '_cache', EmbeddingCache())
    embedder.embed_text('Python')
    embedder.embed_text('Python')
    assert model.embed_query.call_count == 1
    embedder.embed_text('python')  # exact input; no potentially lossy normalization
    for field, value in [('model', 'm2'), ('openai_api_base', 'https://two'),
                         ('dimensions', 3), ('openai_api_key', 'key2'),
                         ('model_kwargs', {'revision': 'v2'})]:
        setattr(model, field, value)
        embedder.embed_text('Python')
    assert model.embed_query.call_count == 7
