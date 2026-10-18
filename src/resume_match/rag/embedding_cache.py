"""Bounded process-local LRU/TTL cache with same-key concurrent request sharing."""
from collections import OrderedDict
from concurrent.futures import Future
import math
from threading import RLock
import time


def valid_vector(values) -> tuple[float, ...]:
    vector = tuple(float(v) for v in values)
    if not vector or not all(math.isfinite(v) for v in vector) or not any(vector):
        raise ValueError('Embedding must be finite, non-empty and non-zero')
    return vector


class EmbeddingCache:
    def __init__(self, max_entries=256, ttl_seconds=3600, *, clock=time.monotonic):
        if max_entries < 0 or not math.isfinite(ttl_seconds) or ttl_seconds < 0:
            raise ValueError('Cache capacity and TTL must be finite and non-negative')
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._lock = RLock()
        self._entries = OrderedDict()
        self._inflight = {}
        self._generation = 0
        self._hits = self._misses = self._waits = 0

    def get_or_compute(self, key: str, loader) -> list[float]:
        """Return a copy. Failures never become cached results; no lock during I/O."""
        with self._lock:
            enabled = self.max_entries > 0 and self.ttl_seconds > 0
            if not enabled:
                self._misses += 1
            else:
                now = self._clock()
                # Capacity is small; also reclaim expired entries on unrelated keys.
                for stale in [k for k, (expiry, _) in self._entries.items() if expiry <= now]:
                    self._entries.pop(stale)
                item = self._entries.get(key)
                if item is not None:
                    self._entries.move_to_end(key)
                    self._hits += 1
                    return list(item[1])
                generation = self._generation
                pending_key = (generation, key)
                pending = self._inflight.get(pending_key)
                owner = pending is None
                if owner:
                    pending = Future()
                    self._inflight[pending_key] = pending
                    self._misses += 1
                else:
                    self._waits += 1
        if not enabled:
            return list(valid_vector(loader()))
        if not owner:
            return list(pending.result())
        try:
            vector = valid_vector(loader())
            with self._lock:
                # Clearing while a request runs must not repopulate the new cache.
                if generation == self._generation:
                    self._entries[key] = (self._clock() + self.ttl_seconds, vector)
                    self._entries.move_to_end(key)
                    while len(self._entries) > self.max_entries:
                        self._entries.popitem(last=False)
                self._inflight.pop(pending_key, None)
                pending.set_result(vector)
            return list(vector)
        except BaseException as exc:
            with self._lock:
                self._inflight.pop(pending_key, None)
                pending.set_exception(exc)
            raise

    def clear(self):
        with self._lock:
            self._generation += 1
            self._entries.clear()
            self._hits = self._misses = self._waits = 0

    def info(self):
        with self._lock:
            return {'hits': self._hits, 'misses': self._misses, 'waits': self._waits,
                    'entries': len(self._entries), 'max_entries': self.max_entries,
                    'ttl_seconds': self.ttl_seconds, 'inflight': len(self._inflight)}
