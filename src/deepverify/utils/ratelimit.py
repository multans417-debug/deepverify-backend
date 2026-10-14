"""Tiny in-process sliding-window rate limiter (per worker process; see docs for limits)."""
from __future__ import annotations

import time
from collections import deque
from typing import Callable


class SlidingWindowLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window, self._clock = limit, window_seconds, clock
        self._hits: dict[str, deque] = {}
        self._calls = 0

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = self._clock()
        self._calls += 1
        if self._calls % 500 == 0:
            self._prune(now)
        q = self._hits.setdefault(key, deque())
        while q and now - q[0] >= self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True

    def _prune(self, now: float) -> None:
        for k in [k for k, q in self._hits.items() if not q or now - q[-1] >= self.window]:
            del self._hits[k]
