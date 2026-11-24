"""One GPU, with Bot Builder requests served ahead of bulk ingestion."""

import os
import threading
import time
from contextlib import contextmanager


class QueueFull(Exception):
    def __init__(self, retry_after: int):
        super().__init__("Embedding queue is full.")
        self.retry_after = retry_after


class WaitTimeout(Exception):
    def __init__(self, retry_after: int):
        super().__init__("Timed out waiting for the GPU.")
        self.retry_after = retry_after


def lane_for(source: str | None) -> str:
    return "bot_builder" if source == "bot_builder" else "kbm"


class PriorityGate:
    def __init__(self, max_waiting: dict[str, int], wait_timeout_s: dict[str, float]):
        self._cond = threading.Condition()
        self._busy = False
        self._inflight_source: str | None = None
        self._waiting = {"bot_builder": 0, "kbm": 0}
        self._max_waiting = max_waiting
        self._timeout = wait_timeout_s

    @classmethod
    def from_env(cls) -> "PriorityGate":
        return cls(
            max_waiting={
                "bot_builder": int(os.environ.get("QWEN_MAX_WAITING_BOT_BUILDER", "16")),
                "kbm": int(os.environ.get("QWEN_MAX_WAITING_KBM", "8")),
            },
            wait_timeout_s={
                "bot_builder": float(os.environ.get("QWEN_WAIT_TIMEOUT_BOT_BUILDER_S", "30")),
                "kbm": float(os.environ.get("QWEN_WAIT_TIMEOUT_KBM_S", "120")),
            },
        )

    def snapshot(self) -> dict:
        with self._cond:
            return {
                "busy": self._busy,
                "inflight_source": self._inflight_source,
                "waiting": dict(self._waiting),
            }

    @contextmanager
    def acquire(self, source: str | None):
        lane = lane_for(source)
        retry_after = 2 if lane == "bot_builder" else 5
        started = time.perf_counter()
        with self._cond:
            if self._waiting[lane] >= self._max_waiting[lane]:
                raise QueueFull(retry_after)
            self._waiting[lane] += 1
            deadline = time.monotonic() + self._timeout[lane]
            try:
                while self._busy or (lane == "kbm" and self._waiting["bot_builder"] > 0):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise WaitTimeout(retry_after)
                    self._cond.wait(timeout=remaining)
                self._waiting[lane] -= 1
                self._busy = True
                self._inflight_source = lane
            except BaseException:
                self._waiting[lane] -= 1
                self._cond.notify_all()
                raise
        queue_ms = (time.perf_counter() - started) * 1000.0
        try:
            yield queue_ms
        finally:
            with self._cond:
                self._busy = False
                self._inflight_source = None
                self._cond.notify_all()
