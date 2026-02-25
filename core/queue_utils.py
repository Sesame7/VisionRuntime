"""Shared queue helpers for non-blocking drain/cleanup paths."""

from __future__ import annotations

import queue
from collections.abc import Callable
from typing import Any


def drain_queue_nowait(
    q: queue.Queue,
    *,
    on_item: Callable[[Any], None] | None = None,
) -> int:
    """Drain all currently queued items without blocking.

    Calls `task_done()` for each popped item (errors suppressed), which keeps queue
    counters consistent during shutdown/reset cleanup flows.
    """
    drained = 0
    while True:
        try:
            item = q.get_nowait()
        except queue.Empty:
            return drained
        try:
            if on_item is not None:
                on_item(item)
        finally:
            q.task_done()
        drained += 1


__all__ = ["drain_queue_nowait"]
