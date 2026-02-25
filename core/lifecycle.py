"""Lightweight lifecycle helpers and shared async-loop primitives."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Coroutine
from concurrent.futures import TimeoutError
from typing import Any, Callable, TypeVar

# Keep logger name stable with previous shutdown_loop/run_async logs.
L = logging.getLogger("vision_runtime.runtime")


T = TypeVar("T")


class LoopRunner:
    """Owns a shared background asyncio loop and provides sync bridge helpers."""

    def __init__(self, *, logger: logging.Logger | None = None):
        self._logger = logger or L
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._lock = threading.Lock()
        self._loop_ready: threading.Event | None = None
        self._loop_thread_ident: int | None = None

    def _ensure_loop(self):
        with self._lock:
            if self._stopped:
                raise RuntimeError("Async loop already stopped")
            if self._loop and self._thread and self._thread.is_alive():
                return self._loop
            self._loop = asyncio.new_event_loop()
            self._loop_ready = threading.Event()
            self._loop_thread_ident = None

            def _runner():
                loop = self._loop
                if loop is None:
                    return
                asyncio.set_event_loop(loop)
                self._loop_thread_ident = threading.get_ident()
                if self._loop_ready:
                    self._loop_ready.set()
                loop.run_forever()

            self._thread = threading.Thread(target=_runner, daemon=True)
            self._thread.start()
            if self._loop_ready:
                self._loop_ready.wait(timeout=0.5)
            return self._loop

    def run_async(self, coro: Coroutine[Any, Any, T], timeout: float | None = 0.5) -> T:
        """Submit coroutine to the shared loop from a non-loop thread and wait."""
        loop = self._ensure_loop()
        if (
            self._loop_thread_ident is not None
            and threading.get_ident() == self._loop_thread_ident
        ):
            raise RuntimeError(
                "run_async must not be called from the loop thread; await directly"
            )
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return fut.result(timeout=timeout)
        except TimeoutError:
            fut.cancel()
            self._logger.warning("run_async timeout after %.2fs", timeout or 0)
            raise

    def spawn_background_task(self, coro: Coroutine[Any, Any, Any]):
        """Fire-and-forget task on the shared loop; returns the task/future handle."""
        loop = self._ensure_loop()
        if (
            self._loop_thread_ident is not None
            and threading.get_ident() == self._loop_thread_ident
        ):
            return loop.create_task(coro)
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def shutdown_loop(self, timeout: float = 1.0):
        """Cancel pending tasks and stop the shared loop."""
        if (
            self._loop_thread_ident is not None
            and threading.get_ident() == self._loop_thread_ident
        ):
            raise RuntimeError("shutdown_loop must not be called from the loop thread")
        with self._lock:
            loop = self._loop
            thread = self._thread
            if not loop or not thread:
                self._stopped = True
                return
            if loop.is_closed():
                self._stopped = True
                return
            self._stopped = True

        async def _shutdown():
            current = asyncio.current_task()
            tasks = [
                t for t in asyncio.all_tasks() if t is not current and not t.done()
            ]
            if tasks:
                names = [t.get_name() or repr(t) for t in tasks[:10]]
                suffix = ""
                if len(tasks) > 10:
                    suffix = f" (+{len(tasks) - 10} more)"
                self._logger.info(
                    "shutdown_loop pending_tasks=%d names=%s%s",
                    len(tasks),
                    ", ".join(names),
                    suffix,
                )
            else:
                self._logger.debug("shutdown_loop pending_tasks=0")
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await loop.shutdown_asyncgens()

        try:
            fut = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
            fut.result(timeout=timeout)
            loop.call_soon_threadsafe(loop.stop)
        except TimeoutError:
            fut.cancel()
            loop.call_soon_threadsafe(loop.stop)
            raise
        finally:
            if thread.is_alive():
                thread.join(timeout=timeout)
            if not loop.is_closed():
                loop.close()
            self._loop = None
            self._thread = None
            self._loop_ready = None
            self._loop_thread_ident = None


class AsyncTaskOwner:
    """Own background tasks locally unless an external registrar takes ownership."""

    def __init__(
        self,
        *,
        loop_runner: LoopRunner,
        task_reg: Callable[[Any], Any] | None = None,
        logger: logging.Logger | None = None,
        owner_name: str = "async_service",
    ):
        self._task_reg = task_reg
        self._logger = logger
        self._owner_name = owner_name
        self._loop_runner = loop_runner
        self._local_tasks: list[Any] = []

    def spawn(self, coro: Coroutine[Any, Any, Any]):
        task = self._loop_runner.spawn_background_task(coro)
        return self.register(task)

    def register(self, task: Any):
        if task is None:
            return None
        if self._task_reg:
            adopted = self._task_reg(task)
            if adopted:
                return task
        self._local_tasks.append(task)
        return task

    def cancel_local_tasks(self):
        for task in list(self._local_tasks):
            task.cancel()

    def clear_local_tasks(self):
        self._local_tasks.clear()

    def cancel_and_clear_local_tasks(self):
        self.cancel_local_tasks()
        self.clear_local_tasks()

    @property
    def loop_runner(self) -> LoopRunner:
        return self._loop_runner


def run_async_cleanup(
    coro: Coroutine[Any, Any, Any],
    *,
    loop_runner: LoopRunner,
    timeout: float = 0.5,
):
    """Run async cleanup from sync code with a bounded wait."""
    loop_runner.run_async(coro, timeout=timeout)


__all__ = [
    "LoopRunner",
    "AsyncTaskOwner",
    "run_async_cleanup",
]
