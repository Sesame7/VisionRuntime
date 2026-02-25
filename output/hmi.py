# -- coding: utf-8 --
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from aiohttp import web
import os

from core.lifecycle import AsyncTaskOwner, LoopRunner, run_async_cleanup
from datetime import datetime, timezone
from core.contracts import OutputRecord

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from core.runtime import ResultReadApi


class AppContextLike(Protocol):
    @property
    def trigger_gateway(self) -> Any: ...

    @property
    def results(self) -> "ResultReadApi": ...


L = logging.getLogger("vision_runtime.output.hmi")


class _ApiServer:
    def __init__(
        self,
        host: str,
        port: int,
        context: AppContextLike,
        index_path: str,
        task_reg=None,
        *,
        loop_runner: LoopRunner,
    ):
        self.host = host
        self.port = port
        self.context = context
        self.index_path = index_path
        self.app = web.Application()
        self._setup_routes()
        self._runner = None
        self._started = False
        self._tasks = AsyncTaskOwner(
            task_reg=task_reg,
            logger=L,
            owner_name="hmi_api_server",
            loop_runner=loop_runner,
        )

    def _setup_routes(self):
        app = self.app
        ctx = self.context
        index_path = self.index_path
        store: ResultReadApi = ctx.results

        def _serialize_record(rec):
            def _dt(val):
                if isinstance(val, datetime):
                    ref = (
                        val.astimezone(timezone.utc)
                        if val.tzinfo
                        else val.replace(tzinfo=timezone.utc)
                    )
                    return ref.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
                return val

            data = rec.__dict__.copy()
            for key in ("triggered_at", "captured_at", "detected_at"):
                if key in data:
                    data[key] = _dt(data[key])
            return data

        async def index(_request):
            return web.FileResponse(index_path)

        async def status(_request):
            records = [_serialize_record(r) for r in store.latest_records]
            payload = {
                "records": records,
                "stats": store.stats(),
                "max_records": store.max_records,
                "heartbeat_seq": store.heartbeat_seq(),
            }
            return web.json_response(payload)

        async def latest_preview(_request):
            item = ctx.results.latest_overlay()
            if item is not None:
                data, ctype = item
                return web.Response(body=data, content_type=ctype)

            # No binary preview available: for ERROR/TIMEOUT, return a placeholder preview
            # with a large abbreviation so HMI still shows a clear state.
            last_items = store.latest_records
            if not last_items:
                return web.Response(status=404)
            rec = last_items[0]
            result = str(rec.result or "").upper()
            abbr = "ERR" if result == "ERROR" else ("TO" if result == "TIMEOUT" else "")
            if not abbr:
                return web.Response(status=404)
            svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 720">
  <rect width="100%" height="100%" fill="#000"/>
  <text x="50%" y="50%" fill="#ff3b30" font-size="220" font-family="Arial, sans-serif"
        font-weight="700" text-anchor="middle" dominant-baseline="middle">{abbr}</text>
</svg>"""
            return web.Response(body=svg.encode("utf-8"), content_type="image/svg+xml")

        async def trigger(request):
            remote_ip = request.remote
            ok = ctx.trigger_gateway.report_raw_trigger("WEB", remote_ip=remote_ip)
            return web.json_response({"accepted": ok})

        app.router.add_get("/", index)
        app.router.add_get("/index.html", index)
        app.router.add_static("/static/", os.path.dirname(index_path))
        app.router.add_get("/status", status)
        app.router.add_get("/preview/latest", latest_preview)
        app.router.add_post("/trigger", trigger)

    def start(self):
        if self._started:
            return
        L.info("HMI web service running @ http://%s:%d", self.host, self.port)
        self._tasks.spawn(self._serve())
        self._started = True

    async def _serve(self):
        self._runner = web.AppRunner(self.app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()

    def stop(self):
        self._tasks.cancel_and_clear_local_tasks()

        async def _cleanup():
            if self._runner:
                await self._runner.cleanup()
            self._runner = None

        run_async_cleanup(
            _cleanup(),
            timeout=0.5,
            loop_runner=self._tasks.loop_runner,
        )
        self._started = False
        L.info("HMI web service stopped")


class HmiOutput:
    def __init__(
        self,
        host: str,
        port: int,
        context: AppContextLike,
        index_path: str,
        task_reg=None,
        *,
        loop_runner: LoopRunner,
    ):
        self.server = _ApiServer(
            host,
            port,
            context,
            index_path=index_path,
            task_reg=task_reg,
            loop_runner=loop_runner,
        )

    def start(self):
        self.server.start()

    def stop(self):
        self.server.stop()

    def publish(self, rec: OutputRecord, overlay: tuple[bytes, str] | None):
        # HMI pulls data via HTTP; no push needed.
        _ = rec, overlay
        return None

    def publish_heartbeat(self, ts: float | None = None):
        # Heartbeat is served via /status; no push needed.
        _ = ts
        return None
