# -- coding: utf-8 --

import argparse
import datetime
import socketserver


def _format_ts() -> str:
    ts = datetime.datetime.now()
    return f"{ts:%Y-%m-%d %H:%M:%S}.{ts.microsecond // 1000:03d}"


def _preview_bytes(data: bytes, max_preview: int) -> tuple[bytes, bool]:
    if max_preview <= 0 or len(data) <= max_preview:
        return data, False
    return data[:max_preview], True


def _hexlify(data: bytes) -> str:
    return " ".join(f"{b:02x}" for b in data)


class _TCPServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address,
        handler_cls,
        *,
        max_bytes: int,
        encoding: str,
        max_preview: int,
        show_text: bool,
    ):
        super().__init__(server_address, handler_cls)
        self.max_bytes = max_bytes
        self.encoding = encoding
        self.max_preview = max_preview
        self.show_text = show_text

    def log_payload(self, addr, data: bytes):
        ts = _format_ts()
        preview, truncated = _preview_bytes(data, self.max_preview)
        hex_text = _hexlify(preview)
        suffix = "..." if truncated else ""
        msg = f"{ts} len={len(data)} bytes hex={hex_text}{suffix}"
        if self.show_text and self.encoding:
            try:
                text_preview = preview.decode(self.encoding, errors="replace")
            except LookupError:
                text_preview = preview.decode("utf-8", errors="replace")
            msg += f" text={text_preview!r}{suffix}"
        print(msg, flush=True)


class _Handler(socketserver.BaseRequestHandler):
    server: _TCPServer

    def handle(self):
        addr = self.client_address
        print(f"{_format_ts()} CONNECT {addr[0]}:{addr[1]}", flush=True)
        while True:
            data = self.request.recv(self.server.max_bytes)
            if not data:
                break
            self.server.log_payload(addr, data)
        print(f"{_format_ts()} DISCONNECT {addr[0]}:{addr[1]}", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Listen on a TCP port and print incoming payloads"
    )
    p.add_argument("--host", default="0.0.0.0", help="Listen host")
    p.add_argument("--port", type=int, default=9000, help="Listen port")
    p.add_argument("--max-bytes", type=int, default=4096, help="Max bytes per recv")
    p.add_argument(
        "--max-preview",
        type=int,
        default=256,
        help="Preview bytes for hex/text output (0=full)",
    )
    p.add_argument(
        "--encoding", default="utf-8", help="Decode bytes using this encoding"
    )
    p.add_argument("--no-text", action="store_true", help="Disable decoded text output")
    args = p.parse_args()

    show_text = not args.no_text and bool(args.encoding)
    server = _TCPServer(
        (args.host, args.port),
        _Handler,
        max_bytes=args.max_bytes,
        encoding=args.encoding,
        max_preview=args.max_preview,
        show_text=show_text,
    )
    print(f"{_format_ts()} Listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"{_format_ts()} Stopped", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
