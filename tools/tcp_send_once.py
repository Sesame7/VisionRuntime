# -- coding: utf-8 --

import argparse
import datetime
import socket


def _format_ts() -> str:
	ts = datetime.datetime.now()
	return f"{ts:%Y-%m-%d %H:%M:%S}.{ts.microsecond // 1000:03d}"


def _hexlify(data: bytes) -> str:
	return " ".join(f"{b:02x}" for b in data)


def main():
	p = argparse.ArgumentParser(description="Send a TCP trigger to the listener")
	p.add_argument('--host', default='127.0.0.1', help='Trigger listener host')
	p.add_argument('--port', type=int, default=9000, help='Trigger listener port')
	p.add_argument('--word', default='TRIG', help='Trigger word')
	args = p.parse_args()

	payload = args.word.encode('utf-8')
	with socket.create_connection((args.host, args.port), timeout=2.0) as conn:
		conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		local = conn.getsockname()
		print(f"{_format_ts()} CONNECT {args.host}:{args.port} local={local[0]}:{local[1]}")
		conn.sendall(payload)
		hex_text = _hexlify(payload)
		text_preview = payload.decode("utf-8", errors="replace")
		print(f"{_format_ts()} SEND len={len(payload)} bytes hex={hex_text} text={text_preview!r}")
	print(f"{_format_ts()} CLOSE {args.host}:{args.port}")


if __name__ == "__main__":
	main()
