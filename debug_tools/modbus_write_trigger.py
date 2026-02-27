# -- coding: utf-8 --

"""Single-run Modbus TCP trigger writer (toggle or fixed value)."""

import argparse
import sys

from pymodbus.client import ModbusTcpClient


def main():
    p = argparse.ArgumentParser(description="Write a Modbus trigger coil once and exit")
    p.add_argument("--host", default="127.0.0.1", help="Modbus TCP host")
    p.add_argument("--port", type=int, default=1502, help="Modbus TCP port")
    p.add_argument("--device-id", type=int, default=1, help="Device ID")
    p.add_argument(
        "--coil-offset", type=int, default=800, help="Trigger coil offset (PDU 0-based)"
    )
    p.add_argument(
        "--value",
        type=int,
        choices=[0, 1],
        default=None,
        help="Write fixed value (0/1)",
    )
    args = p.parse_args()

    print(f"Connecting TCP {args.host}:{args.port}")
    with ModbusTcpClient(host=args.host, port=args.port) as client:
        if not client.connect():
            print(f"Failed to connect to {args.host}:{args.port}")
            sys.exit(1)

        value = args.value
        if value is None:
            read_res = client.read_coils(
                address=args.coil_offset, count=1, device_id=args.device_id
            )
            if read_res.isError():
                print(f"Read coil error: {read_res}")
                sys.exit(1)
            current = 1 if (read_res.bits and read_res.bits[0]) else 0
            value = 0 if current else 1

        res = client.write_coil(
            address=args.coil_offset, value=bool(value), device_id=args.device_id
        )
        if res.isError():
            print(f"Write coil error: {res}")
            sys.exit(1)

        print(f"Wrote coil @{args.coil_offset} value={value}")


if __name__ == "__main__":
    main()
