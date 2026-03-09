# -- coding: utf-8 --

"""Single-run Modbus TCP holding-register writer (generic or recipe shortcut)."""

import argparse
import sys

from pymodbus.client import ModbusTcpClient


def _ensure_u16(value: int, label: str) -> int:
    iv = int(value)
    if iv < 0 or iv > 0xFFFF:
        raise ValueError(f"{label} must be in range 0..65535, got {value}")
    return iv


def main():
    p = argparse.ArgumentParser(
        description="Write Modbus holding registers once and exit"
    )
    p.add_argument("--host", default="127.0.0.1", help="Modbus TCP host")
    p.add_argument("--port", type=int, default=1502, help="Modbus TCP port")
    p.add_argument("--device-id", type=int, default=1, help="Device ID")
    p.add_argument(
        "--hr-offset", type=int, default=50, help="Holding-register base offset (PDU)"
    )
    p.add_argument(
        "--address",
        type=int,
        default=0,
        help="Relative address under --hr-offset (generic mode)",
    )
    p.add_argument(
        "--values",
        type=int,
        nargs="+",
        default=None,
        help="One or more uint16 values in generic mode",
    )
    p.add_argument(
        "--slot",
        type=int,
        default=None,
        help="Recipe slot shortcut: writes to HR0 (used with --seq)",
    )
    p.add_argument(
        "--seq",
        type=int,
        default=None,
        help="Recipe sequence shortcut: writes to HR1 (used with --slot)",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Read back written registers and print them",
    )
    args = p.parse_args()

    if args.hr_offset < 0:
        print("--hr-offset must be >= 0")
        sys.exit(2)
    if args.address < 0:
        print("--address must be >= 0")
        sys.exit(2)

    use_recipe_shortcut = args.slot is not None or args.seq is not None
    if use_recipe_shortcut:
        if args.slot is None or args.seq is None:
            print("Recipe shortcut requires both --slot and --seq")
            sys.exit(2)
        if args.values is not None:
            print("--values cannot be used together with --slot/--seq")
            sys.exit(2)
        try:
            values = [
                _ensure_u16(args.slot, "slot"),
                _ensure_u16(args.seq, "seq"),
            ]
        except ValueError as exc:
            print(str(exc))
            sys.exit(2)
        start_addr = args.hr_offset
    else:
        if not args.values:
            print("Generic mode requires --values")
            sys.exit(2)
        try:
            values = [_ensure_u16(v, f"values[{i}]") for i, v in enumerate(args.values)]
        except ValueError as exc:
            print(str(exc))
            sys.exit(2)
        start_addr = args.hr_offset + args.address

    print(f"Connecting TCP {args.host}:{args.port}")
    with ModbusTcpClient(host=args.host, port=args.port) as client:
        if not client.connect():
            print(f"Failed to connect to {args.host}:{args.port}")
            sys.exit(1)

        if len(values) == 1:
            res = client.write_register(
                address=start_addr, value=values[0], device_id=args.device_id
            )
        else:
            res = client.write_registers(
                address=start_addr, values=values, device_id=args.device_id
            )
        if res.isError():
            print(f"Write holding registers error: {res}")
            sys.exit(1)

        print(f"Wrote HR @{start_addr} values={values}")

        if args.verify:
            read_res = client.read_holding_registers(
                address=start_addr, count=len(values), device_id=args.device_id
            )
            if read_res.isError():
                print(f"Read-back error: {read_res}")
                sys.exit(1)
            print(
                f"Read-back HR @{start_addr} count={len(values)}: {read_res.registers}"
            )


if __name__ == "__main__":
    main()
