# -- coding: utf-8 --

"""Single-run Modbus TCP reader for v2.2 DI/IR layout (PDU 0-based)."""

import argparse
import sys

from pymodbus.client import ModbusTcpClient


def main():
	p = argparse.ArgumentParser(description="Read Modbus v2.2 DI/IR once and exit")
	p.add_argument("--host", default="127.0.0.1", help="Modbus TCP host")
	p.add_argument("--port", type=int, default=1502, help="Modbus TCP port")
	p.add_argument("--device-id", type=int, default=1, help="Device ID")
	p.add_argument("--coil-offset", type=int, default=800, help="Coil start offset (PDU 0-based)")
	p.add_argument("--coil-count", type=int, default=2, help="Number of coils to read")
	p.add_argument("--di-offset", type=int, default=800, help="Discrete input start offset (PDU 0-based)")
	p.add_argument("--di-count", type=int, default=6, help="Number of discrete inputs to read")
	p.add_argument("--ir-offset", type=int, default=50, help="Input register start offset (PDU 0-based)")
	p.add_argument("--ir-count", type=int, default=10, help="Number of input registers to read")
	args = p.parse_args()

	print(f"Connecting TCP {args.host}:{args.port}")
	with ModbusTcpClient(host=args.host, port=args.port) as client:
		if not client.connect():
			print(f"Failed to connect to {args.host}:{args.port}")
			sys.exit(1)

		coil_res = client.read_coils(address=args.coil_offset, count=args.coil_count, device_id=args.device_id)
		di_res = client.read_discrete_inputs(address=args.di_offset, count=args.di_count, device_id=args.device_id)
		ir_res = client.read_input_registers(address=args.ir_offset, count=args.ir_count, device_id=args.device_id)

		if coil_res.isError():
			print(f"Coils    @{args.coil_offset} error: {coil_res}")
		else:
			coil_vals = [1 if v else 0 for v in list(coil_res.bits)[: args.coil_count]]
			print(f"Coils    @{args.coil_offset} count={args.coil_count}: {coil_vals}")

		if di_res.isError():
			print(f"DInputs  @{args.di_offset} error: {di_res}")
		else:
			di_vals = [1 if v else 0 for v in list(di_res.bits)[: args.di_count]]
			print(f"DInputs  @{args.di_offset} count={args.di_count}: {di_vals}")

		if ir_res.isError():
			print(f"IRegister @{args.ir_offset} error: {ir_res}")
		else:
			print(f"IRegister @{args.ir_offset} count={args.ir_count}: {list(ir_res.registers)}")


if __name__ == "__main__":
	main()
