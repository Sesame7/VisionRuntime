# -- coding: utf-8 --
"""Compatibility helpers for pymodbus 3.x imports and APIs."""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any, TYPE_CHECKING, TypeAlias

from pymodbus.datastore import ModbusSequentialDataBlock, ModbusServerContext
from pymodbus.server import ModbusTcpServer


def _import_attr(module_name: str, attr_name: str) -> Any | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, attr_name, None)


def _first_attr(candidates: list[tuple[str, str]], label: str) -> Any:
    for module_name, attr_name in candidates:
        value = _import_attr(module_name, attr_name)
        if value is not None:
            return value
    raise ImportError(f"pymodbus missing {label}; tried {candidates!r}")


ExcCodes = _import_attr("pymodbus.constants", "ExcCodes")
ExceptionResponse = _import_attr("pymodbus.pdu", "ExceptionResponse")

ModbusDeviceContext = _first_attr(
    [
        ("pymodbus.datastore", "ModbusDeviceContext"),
        ("pymodbus.datastore", "ModbusSlaveContext"),
    ],
    "ModbusDeviceContext/ModbusSlaveContext",
)

if TYPE_CHECKING:
    try:
        from pymodbus.datastore import ModbusDeviceContext as _ModbusDeviceContextType  # type: ignore[reportAttributeAccessIssue]
    except Exception:
        from pymodbus.datastore import ModbusSlaveContext as _ModbusDeviceContextType  # type: ignore[reportAttributeAccessIssue]
    ModbusDeviceContextType: TypeAlias = _ModbusDeviceContextType
else:
    ModbusDeviceContextType: TypeAlias = Any

ModbusServerContextType: TypeAlias = ModbusServerContext
ModbusTcpServerType: TypeAlias = ModbusTcpServer


def is_modbus_exception(value: object) -> bool:
    if ExceptionResponse is not None and isinstance(value, ExceptionResponse):
        return True
    if ExcCodes is not None and isinstance(value, ExcCodes):
        return True
    return False


def build_server_context(device_ctx: Any) -> Any:
    try:
        return ModbusServerContext(devices=device_ctx, single=True)
    except TypeError:
        return ModbusServerContext(slaves=device_ctx, single=True)


__all__ = [
    "ExcCodes",
    "ExceptionResponse",
    "ModbusDeviceContext",
    "ModbusDeviceContextType",
    "ModbusSequentialDataBlock",
    "ModbusServerContext",
    "ModbusServerContextType",
    "ModbusTcpServer",
    "ModbusTcpServerType",
    "build_server_context",
    "is_modbus_exception",
]
