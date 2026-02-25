from __future__ import annotations

import importlib
from collections.abc import MutableMapping
from typing import TypeVar

T = TypeVar("T")


def register_named(registry: MutableMapping[str, T], name: str):
    """Decorator to register an object under a string key."""

    def decorator(obj: T) -> T:
        registry[name] = obj
        return obj

    return decorator


def resolve_registered(
    registry: MutableMapping[str, T],
    name: str,
    *,
    package: str,
    unknown_label: str,
) -> T:
    """Resolve a registry entry, lazily importing `<package>.<name>` if needed."""
    if name not in registry:
        importlib.import_module(f"{package}.{name}")
    if name not in registry:
        raise ValueError(
            f"Unknown {unknown_label} '{name}'. "
            f"Available: {', '.join(registry.keys()) or 'none'}"
        )
    return registry[name]


__all__ = ["register_named", "resolve_registered"]
