# -- coding: utf-8 --

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Type
from core.registry import register_named, resolve_registered

TriggerFactory = Dict[str, Type["BaseTrigger"]]
_registry: TriggerFactory = {}


@dataclass
class TriggerConfig:
    host: str = "0.0.0.0"
    port: int = 9000
    word: bytes = b"TRIG"
    global_min_interval_ms: float = 0.0
    high_priority_cooldown_ms: float = 0.0
    high_priority_sources: list[str] = field(default_factory=list)
    low_priority_sources: list[str] = field(default_factory=list)
    ip_whitelist: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.word = _ensure_bytes(self.word)


class BaseTrigger(ABC):
    def __init__(self, cfg: TriggerConfig, on_trigger: Callable):
        self.cfg = cfg
        self.on_trigger = on_trigger

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def raise_if_failed(self):
        return None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()


def register_trigger(name: str):
    return register_named(_registry, name)


def create_trigger(
    name: str, cfg: TriggerConfig, on_trigger: Callable, **kwargs
) -> BaseTrigger:
    cls = resolve_registered(
        _registry,
        name,
        package=__package__ or "trigger",
        unknown_label="trigger type",
    )
    return cls(cfg, on_trigger, **kwargs)


def build_trigger_config_from_loaded_config(cfg) -> TriggerConfig:
    return TriggerConfig(
        host=cfg.comm.tcp.host,
        port=cfg.comm.tcp.port,
        word=cfg.trigger.tcp.word,
        global_min_interval_ms=cfg.trigger.global_min_interval_ms,
        high_priority_cooldown_ms=cfg.trigger.high_priority_cooldown_ms,
        high_priority_sources=cfg.trigger.high_priority_sources,
        low_priority_sources=cfg.trigger.low_priority_sources,
        ip_whitelist=cfg.trigger.ip_whitelist,
    )


def _ensure_bytes(word) -> bytes:
    if isinstance(word, bytes):
        return word
    if isinstance(word, str):
        return word.encode("utf-8")
    return str(word).encode("utf-8")


__all__ = [
    "TriggerConfig",
    "BaseTrigger",
    "register_trigger",
    "create_trigger",
    "build_trigger_config_from_loaded_config",
]
