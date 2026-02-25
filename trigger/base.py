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


__all__ = ["TriggerConfig", "BaseTrigger", "register_trigger", "create_trigger"]
