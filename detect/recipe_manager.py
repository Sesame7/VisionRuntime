from __future__ import annotations

import copy
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import yaml

from .base import create_detector

L = logging.getLogger("vision_runtime.recipe")


@dataclass(slots=True)
class _RecipeEntry:
    name: str
    path: str
    valid: bool
    params: dict[str, Any] | None = None
    error: str | None = None


class RecipeManager:
    def __init__(
        self,
        *,
        impl: str,
        recipe_dir: str,
        default_recipe: str,
        preview_enabled: bool,
        preview_max_edge: int,
        input_pixel_format: str,
    ):
        self.impl = str(impl)
        self.recipe_dir = os.path.abspath(str(recipe_dir))
        self.default_recipe = str(default_recipe)
        self.preview_enabled = bool(preview_enabled)
        self.preview_max_edge = int(preview_max_edge)
        self.input_pixel_format = str(input_pixel_format)

        self._lock = threading.Lock()
        self._entries: dict[str, _RecipeEntry] = {}
        self._ordered_names: list[str] = []
        self._active_recipe: str = ""
        self._load_all()

    def _load_all(self) -> None:
        if not os.path.isdir(self.recipe_dir):
            raise ValueError(f"recipe_dir not found: {self.recipe_dir}")
        names = [
            entry
            for entry in sorted(os.listdir(self.recipe_dir))
            if _is_yaml_file(entry)
            and os.path.isfile(os.path.join(self.recipe_dir, entry))
        ]
        if not names:
            raise ValueError(f"No recipe files found under: {self.recipe_dir}")

        entries: dict[str, _RecipeEntry] = {}
        for name in names:
            path = os.path.join(self.recipe_dir, name)
            entry = _RecipeEntry(name=name, path=path, valid=False)
            try:
                params = _read_yaml_mapping(path)
                # Validate business params by constructing detector once.
                _ = create_detector(
                    self.impl,
                    params,
                    generate_overlay=self.preview_enabled,
                    input_pixel_format=self.input_pixel_format,
                    preview_max_edge=self.preview_max_edge,
                )
                entry.valid = True
                entry.params = params
            except Exception as exc:
                entry.error = str(exc) or type(exc).__name__
                L.warning(
                    "Recipe invalid name=%s path=%s err=%s", name, path, entry.error
                )
            entries[name] = entry

        default_entry = entries.get(self.default_recipe)
        if default_entry is None:
            raise ValueError(
                f"default_recipe not found under recipe_dir: {self.default_recipe}"
            )
        if not default_entry.valid:
            raise ValueError(
                f"default_recipe invalid: {self.default_recipe}; {default_entry.error}"
            )

        with self._lock:
            self._entries = entries
            self._ordered_names = names
            self._active_recipe = self.default_recipe

    def active_recipe(self) -> str:
        with self._lock:
            return self._active_recipe

    def set_active_recipe(self, recipe_name: str) -> None:
        name = str(recipe_name)
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise ValueError(f"recipe not found: {name}")
            if not entry.valid:
                raise ValueError(f"recipe is invalid: {name}")
            self._active_recipe = name

    def list_slots(self) -> list[dict[str, Any]]:
        with self._lock:
            names = list(self._ordered_names)
            entries = dict(self._entries)
        return [
            {
                "slot": idx,
                "name": entries[name].name,
                "valid": bool(entries[name].valid),
                "error": str(entries[name].error or ""),
            }
            for idx, name in enumerate(names, start=1)
        ]

    def recipe_count(self) -> int:
        with self._lock:
            return len(self._ordered_names)

    def recipe_name_at_slot(self, slot: int) -> str:
        try:
            idx = int(slot)
        except Exception as exc:
            raise ValueError(f"recipe slot must be an integer: {slot!r}") from exc
        with self._lock:
            if idx < 1 or idx > len(self._ordered_names):
                raise ValueError(f"recipe slot out of range: {idx}")
            return self._ordered_names[idx - 1]

    def slot_of_recipe(self, recipe_name: str) -> int:
        name = str(recipe_name)
        with self._lock:
            if name not in self._entries:
                raise ValueError(f"recipe not found: {name}")
            for idx, recipe in enumerate(self._ordered_names, start=1):
                if recipe == name:
                    return idx
        raise ValueError(f"recipe not found: {name}")

    def build_detector_for(self, recipe_name: str):
        name = str(recipe_name)
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                raise ValueError(f"recipe not found: {name}")
            if not entry.valid or entry.params is None:
                raise ValueError(f"recipe is invalid: {name}")
            params = copy.deepcopy(entry.params)
        return create_detector(
            self.impl,
            params,
            generate_overlay=self.preview_enabled,
            input_pixel_format=self.input_pixel_format,
            preview_max_edge=self.preview_max_edge,
        )


def _read_yaml_mapping(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"recipe YAML root must be a mapping: {path}")
    return data


def _is_yaml_file(path: str) -> bool:
    lower = str(path).lower()
    return lower.endswith(".yaml") or lower.endswith(".yml")


__all__ = ["RecipeManager"]
