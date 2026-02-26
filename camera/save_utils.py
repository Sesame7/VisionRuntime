from __future__ import annotations

import os
from datetime import datetime, timezone


def normalize_utc_datetime(value: datetime | None) -> datetime:
    ref = value or datetime.now(timezone.utc)
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    return ref.astimezone(timezone.utc)


def format_frame_filename(
    frame_id: int | float, ext: str, ts_utc: datetime | None = None
) -> str:
    ref = normalize_utc_datetime(ts_utc)
    ts = ref.strftime("%H-%M-%S.%f")[:-3] + "Z"
    return f"{ts}_{int(frame_id):05d}{ext}"


class DailyDirCache:
    """Cache the current UTC date directory to avoid repeated mkdir/path joins."""

    def __init__(self):
        self._root_key: str | None = None
        self._date_key: str | None = None
        self._dir_path: str | None = None

    def get_or_create(self, root_dir: str, ts_utc: datetime) -> str:
        ref = normalize_utc_datetime(ts_utc)
        date_key = ref.date().isoformat()
        root_key = os.path.abspath(root_dir)
        if (
            self._root_key != root_key
            or self._date_key != date_key
            or not self._dir_path
        ):
            target_dir = os.path.join(root_dir, date_key)
            os.makedirs(target_dir, exist_ok=True)
            self._root_key = root_key
            self._date_key = date_key
            self._dir_path = target_dir
        return self._dir_path


def build_dated_frame_path(
    root_dir: str,
    frame_id: int | float,
    ext: str,
    *,
    ts_utc: datetime | None,
    cache: DailyDirCache,
) -> tuple[str, datetime]:
    ref = normalize_utc_datetime(ts_utc)
    target_dir = cache.get_or_create(root_dir, ref)
    return os.path.join(
        target_dir, format_frame_filename(frame_id, ext, ts_utc=ref)
    ), ref


__all__ = [
    "DailyDirCache",
    "build_dated_frame_path",
    "format_frame_filename",
    "normalize_utc_datetime",
]
