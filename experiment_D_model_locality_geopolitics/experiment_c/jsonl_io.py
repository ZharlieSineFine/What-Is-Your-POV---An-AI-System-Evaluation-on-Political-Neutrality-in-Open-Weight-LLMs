"""JSONL helpers shared by generation and judging."""

from __future__ import annotations

import json
import time
from pathlib import Path

# On Windows, sync clients (iCloud, OneDrive) and Excel briefly lock a file
# while uploading or showing it. Opening it then fails with PermissionError.
# Retrying for up to about 30 seconds rides out a sync lock without losing work.
LOCK_RETRIES = 30


def open_with_retry(path: Path, mode: str, **kwargs):
    for attempt in range(1, LOCK_RETRIES + 1):
        try:
            return path.open(mode, **kwargs)
        except PermissionError as exc:
            if attempt == LOCK_RETRIES:
                raise PermissionError(
                    f"{path} stayed locked by another program for about 30 seconds "
                    "(usually iCloud/OneDrive syncing it, or Excel having it open). "
                    "Pause syncing or close the file, then run the command again; "
                    "rows already saved are kept."
                ) from exc
            time.sleep(min(2.0, 0.2 * attempt))
    raise RuntimeError("unreachable")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_with_retry(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def iter_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open_with_retry(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError:
                # A crashed append can leave a partial last line. Skip it.
                continue
    return rows


def last_by_key(rows: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    """Keep the last record for each key, preserving first-seen order."""
    order: list[tuple] = []
    latest: dict[tuple, dict] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        if key not in latest:
            order.append(key)
        latest[key] = row
    return [latest[key] for key in order]
