from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    import fcntl as _fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False  # Windows — no advisory locking, fall back silently


class MonitorLog:
    """Append-only JSON-lines log stored at ``log_path/monitor.jsonl``."""

    def __init__(self, log_path: str | os.PathLike) -> None:
        self._path = Path(log_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._file = self._path / "monitor.jsonl"

    def append(self, entry: dict[str, Any]) -> None:
        """Append one log entry (must be JSON-serialisable). Thread- and process-safe on Unix."""
        with self._file.open("a", encoding="utf-8") as fh:
            if _HAS_FCNTL:
                _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
            try:
                fh.write(json.dumps(entry, default=str) + "\n")
            finally:
                if _HAS_FCNTL:
                    _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)

    def read_last(self, n: int = 100) -> list[dict[str, Any]]:
        """Return the last *n* log entries, oldest first."""
        return self.read_all()[-n:]

    def read_all(self) -> list[dict[str, Any]]:
        """Return all log entries, oldest first."""
        if not self._file.exists():
            return []
        entries: list[dict[str, Any]] = []
        with self._file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    def clear(self) -> None:
        """Delete all log entries."""
        if self._file.exists():
            self._file.unlink()
