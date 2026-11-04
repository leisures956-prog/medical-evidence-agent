from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class LLMResponseCache:
    """Small file cache that never stores credentials or writes outside the project."""

    def __init__(self, directory: Path | str | None = None, enabled: bool | None = None):
        project_root = Path(__file__).resolve().parents[1]
        configured = directory or os.getenv("LLM_CACHE_DIR")
        self.directory = Path(configured) if configured else project_root / "artifacts" / "llm_cache"
        if enabled is None:
            enabled = os.getenv("LLM_CACHE_ENABLED", "1").strip().lower() not in {
                "0",
                "false",
                "no",
            }
        self.enabled = enabled

    @staticmethod
    def key(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self.directory / f"{key}.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def delete(self, key: str) -> None:
        """Remove one cached response when it failed downstream validation."""

        if not self.enabled:
            return
        try:
            (self.directory / f"{key}.json").unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # A cache cleanup failure must not hide the original validation error.
            pass
