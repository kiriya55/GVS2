from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SettingsStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("settings load failed (%s), returning empty dict", exc)
            return {}

    def save(self, data: dict[str, Any]) -> None:
        current = self.load()
        current.update(data)
        tmp = self.path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            logger.error("settings save failed: %s", exc)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise
