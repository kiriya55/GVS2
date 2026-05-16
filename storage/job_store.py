from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from models.job_result import EventJobResult


def default_jobs_dir() -> Path:
    appdata = os.environ.get("APPDATA", "").strip()
    candidates = []
    if appdata:
        candidates.append(Path(appdata) / "GVS2" / "jobs")
    candidates.append(Path(__file__).resolve().parents[1] / "storage" / "jobs")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    fallback = Path(__file__).resolve().parents[1] / "storage" / "jobs"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


class JobStore:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else default_jobs_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return default

    def _write_json(self, path: Path, payload: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _make_job_id(self, video_path: str, subtitle_input_path: str, ass_output_path: str, created_at: str) -> str:
        raw = "\n".join([video_path, subtitle_input_path, ass_output_path, created_at])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        timestamp = created_at.replace("-", "").replace(":", "").replace("+", "Z").split(".")[0]
        return f"{timestamp}-{digest}"

    def _output_key(self, ass_output_path: str) -> str:
        return str(Path(ass_output_path).expanduser().resolve(strict=False))

    def _task_status(self, item: EventJobResult, task: str) -> dict[str, Any]:
        result = item.style_result if task == "style" else item.text_result
        if task in item.failed_tasks:
            status = {"status": "failed"}
            if result is not None and result.raw_response:
                status["raw_response"] = result.raw_response
            return status
        if result is None:
            return {"status": "skipped"}
        status = {"status": "success" if result.matched else "no_match"}
        if result.raw_response:
            status["raw_response"] = result.raw_response
        if task == "text" and getattr(result, "text", ""):
            status["parsed_text"] = result.text
        if task == "text" and getattr(result, "review_required", False):
            status["review_required"] = True
            status["review_reasons"] = list(getattr(result, "review_reasons", []))
        return status

    def save_run(
        self,
        video_path: str,
        subtitle_input_path: str,
        ass_output_path: str,
        results: list[EventJobResult],
        event_lookup: dict[str, Any],
        previous_record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        created_at = datetime.now(timezone.utc).isoformat()
        job_id = self._make_job_id(video_path, subtitle_input_path, ass_output_path, created_at)
        previous_events = {
            str(item.get("event_id", "")): item
            for item in (previous_record or {}).get("events", [])
        }
        events = []
        for item in results:
            event = event_lookup.get(item.event_id)
            tasks = {
                "style": self._task_status(item, "style"),
                "text": self._task_status(item, "text"),
            }
            previous = previous_events.get(item.event_id, {})
            previous_tasks = previous.get("tasks", {})
            for task_name, task_status in tasks.items():
                if task_status["status"] == "skipped" and previous_tasks.get(task_name):
                    tasks[task_name] = previous_tasks[task_name]
            events.append(
                {
                    "event_id": item.event_id,
                    "start_ms": event.start_ms if event is not None else None,
                    "end_ms": event.end_ms if event is not None else None,
                    "text": event.text if event is not None else "",
                    "original_style": event.original_style if event is not None else "",
                    "final_action": item.final_action,
                    "error_messages": item.error_messages,
                    "failed_tasks": item.failed_tasks,
                    "tasks": tasks,
                }
            )

        record = {
            "schema_version": 1,
            "job_id": job_id,
            "created_at": created_at,
            "video_path": video_path,
            "subtitle_input_path": subtitle_input_path,
            "ass_output_path": ass_output_path,
            "summary": {
                "events": len(results),
                "failed_events": sum(1 for item in results if item.error_messages),
                "failed_tasks": sum(len(item.failed_tasks) for item in results),
            },
            "events": events,
        }
        self._write_json(self.root / f"{job_id}.json", record)

        index = self._read_json(self.index_path, {"outputs": {}})
        outputs = index.setdefault("outputs", {})
        outputs[self._output_key(ass_output_path)] = job_id
        self._write_json(self.index_path, index)
        return record

    def path_for(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def load(self, job_id: str) -> dict[str, Any]:
        return self._read_json(self.root / f"{job_id}.json", {})

    def load_latest_for_output(self, ass_output_path: str) -> dict[str, Any]:
        index = self._read_json(self.index_path, {"outputs": {}})
        job_id = index.get("outputs", {}).get(self._output_key(ass_output_path), "")
        return self.load(job_id) if job_id else {}

    def failed_tasks_map(self, record: dict[str, Any]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for item in record.get("events", []):
            event_id = str(item.get("event_id", "")).strip()
            tasks = item.get("failed_tasks") or []
            if event_id and tasks:
                result[event_id] = [str(task) for task in tasks]
        return result
