from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def _build_startupinfo() -> subprocess.STARTUPINFO | None:
    if os.name != "nt":
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startupinfo


def _run_probe(cmd: list[str], label: str) -> str | None:
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=_build_startupinfo(), check=False)
        if result.returncode != 0:
            stderr_preview = result.stderr.decode("utf-8", errors="ignore")[:200]
            logger.warning("%s failed (rc=%d): %s", label, result.returncode, stderr_preview)
            return None
        return result.stdout.decode("utf-8", errors="ignore").strip()
    except FileNotFoundError:
        logger.error("%s: binary not found on PATH", label)
        return None
    except subprocess.SubprocessError as exc:
        logger.error("%s error: %s", label, exc)
        return None


def extract_frame_ffmpeg(video_path: str, time_sec: float) -> bytes | None:
    cmd = [
        "ffmpeg",
        "-ss",
        str(time_sec),
        "-i",
        video_path,
        "-vframes",
        "1",
        "-q:v",
        "2",
        "-f",
        "image2",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=_build_startupinfo(), check=False)
        if result.returncode != 0:
            stderr_preview = result.stderr.decode("utf-8", errors="ignore")[:200]
            logger.warning("ffmpeg frame extraction failed (rc=%d): %s", result.returncode, stderr_preview)
            return None
        return result.stdout
    except FileNotFoundError:
        logger.error("ffmpeg not found on PATH")
        return None
    except subprocess.SubprocessError as exc:
        logger.error("ffmpeg subprocess error: %s", exc)
        return None


def probe_video_duration_ffprobe(video_path: str) -> float | None:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]
    value = _run_probe(cmd, "ffprobe duration")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        logger.error("ffprobe duration: cannot parse '%s'", value)
        return None


def probe_video_resolution_ffprobe(video_path: str) -> tuple[int, int] | None:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", video_path]
    value = _run_probe(cmd, "ffprobe resolution")
    if not value or "x" not in value:
        return None
    try:
        width_text, height_text = value.split("x", 1)
        width, height = int(width_text), int(height_text)
        return (width, height) if width > 0 and height > 0 else None
    except (ValueError, IndexError):
        logger.error("ffprobe resolution: cannot parse '%s'", value)
        return None
