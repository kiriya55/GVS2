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


def extract_frame_ffmpeg(video_path: str, time_sec: float) -> bytes | None:
    try:
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
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=_build_startupinfo(), check=False)
        if result.returncode != 0:
            stderr_preview = result.stderr.decode("utf-8", errors="ignore")[:200]
            logger.warning("ffprobe duration probe failed (rc=%d): %s", result.returncode, stderr_preview)
            return None
        value = result.stdout.decode("utf-8", errors="ignore").strip()
        return float(value) if value else None
    except FileNotFoundError:
        logger.error("ffprobe not found on PATH")
        return None
    except (subprocess.SubprocessError, ValueError) as exc:
        logger.error("ffprobe duration error: %s", exc)
        return None


def probe_video_resolution_ffprobe(video_path: str) -> tuple[int, int] | None:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            video_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, startupinfo=_build_startupinfo(), check=False)
        if result.returncode != 0:
            stderr_preview = result.stderr.decode("utf-8", errors="ignore")[:200]
            logger.warning("ffprobe resolution probe failed (rc=%d): %s", result.returncode, stderr_preview)
            return None
        value = result.stdout.decode("utf-8", errors="ignore").strip()
        if not value or "x" not in value:
            return None
        width_text, height_text = value.split("x", 1)
        width = int(width_text)
        height = int(height_text)
        if width <= 0 or height <= 0:
            return None
        return width, height
    except FileNotFoundError:
        logger.error("ffprobe not found on PATH")
        return None
    except (subprocess.SubprocessError, ValueError) as exc:
        logger.error("ffprobe resolution error: %s", exc)
        return None
