from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".png": "image/png",
}


@dataclass(slots=True)
class StyleSample:
    image_path: str = ""
    timestamp_ms: int = 0
    note: str = ""


@dataclass(slots=True)
class StyleProfile:
    style_id: int
    display_name: str
    ass_style_name: str
    feature_notes: str = ""
    layout_hint: str = "either"
    samples: List[StyleSample] = field(default_factory=list)

    @property
    def short_label(self) -> str:
        return str(self.style_id)


def load_sample_images(profiles: list[StyleProfile], max_per_style: int = 1) -> list[tuple[int, str, str]]:
    """Load sample images from disk for style profiles that have saved samples.

    Returns a list of (style_id, mime_type, base64_data) tuples.
    At most *max_per_style* images are loaded per style.
    """
    loaded: list[tuple[int, str, str]] = []
    for profile in profiles:
        count = 0
        for sample in profile.samples:
            if count >= max_per_style:
                break
            path = Path(sample.image_path)
            if not path.exists():
                logger.debug("sample image not found: %s", sample.image_path)
                continue
            suffix = path.suffix.lower()
            mime = _MIME_MAP.get(suffix, "image/jpeg")
            try:
                data = base64.b64encode(path.read_bytes()).decode("ascii")
                loaded.append((profile.style_id, mime, data))
                count += 1
            except OSError as exc:
                logger.warning("failed to load sample image %s: %s", sample.image_path, exc)
    return loaded
