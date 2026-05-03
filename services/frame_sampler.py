from __future__ import annotations

from typing import List

from models.subtitle_event import SubtitleEvent


def build_sample_times(event: SubtitleEvent, frame_count: int = 1) -> List[int]:
    if frame_count <= 1 or event.duration_ms <= 0:
        return [event.midpoint_ms]

    duration = event.duration_ms
    offsets = [0.2, 0.5, 0.8][:frame_count]
    points = [event.start_ms + int(duration * ratio) for ratio in offsets]
    return sorted({min(max(point, event.start_ms), event.end_ms) for point in points})
