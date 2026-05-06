from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

REVIEW_STYLE_NAME = "需核查"


@dataclass(slots=True)
class SubtitleEvent:
    event_id: str
    index: int
    line_index: int
    event_type: str
    format_fields: List[str]
    field_values: List[str]
    start_ms: int
    end_ms: int
    text: str
    original_style: str
    sample_times_ms: List[int] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def midpoint_ms(self) -> int:
        return self.start_ms + self.duration_ms // 2

    @property
    def style(self) -> str:
        try:
            idx = self.format_fields.index("Style")
        except ValueError:
            return self.original_style
        return self.field_values[idx]

    @style.setter
    def style(self, value: str) -> None:
        try:
            idx = self.format_fields.index("Style")
        except ValueError:
            return
        self.field_values[idx] = value

    def set_text(self, value: str) -> None:
        try:
            idx = self.format_fields.index("Text")
        except ValueError:
            return
        self.field_values[idx] = value
        self.text = value

    def to_ass_line(self) -> str:
        return f"{self.event_type}: " + ",".join(self.field_values)
