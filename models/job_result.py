from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from providers.base import ProviderUsage


@dataclass(slots=True)
class StyleJobResult:
    matched: bool
    style_id: Optional[int] = None
    line_count: Optional[int] = None
    review_required: bool = False
    raw_response: str = ""
    usage: ProviderUsage = field(default_factory=ProviderUsage)


@dataclass(slots=True)
class TextJobResult:
    matched: bool
    text: str = ""
    raw_response: str = ""
    review_required: bool = False
    review_reasons: list[str] = field(default_factory=list)
    usage: ProviderUsage = field(default_factory=ProviderUsage)


@dataclass(slots=True)
class EventJobResult:
    event_id: str
    style_result: Optional[StyleJobResult] = None
    text_result: Optional[TextJobResult] = None
    final_action: str = "skip"
    error_messages: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)
