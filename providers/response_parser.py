from __future__ import annotations

import json
import re

from models.job_result import StyleJobResult, TextJobResult
from providers.base import NO_MATCH_JSON


def _extract_json(raw: str) -> dict:
    text = raw.strip().replace("```json", "").replace("```", "").strip()
    if text == NO_MATCH_JSON:
        return {"m": 0}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"response does not contain json: {raw}")
    json_str = text[start:end + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        sanitized = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f]', lambda m: f'\\u{ord(m.group(0)):04x}', json_str)
        return json.loads(sanitized)


def parse_style_result(raw: str) -> StyleJobResult:
    data = _extract_json(raw)
    if data.get("m") == 0:
        return StyleJobResult(matched=False, raw_response=raw)
    return StyleJobResult(
        matched=True,
        style_id=int(data["s"]),
        line_count=int(data["l"]),
        review_required=bool(data.get("r", 0)),
        raw_response=raw,
    )


def parse_text_result(raw: str) -> TextJobResult:
    data = _extract_json(raw)
    if data.get("m") == 0:
        return TextJobResult(matched=False, raw_response=raw)
    text = str(data["t"]).replace("\n", "\\N")
    return TextJobResult(matched=True, text=text, raw_response=raw)
