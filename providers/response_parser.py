from __future__ import annotations

import json
import re

from models.job_result import StyleJobResult, TextJobResult
from providers.base import NO_MATCH_JSON


def _clean_response_text(raw: str) -> str:
    return raw.strip().replace("```json", "").replace("```", "").strip()


def _extract_json(raw: str) -> dict:
    text = _clean_response_text(raw)
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
        try:
            return json.loads(json_str, strict=False)
        except json.JSONDecodeError:
            pass
        sanitized = re.sub(r'[\x00-\x09\x0b\x0c\x0e-\x1f\r]', lambda m: f'\\u{ord(m.group(0)):04x}', json_str)
        return json.loads(sanitized)


def _decode_recovered_text_value(value: str) -> str | None:
    quote_index: int | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            quote_index = index
            break

    if quote_index is not None:
        remainder = value[quote_index + 1 :].strip()
        if remainder not in ("", "}"):
            return None
        value = value[:quote_index]
    elif not value or value.endswith("\\"):
        return None

    try:
        return json.loads(f'"{value}"', strict=False)
    except json.JSONDecodeError:
        return value


def _recover_truncated_text_json(raw: str) -> dict | None:
    text = _clean_response_text(raw)
    match = re.fullmatch(r'\{\s*"m"\s*:\s*1\s*,\s*"t"\s*[:：]?\s*"(?P<text>.*)', text, flags=re.DOTALL)
    if match is None:
        return None

    value = _decode_recovered_text_value(match.group("text"))
    if value is None:
        return None
    return {"m": 1, "t": value, "r": 1, "_review_reason": "模型返回的文字 JSON 不完整或格式异常"}


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
    try:
        data = _extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        data = _recover_truncated_text_json(raw)
        if data is None:
            raise
    if data.get("m") == 0:
        return TextJobResult(matched=False, raw_response=raw)
    text = str(data["t"]).replace("\n", "\\N")
    review_reasons = []
    if data.get("r"):
        review_reasons.append(str(data.get("_review_reason") or "模型标记该文字识别结果需要人工核查"))
    return TextJobResult(matched=True, text=text, raw_response=raw, review_required=bool(review_reasons), review_reasons=review_reasons)
