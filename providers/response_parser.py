from __future__ import annotations

import json
import re

from models.job_result import StyleJobResult, TextJobResult


NO_MATCH_JSON = '{"m":0}'


def _extract_json(raw: str) -> dict:
    text = raw.strip().replace("```json", "").replace("```", "").strip()
    if text == NO_MATCH_JSON:
        return {"m": 0}
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"response does not contain json: {raw}")
    json_str = text[start:end + 1]
    # 模型有时会在JSON字符串值里输出未转义的控制字符(如literal换行符0x0A),
    # 违反RFC 8259要求, 导致json.loads()抛出JSONDecodeError。
    # 这里统一将其替换为\uXXXX转义形式, json.loads()会正确还原为对应字符。
    json_str = re.sub(r'[\x00-\x1f]', lambda m: f'\\u{ord(m.group(0)):04x}', json_str)
    return json.loads(json_str)


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
    text = str(data["t"]).replace("\n", "\\N")  # 换行符转ASS格式
    return TextJobResult(matched=True, text=text, raw_response=raw)
