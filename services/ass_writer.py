from __future__ import annotations

from pathlib import Path

from models.subtitle_event import REVIEW_STYLE_NAME
from services.subtitle_parser import AssDocument


def _split_lines(content: str) -> list[str]:
    return content.splitlines(keepends=True)


def _find_style_section_bounds(lines: list[str]) -> tuple[int, int] | None:
    start = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[V4+ Styles]") or stripped.startswith("[V4 Styles]"):
            start = index
            break
    if start is None:
        return None
    end = len(lines)
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            end = index
            break
    return start, end


def _style_name_from_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith("Style:"):
        return None
    payload = stripped[len("Style:"):].lstrip()
    return payload.split(",", 1)[0].strip()


def _remove_review_style_lines(lines: list[str]) -> list[str]:
    return [
        line
        for line in lines
        if _style_name_from_line(line) != REVIEW_STYLE_NAME
    ]


def _normalize_style_section(style_section_lines: list[str]) -> list[str]:
    lines = _remove_review_style_lines(list(style_section_lines))
    normalized = [line if line.endswith(("\n", "\r")) else line + "\n" for line in lines]
    if normalized and normalized[-1].strip():
        normalized.append("\n")
    return normalized


def replace_ass_styles_section(content: str, style_section_lines: list[str]) -> str:
    replacement = _normalize_style_section(style_section_lines)
    if not replacement:
        return content

    lines = _split_lines(content)
    bounds = _find_style_section_bounds(lines)
    if bounds is not None:
        start, end = bounds
        lines[start:end] = replacement
        return "".join(lines)

    insert_index = next((index for index, line in enumerate(lines) if line.strip().startswith("[Events]")), len(lines))
    if insert_index > 0 and lines[insert_index - 1].strip():
        replacement = ["\n"] + replacement
    lines[insert_index:insert_index] = replacement
    return "".join(lines)


def remove_review_style_from_ass(content: str) -> str:
    lines = _split_lines(content)
    bounds = _find_style_section_bounds(lines)
    if bounds is None:
        return content
    start, end = bounds
    lines[start:end] = _remove_review_style_lines(lines[start:end])
    return "".join(lines)


class AssWriter:
    def write(self, document: AssDocument, output_path: str, style_section_lines: list[str] | None = None) -> str:
        content = document.dump()
        if style_section_lines:
            content = replace_ass_styles_section(content, style_section_lines)
        content = remove_review_style_from_ass(content)
        Path(output_path).write_text(content, encoding="utf-8")
        return output_path
