from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List

from models.subtitle_event import SubtitleEvent

logger = logging.getLogger(__name__)


class AssDocument:
    def __init__(self, lines: List[str], event_indices: List[int], events: List[SubtitleEvent]) -> None:
        self.lines = lines
        self.event_indices = event_indices
        self.events = events

    def apply_event_updates(self) -> None:
        logger.info(f"apply_event_updates: 更新 {len(self.events)} 个事件的ASS行")
        for line_index, event in zip(self.event_indices, self.events):
            self.lines[line_index] = event.to_ass_line() + "\n"

    def dump(self) -> str:
        logger.info(f"AssDocument.dump: 开始生成输出，共 {len(self.events)} 个事件")
        self.apply_event_updates()
        result = "".join(self.lines)
        logger.info(f"AssDocument.dump: 输出生成完成，共 {len(result)} 字符")
        return result


class GeneratedAssDocument(AssDocument):
    def __init__(self, events: List[SubtitleEvent]) -> None:
        super().__init__(lines=[], event_indices=[], events=events)
        self.play_res_x = 1920
        self.play_res_y = 1080
        self.font_name = "Arial"
        self.font_size = 54
        self.margin_l = 48
        self.margin_r = 48
        self.margin_v = 42
        self.outline = 3
        self.shadow = 1
        self.alignment = 2

    def configure_render_profile(
        self,
        play_res_x: int,
        play_res_y: int,
        subtitle_region_start: int = 66,
        subtitle_region_end: int = 100,
    ) -> None:
        self.play_res_x = max(640, int(play_res_x))
        self.play_res_y = max(360, int(play_res_y))
        self.font_size = max(28, min(72, int(round(self.play_res_y * 0.05))))
        self.margin_l = max(24, int(round(self.play_res_x * 0.03)))
        self.margin_r = self.margin_l
        bottom_padding = int(round(self.play_res_y * max(0, 100 - subtitle_region_end) / 100))
        region_height = int(round(self.play_res_y * max(0, subtitle_region_end - subtitle_region_start) / 100))
        self.margin_v = max(28, bottom_padding + max(18, int(round(region_height * 0.12))))
        self.outline = max(2, int(round(self.font_size * 0.06)))
        self.shadow = max(1, int(round(self.outline * 0.5)))

    def dump(self) -> str:
        logger.info(f"GeneratedAssDocument.dump: 开始生成ASS输出，共 {len(self.events)} 个事件")
        header = [
            "[Script Info]\n",
            "; Generated from SRT by GVS2\n",
            "ScriptType: v4.00+\n",
            f"PlayResX: {self.play_res_x}\n",
            f"PlayResY: {self.play_res_y}\n",
            "WrapStyle: 0\n",
            "ScaledBorderAndShadow: yes\n",
            "YCbCr Matrix: TV.601\n",
            "\n",
            "[V4+ Styles]\n",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n",
            f"Style: Default,{self.font_name},{self.font_size},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,{self.outline},{self.shadow},{self.alignment},{self.margin_l},{self.margin_r},{self.margin_v},1\n",
            f"Style: 需核查,{self.font_name},{self.font_size},&H0000FFFF,&H000000FF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,{max(self.outline, 3)},{max(self.shadow, 1)},{self.alignment},{self.margin_l},{self.margin_r},{self.margin_v},1\n",
            "\n",
            "[Events]\n",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n",
        ]
        body = [event.to_ass_line() + "\n" for event in self.events]
        result = "".join(header + body)
        logger.info(f"GeneratedAssDocument.dump: 输出生成完成，共 {len(result)} 字符，{len(body)} 条字幕")
        return result


def _ass_time_to_ms(value: str) -> int:
    hours, minutes, sec_part = value.split(":")
    seconds, centiseconds = sec_part.split(".")
    total_ms = (
        int(hours) * 3600 * 1000
        + int(minutes) * 60 * 1000
        + int(seconds) * 1000
        + int(centiseconds) * 10
    )
    return total_ms


def _ms_to_ass_time(value: int) -> str:
    total_cs = max(0, value // 10)
    hours, remain = divmod(total_cs, 360000)
    minutes, remain = divmod(remain, 6000)
    seconds, centiseconds = divmod(remain, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def _split_ass_event_payload(payload: str, field_count: int) -> List[str]:
    if field_count <= 1:
        return [payload]
    parts = payload.split(",", field_count - 1)
    if len(parts) != field_count:
        raise ValueError(f"Invalid ASS event payload: {payload}")
    return [part.strip() for part in parts[:-1]] + [parts[-1].rstrip("\r\n")]


def _build_ass_event(line_index: int, event_type: str, event_format: List[str], values: List[str], event_number: int) -> SubtitleEvent:
    field_map = dict(zip(event_format, values))
    return SubtitleEvent(
        event_id=f"event-{event_number}",
        index=event_number - 1,
        line_index=line_index,
        event_type=event_type,
        format_fields=event_format.copy(),
        field_values=values,
        start_ms=_ass_time_to_ms(field_map["Start"]),
        end_ms=_ass_time_to_ms(field_map["End"]),
        text=field_map.get("Text", ""),
        original_style=field_map.get("Style", ""),
    )


def parse_ass(path: str) -> AssDocument:
    logger.info(f"parse_ass: 开始解析ASS文件: {path}")
    raw_lines = Path(path).read_text(encoding="utf-8-sig").splitlines(keepends=True)
    event_format: List[str] = []
    event_indices: List[int] = []
    events: List[SubtitleEvent] = []
    in_events = False

    for line_index, raw_line in enumerate(raw_lines):
        stripped = raw_line.strip()
        if stripped.startswith("[Events]"):
            in_events = True
            continue
        if in_events and stripped.startswith("["):
            in_events = False
        if not in_events:
            continue
        if stripped.startswith("Format:"):
            event_format = [item.strip() for item in stripped[len("Format:"):].split(",")]
            continue
        if not stripped or stripped.startswith(";"):
            continue
        if stripped.startswith("Dialogue:") or stripped.startswith("Comment:"):
            if not event_format:
                raise ValueError("ASS events format is missing before dialogue lines")
            event_type, payload = stripped.split(":", 1)
            values = _split_ass_event_payload(payload.lstrip(), len(event_format))
            event = _build_ass_event(line_index, event_type, event_format, values, len(events) + 1)
            event_indices.append(line_index)
            events.append(event)

    logger.info(f"parse_ass: 解析完成，共 {len(events)} 条字幕")
    return AssDocument(raw_lines, event_indices, events)


def _srt_time_to_ms(value: str) -> int:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", value.strip())
    if match is None:
        raise ValueError(f"Invalid SRT time: {value}")
    hours, minutes, seconds, milliseconds = (int(part) for part in match.groups())
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds


def _normalize_srt_text(lines: list[str]) -> str:
    return r"\N".join(line.rstrip("\r\n") for line in lines)


def parse_srt(path: str) -> GeneratedAssDocument:
    logger.info(f"parse_srt: 开始解析SRT文件: {path}")
    raw = Path(path).read_text(encoding="utf-8-sig")
    blocks = re.split(r"\r?\n\r?\n+", raw.strip()) if raw.strip() else []
    logger.info(f"parse_srt: 原始块数: {len(blocks)}")
    events: List[SubtitleEvent] = []
    format_fields = ["Layer", "Start", "End", "Style", "Name", "MarginL", "MarginR", "MarginV", "Effect", "Text"]

    for index, block in enumerate(blocks, start=1):
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            logger.debug(f"parse_srt: 块 {index} 行数不足，跳过")
            continue
        time_line_index = 1 if re.fullmatch(r"\d+", lines[0].strip()) else 0
        if time_line_index >= len(lines):
            logger.debug(f"parse_srt: 块 {index} 时间行索引越界，跳过")
            continue
        time_line = lines[time_line_index].strip()
        if "-->" not in time_line:
            logger.debug(f"parse_srt: 块 {index} 无时间轴，跳过")
            continue
        start_text, end_text = [part.strip() for part in time_line.split("-->", 1)]
        text_lines = lines[time_line_index + 1 :]
        values = [
            "0",
            _ms_to_ass_time(_srt_time_to_ms(start_text)),
            _ms_to_ass_time(_srt_time_to_ms(end_text)),
            "Default",
            "",
            "0",
            "0",
            "0",
            "",
            _normalize_srt_text(text_lines),
        ]
        events.append(_build_ass_event(index - 1, "Dialogue", format_fields, values, index))

    logger.info(f"parse_srt: 解析完成，共 {len(events)} 条字幕")
    return GeneratedAssDocument(events)


def parse_subtitle_document(path: str) -> AssDocument:
    suffix = Path(path).suffix.lower()
    logger.info(f"parse_subtitle_document: 文件 {path}, 格式 {suffix}")
    if suffix == ".ass":
        return parse_ass(path)
    if suffix == ".srt":
        return parse_srt(path)
    raise ValueError("仅支持 ASS 或 SRT 输入")
