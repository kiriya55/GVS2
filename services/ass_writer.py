from __future__ import annotations

from pathlib import Path

from services.subtitle_parser import AssDocument, GeneratedAssDocument


REVIEW_STYLE_NAME = "需核查"
REVIEW_STYLE_LINE = "Style: 需核查,Arial,54,&H0000FFFF,&H000000FF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,3,1,2,48,48,42,1\n"


class AssWriter:
    def _inject_review_style(self, document: AssDocument) -> None:
        if isinstance(document, GeneratedAssDocument):
            return
        if any(line.strip().startswith(f"Style: {REVIEW_STYLE_NAME},") for line in document.lines):
            return
        format_index = next((idx for idx, line in enumerate(document.lines) if line.strip().startswith("Format: Name,")), None)
        if format_index is None:
            return
        insert_index = format_index + 1
        while insert_index < len(document.lines) and document.lines[insert_index].strip().startswith("Style:"):
            insert_index += 1
        document.lines.insert(insert_index, REVIEW_STYLE_LINE)
        document.event_indices = [index + 1 if index >= insert_index else index for index in document.event_indices]

    def write(self, document: AssDocument, output_path: str) -> str:
        self._inject_review_style(document)
        content = document.dump()
        Path(output_path).write_text(content, encoding="utf-8")
        return output_path
