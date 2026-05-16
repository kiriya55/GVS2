from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent, QColor, QIcon, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from models.job_result import EventJobResult, TextJobResult
from models.style_profile import StyleProfile, StyleSample
from models.subtitle_event import REVIEW_TEXT_OVERRIDE, SubtitleEvent
from providers.base import ProviderResponse, ProviderUsage
from providers.prompt_builder import (
    build_style_image_analysis_prompt,
    build_style_preview_prompt,
    build_style_refine_prompt,
    build_text_dry_run_prompt,
)
from services.image_preprocess import preprocess_for_llm
from services.media import extract_frame_ffmpeg, probe_video_duration_ffprobe
from services.ass_writer import AssWriter
from services.subtitle_parser import extract_ass_style_section, parse_ass_styles, parse_subtitle_document
from storage.job_store import JobStore
from storage.settings_store import SettingsStore
from ui.widgets import (
    CONFIG_PATH,
    IMAGE_FORMATS,
    LAYOUT_HINTS,
    STYLE_KEYWORDS_ZH,
    SUBTITLE_LANGUAGES,
    ApiSettingsDialog,
    PipelineWorker,
    ProviderJobWidget,
    RunPayload,
    StylePrepWorker,
    browse_row,
)


WINDOW_TITLE = "GVS2"
ICON_PATH = Path(__file__).resolve().parents[1] / "ico.ico"
STYLE_SAMPLE_DIR = Path(__file__).resolve().parents[1] / "style_samples"


class SubtitleRegionPreview(QLabel):
    region_changed = Signal(dict)

    HANDLE_SIZE = 10
    MIN_REGION_PERCENT = 1.0

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._frame_pixmap: QPixmap | None = None
        self._region = {"x": 0.0, "y": 66.0, "width": 100.0, "height": 34.0}
        self._drag_mode: str | None = None
        self._drag_start = QPointF()
        self._drag_origin = dict(self._region)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_frame(self, image: QImage | None) -> None:
        self._frame_pixmap = QPixmap.fromImage(image) if image is not None and not image.isNull() else None
        self.update()

    def set_region(self, region: dict[str, float], *, emit: bool = False) -> None:
        self._region = self._normalize_region(region)
        self.update()
        if emit:
            self.region_changed.emit(dict(self._region))

    def region(self) -> dict[str, float]:
        return dict(self._region)

    def _normalize_region(self, region: dict[str, float]) -> dict[str, float]:
        x = float(region.get("x", 0.0))
        y = float(region.get("y", 66.0))
        width = float(region.get("width", 100.0))
        height = float(region.get("height", 34.0))
        x = max(0.0, min(100.0 - self.MIN_REGION_PERCENT, x))
        y = max(0.0, min(100.0 - self.MIN_REGION_PERCENT, y))
        width = max(self.MIN_REGION_PERCENT, min(100.0 - x, width))
        height = max(self.MIN_REGION_PERCENT, min(100.0 - y, height))
        return {"x": x, "y": y, "width": width, "height": height}

    def _image_rect(self) -> QRectF:
        if self._frame_pixmap is None or self._frame_pixmap.isNull():
            return QRectF()
        size = self._frame_pixmap.size()
        scale = min(self.width() / size.width(), self.height() / size.height())
        draw_width = size.width() * scale
        draw_height = size.height() * scale
        return QRectF((self.width() - draw_width) / 2, (self.height() - draw_height) / 2, draw_width, draw_height)

    def _region_rect(self) -> QRectF:
        image_rect = self._image_rect()
        if image_rect.isNull():
            return QRectF()
        return QRectF(
            image_rect.left() + image_rect.width() * self._region["x"] / 100,
            image_rect.top() + image_rect.height() * self._region["y"] / 100,
            image_rect.width() * self._region["width"] / 100,
            image_rect.height() * self._region["height"] / 100,
        )

    def _point_to_percent(self, point: QPointF) -> tuple[float, float]:
        image_rect = self._image_rect()
        if image_rect.isNull():
            return 0.0, 0.0
        x = (point.x() - image_rect.left()) * 100 / image_rect.width()
        y = (point.y() - image_rect.top()) * 100 / image_rect.height()
        return max(0.0, min(100.0, x)), max(0.0, min(100.0, y))

    def _hit_mode(self, point: QPointF) -> str | None:
        rect = self._region_rect()
        if rect.isNull():
            return None
        tolerance = self.HANDLE_SIZE
        near_left = abs(point.x() - rect.left()) <= tolerance and rect.top() - tolerance <= point.y() <= rect.bottom() + tolerance
        near_right = abs(point.x() - rect.right()) <= tolerance and rect.top() - tolerance <= point.y() <= rect.bottom() + tolerance
        near_top = abs(point.y() - rect.top()) <= tolerance and rect.left() - tolerance <= point.x() <= rect.right() + tolerance
        near_bottom = abs(point.y() - rect.bottom()) <= tolerance and rect.left() - tolerance <= point.x() <= rect.right() + tolerance
        if near_left and near_top:
            return "top-left"
        if near_right and near_top:
            return "top-right"
        if near_left and near_bottom:
            return "bottom-left"
        if near_right and near_bottom:
            return "bottom-right"
        if near_left:
            return "left"
        if near_right:
            return "right"
        if near_top:
            return "top"
        if near_bottom:
            return "bottom"
        if rect.contains(point):
            return "move"
        if self._image_rect().contains(point):
            return "draw"
        return None

    def _cursor_for_mode(self, mode: str | None) -> Qt.CursorShape:
        return {
            "move": Qt.SizeAllCursor,
            "left": Qt.SizeHorCursor,
            "right": Qt.SizeHorCursor,
            "top": Qt.SizeVerCursor,
            "bottom": Qt.SizeVerCursor,
            "top-left": Qt.SizeFDiagCursor,
            "bottom-right": Qt.SizeFDiagCursor,
            "top-right": Qt.SizeBDiagCursor,
            "bottom-left": Qt.SizeBDiagCursor,
            "draw": Qt.CrossCursor,
        }.get(mode, Qt.ArrowCursor)

    def _resize_region(self, mode: str, x: float, y: float) -> dict[str, float]:
        left = self._drag_origin["x"]
        top = self._drag_origin["y"]
        right = left + self._drag_origin["width"]
        bottom = top + self._drag_origin["height"]
        if "left" in mode:
            left = min(x, right - self.MIN_REGION_PERCENT)
        if "right" in mode:
            right = max(x, left + self.MIN_REGION_PERCENT)
        if "top" in mode:
            top = min(y, bottom - self.MIN_REGION_PERCENT)
        if "bottom" in mode:
            bottom = max(y, top + self.MIN_REGION_PERCENT)
        return self._normalize_region({"x": left, "y": top, "width": right - left, "height": bottom - top})

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton or self._frame_pixmap is None:
            return
        point = QPointF(event.position())
        self._drag_mode = self._hit_mode(point)
        if self._drag_mode is None:
            return
        self._drag_start = point
        self._drag_origin = dict(self._region)
        if self._drag_mode == "draw":
            x, y = self._point_to_percent(point)
            self._region = self._normalize_region({"x": x, "y": y, "width": self.MIN_REGION_PERCENT, "height": self.MIN_REGION_PERCENT})
            self._drag_origin = dict(self._region)
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        point = QPointF(event.position())
        if self._drag_mode is None:
            self.setCursor(self._cursor_for_mode(self._hit_mode(point)))
            return
        x, y = self._point_to_percent(point)
        if self._drag_mode == "move":
            start_x, start_y = self._point_to_percent(self._drag_start)
            next_region = dict(self._drag_origin)
            next_region["x"] += x - start_x
            next_region["y"] += y - start_y
            self._region = self._normalize_region(next_region)
        elif self._drag_mode == "draw":
            start_x = self._drag_origin["x"]
            start_y = self._drag_origin["y"]
            self._region = self._normalize_region(
                {
                    "x": min(start_x, x),
                    "y": min(start_y, y),
                    "width": abs(x - start_x),
                    "height": abs(y - start_y),
                }
            )
        else:
            self._region = self._resize_region(self._drag_mode, x, y)
        self.region_changed.emit(dict(self._region))
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._drag_mode is not None:
            self._drag_mode = None
            self.region_changed.emit(dict(self._region))
            self.update()

    def paintEvent(self, event) -> None:
        if self._frame_pixmap is None or self._frame_pixmap.isNull():
            super().paintEvent(event)
            return
        painter = QPainter(self)
        image_rect = self._image_rect()
        painter.drawPixmap(image_rect, self._frame_pixmap, QRectF(self._frame_pixmap.rect()))
        region_rect = self._region_rect()
        painter.fillRect(image_rect, QColor(0, 0, 0, 90))
        painter.fillRect(region_rect, QColor(255, 255, 255, 38))
        painter.setPen(QPen(QColor(255, 70, 70), 2))
        painter.drawRect(region_rect)
        painter.setBrush(QColor(255, 70, 70))
        for point in (region_rect.topLeft(), region_rect.topRight(), region_rect.bottomLeft(), region_rect.bottomRight()):
            painter.drawRect(QRectF(point.x() - 4, point.y() - 4, 8, 8))
        painter.end()


class RunSummaryDialog(QDialog):
    def __init__(self, parent: QWidget, payload: RunPayload, event_count: int, retry_count: int, output_exists: bool, style_section_source: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("运行前确认")
        self.resize(720, 520)
        self.use_output_style_override = bool(payload.output_style_section_lines)

        layout = QVBoxLayout(self)
        title = QLabel("请确认本次运行计划")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(title)

        form = QFormLayout()
        tasks = []
        if payload.style_provider is not None:
            tasks.append(f"样式识别 / {payload.style_provider.model} / 并发 {payload.style_provider.concurrency}")
        if payload.text_provider is not None:
            tasks.append(f"文字提取 / {payload.text_provider.model} / 并发 {payload.text_provider.concurrency}")
        form.addRow("任务", QLabel("；".join(tasks) if tasks else "未启用"))
        form.addRow("处理范围", QLabel(f"复跑 {retry_count} 条事件" if retry_count else f"全量 {event_count} 条事件"))
        form.addRow("字幕区域", QLabel(self._format_region(payload.subtitle_region_rect)))
        form.addRow("字幕语言", QLabel(payload.subtitle_language))
        form.addRow("抽帧并发", QLabel(str(payload.frame_concurrency)))
        form.addRow("样式样本", QLabel("启用" if payload.include_samples else "未启用"))
        form.addRow("输出文件", QLabel(payload.ass_output_path + ("（将覆盖已有文件）" if output_exists else "")))
        layout.addLayout(form)

        self.override_check = QCheckBox("用导入 ASS 的 [V4+ Styles] 覆盖输出样式段")
        self.override_check.setChecked(self.use_output_style_override)
        self.override_check.setEnabled(bool(payload.output_style_section_lines))
        if style_section_source:
            self.override_check.setToolTip(style_section_source)
        self.override_check.toggled.connect(lambda checked: setattr(self, "use_output_style_override", checked))
        layout.addWidget(self.override_check)

        details = QPlainTextEdit()
        details.setReadOnly(True)
        details.setMaximumHeight(130)
        details.setPlainText(
            "\n".join(
                [
                    f"视频：{payload.video_path}",
                    f"输入字幕：{payload.ass_input_path}",
                    f"输出样式段来源：{style_section_source or '不覆盖'}",
                    f"样式图片：{payload.style_image_options.format_name} max_edge={payload.style_image_options.max_edge} q={payload.style_image_options.quality}",
                    f"文字图片：{payload.text_image_options.format_name} max_edge={payload.text_image_options.max_edge} q={payload.text_image_options.quality}",
                ]
            )
        )
        layout.addWidget(details)

        actions = QHBoxLayout()
        actions.addStretch(1)
        cancel_button = QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        start_button = QPushButton("开始运行")
        start_button.setDefault(True)
        start_button.clicked.connect(self.accept)
        actions.addWidget(cancel_button)
        actions.addWidget(start_button)
        layout.addLayout(actions)

    def _format_region(self, region: dict[str, float]) -> str:
        return f"X {region.get('x', 0):.1f}% / Y {region.get('y', 0):.1f}% / W {region.get('width', 0):.1f}% / H {region.get('height', 0):.1f}%"


class ResultReviewDialog(QDialog):
    def __init__(
        self,
        parent: QWidget,
        output_path: str,
        results: list[EventJobResult],
        style_profiles: list[StyleProfile],
        video_path: str = "",
        subtitle_region_rect: dict[str, float] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("结果审查工作台")
        self.resize(1280, 760)
        self.output_path = output_path
        self.video_path = video_path
        self.subtitle_region_rect = subtitle_region_rect or {"x": 0.0, "y": 66.0, "width": 100.0, "height": 34.0}
        self.results = results
        self.result_map = {result.event_id: result for result in results}
        self.style_profiles = style_profiles
        self.retry_failed_tasks_map: dict[str, list[str]] | None = None
        self.applied_style_event_ids: set[str] = set()
        self.document = parse_subtitle_document(output_path) if output_path and Path(output_path).exists() else None
        self.events = self.document.events if self.document is not None else []
        self.event_map = {event.event_id: event for event in self.events}
        self.row_event_ids: list[str] = []
        self.style_combos: dict[str, QComboBox] = {}
        self.preview_cache: dict[str, QImage | None] = {}

        layout = QVBoxLayout(self)
        summary = QLabel(self._summary_text())
        summary.setStyleSheet("font-size: 15px; font-weight: 600;")
        layout.addWidget(summary)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("筛选"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["需处理", "需核查样式", "文字需核查", "文字失败", "全部"])
        self.filter_combo.currentIndexChanged.connect(self._populate_table)
        toolbar.addWidget(self.filter_combo)
        toolbar.addStretch(1)
        apply_button = QPushButton("应用样式修正")
        apply_button.clicked.connect(self._apply_style_fixes)
        retry_selected_button = QPushButton("复跑选中文字失败")
        retry_selected_button.clicked.connect(self._retry_selected_text_failures)
        retry_all_button = QPushButton("复跑全部文字失败")
        retry_all_button.clicked.connect(self._retry_all_text_failures)
        toolbar.addWidget(apply_button)
        toolbar.addWidget(retry_selected_button)
        toolbar.addWidget(retry_all_button)
        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Horizontal)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["事件", "时间", "状态", "当前样式", "人工输出样式", "字幕文本", "错误"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._refresh_preview)
        left_layout.addWidget(self.table, 1)
        splitter.addWidget(left_panel)

        preview_panel = QGroupBox("选中事件预览")
        preview_layout = QVBoxLayout(preview_panel)
        self.preview_title = QLabel("选择一条记录查看画面和输出内容")
        self.preview_title.setStyleSheet("font-size: 14px; font-weight: 600;")
        preview_layout.addWidget(self.preview_title)
        self.preview_frame = SubtitleRegionPreview("选择一条记录后预览视频帧")
        self.preview_frame.setMinimumSize(360, 240)
        self.preview_frame.set_region(self.subtitle_region_rect)
        preview_layout.addWidget(self.preview_frame, 1)
        self.preview_details = QPlainTextEdit()
        self.preview_details.setReadOnly(True)
        self.preview_details.setMaximumHeight(180)
        preview_layout.addWidget(self.preview_details)
        splitter.addWidget(preview_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        actions = QHBoxLayout()
        actions.addStretch(1)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        actions.addWidget(close_button)
        layout.addLayout(actions)
        self._populate_table()
        if self.table.rowCount() > 0:
            self.table.selectRow(0)

    def _summary_text(self) -> str:
        review_count = sum(1 for result in self.results if result.style_result and result.style_result.review_required)
        text_review_count = sum(1 for result in self.results if result.text_result and result.text_result.review_required)
        text_failed_count = sum(1 for result in self.results if "text" in result.failed_tasks)
        failed_count = sum(1 for result in self.results if result.error_messages)
        return f"需核查样式 {review_count} 条 / 文字需核查 {text_review_count} 条 / 文字失败 {text_failed_count} 条 / 总失败 {failed_count} 条 / 输出 {self.output_path}"

    def _event_time(self, event: SubtitleEvent | None) -> str:
        if event is None:
            return ""
        return f"{event.start_ms / 1000:.2f}s - {event.end_ms / 1000:.2f}s"

    def _needs_review(self, event_id: str) -> bool:
        result = self.result_map.get(event_id)
        event = self.event_map.get(event_id)
        if event_id in self.applied_style_event_ids:
            return False
        if event is not None:
            return event.text.startswith(REVIEW_TEXT_OVERRIDE)
        return bool(result and result.style_result and result.style_result.review_required)

    def _is_text_failed(self, event_id: str) -> bool:
        result = self.result_map.get(event_id)
        return bool(result and "text" in result.failed_tasks)

    def _is_text_review_required(self, event_id: str) -> bool:
        result = self.result_map.get(event_id)
        return bool(result and result.text_result and result.text_result.review_required)

    def _status_text(self, event_id: str) -> str:
        parts = []
        if event_id in self.applied_style_event_ids:
            parts.append("样式已修正")
        elif self._needs_review(event_id):
            parts.append("需核查样式")
        if self._is_text_review_required(event_id):
            parts.append("文字需核查")
        if self._is_text_failed(event_id):
            parts.append("文字失败")
        result = self.result_map.get(event_id)
        if result and "style" in result.failed_tasks:
            parts.append("样式失败")
        return " / ".join(parts) or "已处理"

    def _include_event(self, event_id: str) -> bool:
        mode = self.filter_combo.currentText()
        if mode == "全部":
            return True
        if mode == "需核查样式":
            return self._needs_review(event_id)
        if mode == "文字需核查":
            return self._is_text_review_required(event_id)
        if mode == "文字失败":
            return self._is_text_failed(event_id)
        return self._needs_review(event_id) or self._is_text_review_required(event_id) or self._is_text_failed(event_id) or bool(self.result_map.get(event_id, EventJobResult(event_id)).error_messages)

    def _set_item(self, row: int, column: int, value: str) -> None:
        item = QTableWidgetItem(value)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(row, column, item)

    def _selected_event_id(self) -> str | None:
        selected_rows = sorted({index.row() for index in self.table.selectedIndexes()})
        if not selected_rows:
            return None
        row = selected_rows[0]
        if row >= len(self.row_event_ids):
            return None
        return self.row_event_ids[row]

    def _preview_image_for_event(self, event: SubtitleEvent) -> QImage | None:
        cached = self.preview_cache.get(event.event_id)
        if event.event_id in self.preview_cache:
            return cached
        if not self.video_path:
            self.preview_cache[event.event_id] = None
            return None
        frame_bytes = extract_frame_ffmpeg(self.video_path, event.midpoint_ms / 1000, timeout_sec=8)
        if not frame_bytes:
            self.preview_cache[event.event_id] = None
            return None
        image = QImage()
        if not image.loadFromData(frame_bytes):
            self.preview_cache[event.event_id] = None
            return None
        self.preview_cache[event.event_id] = image
        return image

    def _refresh_preview(self) -> None:
        event_id = self._selected_event_id()
        event = self.event_map.get(event_id or "")
        result = self.result_map.get(event_id or "")
        if event is None:
            self.preview_title.setText("选择一条记录查看画面和输出内容")
            self.preview_frame.setText("当前记录缺少字幕事件")
            self.preview_frame.set_frame(None)
            self.preview_details.setPlainText("")
            return

        self.preview_title.setText(f"{event.event_id} / {self._event_time(event)} / {self._status_text(event.event_id)}")
        self.preview_frame.set_region(self.subtitle_region_rect)
        image = self._preview_image_for_event(event)
        if image is None:
            self.preview_frame.setText("未能读取视频帧，请检查视频路径或 ffmpeg")
            self.preview_frame.set_frame(None)
        else:
            self.preview_frame.set_frame(image)

        text = event.text
        if text.startswith(REVIEW_TEXT_OVERRIDE):
            text = text[len(REVIEW_TEXT_OVERRIDE):]
        details = [
            f"当前样式：{event.style}",
            f"原始样式：{event.original_style}",
            f"字幕文本：{text}",
        ]
        if result and result.text_result and result.text_result.review_reasons:
            details.append("文字核查：" + "；".join(result.text_result.review_reasons))
        if result and result.error_messages:
            details.append("错误：" + " | ".join(result.error_messages))
        self.preview_details.setPlainText("\n".join(details))

    def _style_choices(self, current_style: str) -> list[str]:
        choices = []
        for profile in self.style_profiles:
            if profile.ass_style_name and profile.ass_style_name not in choices:
                choices.append(profile.ass_style_name)
        if current_style and current_style not in choices:
            choices.insert(0, current_style)
        return choices or ([current_style] if current_style else ["Default"])

    def _populate_table(self) -> None:
        self.table.setRowCount(0)
        self.row_event_ids = []
        self.style_combos = {}
        ordered_ids = [event.event_id for event in self.events] or [result.event_id for result in self.results]
        for event_id in ordered_ids:
            if not self._include_event(event_id):
                continue
            event = self.event_map.get(event_id)
            result = self.result_map.get(event_id)
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.row_event_ids.append(event_id)
            current_style = event.style if event is not None else ""
            text = event.text if event is not None else ""
            if text.startswith(REVIEW_TEXT_OVERRIDE):
                text = text[len(REVIEW_TEXT_OVERRIDE):]
            self._set_item(row, 0, event_id)
            self._set_item(row, 1, self._event_time(event))
            self._set_item(row, 2, self._status_text(event_id))
            self._set_item(row, 3, current_style)
            combo = QComboBox()
            combo.addItems(self._style_choices(current_style))
            combo.setCurrentText(current_style)
            combo.setEnabled(self._needs_review(event_id))
            self.style_combos[event_id] = combo
            self.table.setCellWidget(row, 4, combo)
            self._set_item(row, 5, text)
            self._set_item(row, 6, " | ".join(result.error_messages) if result else "")
        if self.table.rowCount() > 0:
            self.table.selectRow(0)
        else:
            self._refresh_preview()

    def _apply_style_fixes(self) -> None:
        if self.document is None:
            QMessageBox.warning(self, WINDOW_TITLE, "未找到可写回的输出 ASS。")
            return
        changed = 0
        for event_id, combo in self.style_combos.items():
            if not self._needs_review(event_id):
                continue
            event = self.event_map.get(event_id)
            if event is None:
                continue
            selected_style = combo.currentText().strip()
            if selected_style:
                event.style = selected_style
            event.clear_review_text_marker()
            self.applied_style_event_ids.add(event_id)
            changed += 1
        if changed == 0:
            QMessageBox.information(self, WINDOW_TITLE, "当前筛选结果中没有可应用的样式核查项。")
            return
        AssWriter().write(self.document, self.output_path)
        self._populate_table()
        QMessageBox.information(self, WINDOW_TITLE, f"已写回 {changed} 条样式修正。")

    def _text_failed_ids_from_selection(self) -> list[str]:
        rows = sorted({index.row() for index in self.table.selectedIndexes()})
        event_ids = [self.row_event_ids[row] for row in rows if row < len(self.row_event_ids)]
        return [event_id for event_id in event_ids if self._is_text_failed(event_id)]

    def _retry_selected_text_failures(self) -> None:
        event_ids = self._text_failed_ids_from_selection()
        if not event_ids:
            QMessageBox.warning(self, WINDOW_TITLE, "请先选择至少一条文字失败事件。")
            return
        self.retry_failed_tasks_map = {event_id: ["text"] for event_id in event_ids}
        self.accept()

    def _retry_all_text_failures(self) -> None:
        event_ids = [result.event_id for result in self.results if "text" in result.failed_tasks]
        if not event_ids:
            QMessageBox.information(self, WINDOW_TITLE, "没有文字失败事件需要复跑。")
            return
        self.retry_failed_tasks_map = {event_id: ["text"] for event_id in event_ids}
        self.accept()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(980, 820)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.settings_store = SettingsStore(str(CONFIG_PATH))
        self.job_store = JobStore()
        self.worker: PipelineWorker | None = None
        self.style_prep_worker: StylePrepWorker | None = None
        self.style_prep_mode = ""
        self.preview_frame_bytes: bytes | None = None
        self.loaded_events = []
        self.current_event_index = -1
        self.pending_retry_event_ids: set[str] | None = None
        self.pending_retry_failed_tasks: dict[str, list[str]] | None = None
        self.imported_ass_style_path = ""
        self.imported_ass_style_section: list[str] = []
        self.last_run_results: list[EventJobResult] = []
        self.last_run_style_profiles: list[StyleProfile] = []
        self.api_settings_dialog = ApiSettingsDialog(self)
        self.style_job_widget = self.api_settings_dialog.style_job_widget
        self.text_job_widget = self.api_settings_dialog.text_job_widget
        self._style_job_history: list[dict] = []
        self._text_job_history: list[dict] = []
        self._video_duration: float | None = None
        self._preview_debounce = QTimer(self)
        self._preview_debounce.setSingleShot(True)
        self._preview_debounce.setInterval(200)
        self._preview_debounce.timeout.connect(self._refresh_video_preview)
        self._setup_ui()
        self._load_settings()
        self._run_startup_checks()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        api_menu = self.menuBar().addMenu("设置")
        api_settings_action = QAction("API 设置", self)
        api_settings_action.triggered.connect(self._open_api_settings)
        api_menu.addAction(api_settings_action)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        self._setup_files_group(left_layout)
        self._setup_style_prepare_group(left_layout)
        self._setup_styles_group(left_layout)
        self._setup_preview_group(right_layout)
        self._setup_region_group(right_layout)
        self._setup_actions(right_layout)
        self._setup_progress_group(right_layout)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        right_layout.addWidget(self.log_edit, 1)

    def _setup_files_group(self, parent_layout: QVBoxLayout) -> None:
        files_box = QGroupBox("输入输出")
        files_form = QFormLayout()
        self.video_edit = QLineEdit()
        self.ass_input_edit = QLineEdit()
        self.ass_output_edit = QLineEdit()
        files_form.addRow("视频", self._with_browse(self.video_edit, self._browse_video))
        files_form.addRow("ASS 输入", self._with_browse(self.ass_input_edit, self._browse_ass_input))
        files_form.addRow("ASS 输出", self._with_browse(self.ass_output_edit, self._browse_ass_output, save=True))
        files_box.setLayout(files_form)
        parent_layout.addWidget(files_box)

    def _setup_style_prepare_group(self, parent_layout: QVBoxLayout) -> None:
        style_prepare_box = QGroupBox("样式整理")
        style_prepare_layout = QVBoxLayout(style_prepare_box)
        self.style_description_edit = QPlainTextEdit()
        self.style_description_edit.setMaximumHeight(70)
        self.style_description_edit.setPlaceholderText("例如：红色填充，黑色描边，双行字幕")
        self.refine_style_button = QPushButton("自然语言转紧凑描述")
        self.refine_style_button.clicked.connect(self._refine_style_description)
        self.analyze_style_image_button = QPushButton("从预览图生成样式描述")
        self.analyze_style_image_button.clicked.connect(self._analyze_preview_style)
        self.preview_style_prompt_button = QPushButton("点击预览识别提示")
        self.preview_style_prompt_button.clicked.connect(self._show_style_prompt_preview_dialog)
        self.style_compact_edit = QLineEdit()
        self.style_compact_edit.setPlaceholderText("例如：yellow_text; black_outline; 2_lines; bold")
        style_prepare_layout.addWidget(QLabel("自然语言描述"))
        style_prepare_layout.addWidget(self.style_description_edit)
        style_prepare_actions = QHBoxLayout()
        style_prepare_actions.addWidget(self.refine_style_button)
        style_prepare_actions.addWidget(self.analyze_style_image_button)
        style_prepare_actions.addWidget(self.preview_style_prompt_button)
        style_prepare_layout.addLayout(style_prepare_actions)
        style_prepare_layout.addWidget(QLabel("紧凑样式描述"))
        style_prepare_layout.addWidget(self.style_compact_edit)
        parent_layout.addWidget(style_prepare_box)

    def _setup_styles_group(self, parent_layout: QVBoxLayout) -> None:
        styles_box = QGroupBox("锁定样式")
        styles_layout = QVBoxLayout(styles_box)
        styles_layout.addWidget(QLabel("每行一个样式；布局提示可选：任意 / 单行 / 双行。"))
        self.styles_table = QTableWidget(0, 5)
        self.styles_table.setHorizontalHeaderLabels(["样式ID", "显示名称", "ASS样式名", "特征描述", "布局提示"])
        self.styles_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        styles_layout.addWidget(self.styles_table)
        styles_actions = QHBoxLayout()
        self.add_style_button = QPushButton("新增样式")
        self.add_style_button.clicked.connect(self._add_style_row)
        self.remove_style_button = QPushButton("删除选中样式")
        self.remove_style_button.clicked.connect(self._remove_selected_style_rows)
        self.renumber_style_button = QPushButton("自动编号")
        self.renumber_style_button.clicked.connect(self._renumber_style_rows)
        self.import_styles_button = QPushButton("导入模板 JSON")
        self.import_styles_button.clicked.connect(self._import_style_template)
        self.export_styles_button = QPushButton("导出模板 JSON")
        self.export_styles_button.clicked.connect(self._export_style_template)
        self.import_ass_styles_button = QPushButton("从ASS导入样式")
        self.import_ass_styles_button.setToolTip("从已有 ASS 文件的 [V4+ Styles] 段导入样式定义")
        self.import_ass_styles_button.clicked.connect(self._import_ass_styles)
        self.append_compact_style_button = QPushButton("追加当前紧凑描述")
        self.append_compact_style_button.clicked.connect(self._append_compact_style_row)
        self.lock_preview_style_button = QPushButton("从当前预览锁定样式")
        self.lock_preview_style_button.clicked.connect(self._lock_style_from_preview)
        styles_actions.addWidget(self.add_style_button)
        styles_actions.addWidget(self.remove_style_button)
        styles_actions.addWidget(self.renumber_style_button)
        styles_actions.addWidget(self.import_styles_button)
        styles_actions.addWidget(self.export_styles_button)
        styles_actions.addWidget(self.import_ass_styles_button)
        styles_actions.addWidget(self.append_compact_style_button)
        styles_actions.addWidget(self.lock_preview_style_button)
        styles_layout.addLayout(styles_actions)
        self.include_samples_check = QCheckBox("运行时使用已保存的样本图作为视觉参考（few-shot）")
        self.include_samples_check.setToolTip("勾选后，已锁定样式的样本图会作为视觉示例发送给 LLM，提高识别准确率，但会增加 token 消耗")
        styles_layout.addWidget(self.include_samples_check)
        self.override_output_styles_check = QCheckBox("输出时用导入 ASS 样式覆盖样式段")
        self.override_output_styles_check.setEnabled(False)
        self.override_output_styles_check.setToolTip("从 ASS 导入样式后可用；运行前会确认是否用导入文件的 [V4+ Styles] 覆盖输出 ASS 的样式段")
        styles_layout.addWidget(self.override_output_styles_check)
        parent_layout.addWidget(styles_box, 1)

    def _setup_preview_group(self, parent_layout: QVBoxLayout) -> None:
        preview_box = QGroupBox("视频预览")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_label = SubtitleRegionPreview("选择视频后可预览字幕区域")
        self.preview_label.region_changed.connect(self._on_region_changed)
        self.preview_time_slider = QSlider(Qt.Horizontal)
        self.preview_time_slider.setRange(0, 1000)
        self.preview_time_slider.setValue(500)
        self.preview_time_slider.valueChanged.connect(self._request_preview_refresh)
        self.preview_time_hint = QLabel("预览时间：50%")
        preview_nav = QHBoxLayout()
        self.prev_subtitle_button = QPushButton("上一个")
        self.prev_subtitle_button.clicked.connect(lambda: self._jump_subtitle_event(-1))
        self.next_subtitle_button = QPushButton("下一个")
        self.next_subtitle_button.clicked.connect(lambda: self._jump_subtitle_event(1))
        self.subtitle_jump_combo = QComboBox()
        self.subtitle_jump_combo.setMinimumWidth(200)
        self.subtitle_jump_combo.setPlaceholderText("跳到指定字幕…")
        self.subtitle_jump_combo.currentIndexChanged.connect(self._on_jump_combo_selected)
        self.preview_processed_button = QPushButton("预览处理图")
        self.preview_processed_button.clicked.connect(self._show_processed_image_preview)
        self.dry_run_text_button = QPushButton("文字提取试运行")
        self.dry_run_text_button.clicked.connect(self._dry_run_text_recognition)
        preview_nav.addWidget(self.prev_subtitle_button)
        preview_nav.addWidget(self.subtitle_jump_combo, 1)
        preview_nav.addWidget(self.next_subtitle_button)
        preview_nav.addWidget(self.preview_processed_button)
        preview_nav.addWidget(self.dry_run_text_button)
        preview_layout.addWidget(self.preview_label, 1)
        preview_layout.addWidget(self.preview_time_hint)
        preview_layout.addWidget(self.preview_time_slider)
        preview_layout.addLayout(preview_nav)
        parent_layout.addWidget(preview_box, 1)

    def _setup_region_group(self, parent_layout: QVBoxLayout) -> None:
        region_box = QGroupBox("字幕区域")
        region_form = QFormLayout()
        self.subtitle_region_info_edit = QLineEdit()
        self.subtitle_region_info_edit.setReadOnly(True)
        self.subtitle_language_combo = QComboBox()
        for value, label in SUBTITLE_LANGUAGES:
            self.subtitle_language_combo.addItem(label, value)
        self.frame_concurrency_spin = QSpinBox()
        self.frame_concurrency_spin.setRange(1, 12)
        self.frame_concurrency_spin.setValue(5)
        self.frame_concurrency_spin.setToolTip("控制预提取视频帧时同时运行的 ffmpeg 进程数；较高数值更快但更吃 CPU、磁盘和内存")
        region_form.addRow("区域百分比", self.subtitle_region_info_edit)
        region_form.addRow("字幕语言", self.subtitle_language_combo)
        region_form.addRow("抽帧并发数", self.frame_concurrency_spin)
        region_box.setLayout(region_form)
        parent_layout.addWidget(region_box)

    def _setup_actions(self, parent_layout: QVBoxLayout) -> None:
        actions_layout = QHBoxLayout()
        self.refresh_preview_button = QPushButton("刷新预览")
        self.refresh_preview_button.clicked.connect(self._refresh_video_preview)
        self.run_button = QPushButton("开始处理")
        self.run_button.clicked.connect(self._start_run)
        self.save_button = QPushButton("保存设置")
        self.save_button.clicked.connect(self._save_settings)
        actions_layout.addWidget(self.refresh_preview_button)
        actions_layout.addWidget(self.run_button)
        actions_layout.addWidget(self.save_button)
        parent_layout.addLayout(actions_layout)

    def _setup_progress_group(self, parent_layout: QVBoxLayout) -> None:
        progress_box = QGroupBox("运行进度")
        progress_layout = QVBoxLayout(progress_box)
        self.run_progress_label = QLabel("尚未开始")
        self.run_progress_bar = QProgressBar()
        self.run_progress_bar.setRange(0, 100)
        self.run_progress_bar.setValue(0)
        progress_actions = QHBoxLayout()
        self.retry_failed_button = QPushButton("复跑失败项")
        self.retry_failed_button.setEnabled(False)
        self.retry_failed_button.clicked.connect(self._retry_failed_events)
        self.review_results_button = QPushButton("审查结果")
        self.review_results_button.setEnabled(False)
        self.review_results_button.clicked.connect(self._open_result_review)
        progress_actions.addStretch(1)
        progress_actions.addWidget(self.review_results_button)
        progress_actions.addWidget(self.retry_failed_button)
        progress_layout.addWidget(self.run_progress_label)
        progress_layout.addWidget(self.run_progress_bar)
        progress_layout.addLayout(progress_actions)
        parent_layout.addWidget(progress_box)

    # ------------------------------------------------------------------
    # Menu & dialogs
    # ------------------------------------------------------------------

    def _enforce_text_ocr_budget(self, widget: ProviderJobWidget) -> None:
        widget.max_tokens_spin.setValue(max(widget.max_tokens_spin.value(), 512))
        widget.max_edge_spin.setValue(max(widget.max_edge_spin.value(), 1280))

    def _open_api_settings(self) -> None:
        self.api_settings_dialog.load_settings(self.style_job_widget.dump_settings(), self.text_job_widget.dump_settings())
        self._enforce_text_ocr_budget(self.api_settings_dialog.text_job_widget)
        self.api_settings_dialog.load_history(self._style_job_history, self._text_job_history)
        if self.api_settings_dialog.exec() == QDialog.Accepted:
            style_job, text_job = self.api_settings_dialog.dump_settings()
            self.style_job_widget.load_settings(style_job)
            self.text_job_widget.load_settings(text_job)
            self._enforce_text_ocr_budget(self.text_job_widget)
            self._style_job_history, self._text_job_history = self.api_settings_dialog.dump_history()
            self._save_settings()
            self._run_startup_checks()

    def _with_browse(self, edit: QLineEdit, handler, *, save: bool = False) -> QWidget:
        return browse_row(edit, handler)

    # ------------------------------------------------------------------
    # File browsing
    # ------------------------------------------------------------------

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.mkv *.avi *.mov);;All files (*)")
        if path:
            self.video_edit.setText(path)
            self._video_duration = None
            self._refresh_video_preview()

    def _browse_ass_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择字幕文件", "", "Subtitle files (*.ass *.srt);;ASS files (*.ass);;SRT files (*.srt);;All files (*)")
        if path:
            self.ass_input_edit.setText(path)
            if not self.ass_output_edit.text().strip():
                output_path = str(Path(path).with_name(Path(path).stem + "_gvs2.ass"))
                self.ass_output_edit.setText(output_path)
            try:
                self._load_subtitle_events()
                self._append_log(f"已加载字幕事件：{len(self.loaded_events)}")
            except Exception as exc:
                self.loaded_events = []
                self.current_event_index = -1
                QMessageBox.critical(self, WINDOW_TITLE, f"字幕加载失败：{exc}")

    def _browse_ass_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "保存 ASS", self.ass_output_edit.text().strip() or "output.ass", "ASS files (*.ass)")
        if path:
            self.ass_output_edit.setText(path)

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _get_video_duration(self) -> float | None:
        if self._video_duration is not None:
            return self._video_duration
        video_path = self.video_edit.text().strip()
        if not video_path:
            return None
        self._video_duration = probe_video_duration_ffprobe(video_path)
        return self._video_duration

    def _request_preview_refresh(self) -> None:
        self._preview_debounce.start()

    def _ensure_preview_frame(self) -> bytes | None:
        if self.preview_frame_bytes:
            return self.preview_frame_bytes
        self._refresh_video_preview()
        if self.preview_frame_bytes:
            return self.preview_frame_bytes
        QMessageBox.warning(self, WINDOW_TITLE, "请先生成视频预览图")
        return None

    def _preprocess_preview(self, image_options: ImageEncodingOptions) -> tuple[str, str]:
        return preprocess_for_llm(
            self.preview_frame_bytes,
            image_options,
            region_rect=self._subtitle_region_rect(),
        )

    def _subtitle_region_rect(self) -> dict[str, float]:
        return self.preview_label.region()

    def _subtitle_region_start_end(self) -> tuple[int, int]:
        region = self._subtitle_region_rect()
        start = int(round(region["y"]))
        end = int(round(region["y"] + region["height"]))
        return max(0, min(100, start)), max(0, min(100, end))

    def _format_region_info(self, region: dict[str, float]) -> str:
        return f"X {region['x']:.1f}%  Y {region['y']:.1f}%  W {region['width']:.1f}%  H {region['height']:.1f}%"

    def _on_region_changed(self, region: dict) -> None:
        self.subtitle_region_info_edit.setText(self._format_region_info(region))

    def _selected_preview_time_sec(self) -> float:
        duration = self._get_video_duration()
        ratio = self.preview_time_slider.value() / 1000
        self.preview_time_hint.setText(f"预览时间：{ratio:.0%}")
        if duration and duration > 0:
            return duration * ratio
        return 0.0

    def _set_preview_to_time_ms(self, time_ms: int) -> None:
        duration = self._get_video_duration()
        if not duration or duration <= 0:
            self._refresh_video_preview()
            return
        ratio = max(0.0, min(1.0, (time_ms / 1000) / duration))
        self.preview_time_slider.setValue(int(ratio * 1000))

    def _jump_subtitle_event(self, step: int) -> None:
        if not self.loaded_events:
            QMessageBox.warning(self, WINDOW_TITLE, "请先加载 ASS 或 SRT 字幕文件，再使用字幕时点导航。")
            return
        next_index = self.current_event_index + step
        next_index = max(0, min(len(self.loaded_events) - 1, next_index))
        self.current_event_index = next_index
        self.subtitle_jump_combo.blockSignals(True)
        self.subtitle_jump_combo.setCurrentIndex(next_index)
        self.subtitle_jump_combo.blockSignals(False)
        event = self.loaded_events[next_index]
        self._set_preview_to_time_ms(event.midpoint_ms)
        self._append_log(f"跳转到字幕事件 {next_index + 1}/{len(self.loaded_events)}：{event.start_ms}-{event.end_ms}ms")

    def _format_event_label(self, event: SubtitleEvent, index: int) -> str:
        start_sec = event.start_ms / 1000
        end_sec = event.end_ms / 1000
        text_preview = event.text.replace("\\N", " ").strip()[:20]
        return f"#{index + 1} [{start_sec:.1f}s-{end_sec:.1f}s] {text_preview}"

    def _populate_subtitle_jump_combo(self) -> None:
        self.subtitle_jump_combo.blockSignals(True)
        self.subtitle_jump_combo.clear()
        for i, event in enumerate(self.loaded_events):
            self.subtitle_jump_combo.addItem(self._format_event_label(event, i))
        if self.current_event_index >= 0:
            self.subtitle_jump_combo.setCurrentIndex(self.current_event_index)
        self.subtitle_jump_combo.blockSignals(False)

    def _on_jump_combo_selected(self, index: int) -> None:
        if index < 0 or not self.loaded_events:
            return
        self.current_event_index = index
        event = self.loaded_events[index]
        self._set_preview_to_time_ms(event.midpoint_ms)

    def _show_style_prompt_preview_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("样式识别 prompt 预览")
        dialog.resize(720, 520)
        layout = QVBoxLayout(dialog)
        editor = QTextEdit()
        editor.setReadOnly(True)
        try:
            profiles = self._parse_style_profiles()
        except Exception:
            profiles = []
        preview = build_style_preview_prompt(profiles)
        compact = self.style_compact_edit.text().strip()
        if compact:
            preview = f"Current compact style:\n{compact}\n\n{preview}"
        editor.setPlainText(preview)
        layout.addWidget(editor)
        button = QPushButton("关闭")
        button.clicked.connect(dialog.accept)
        layout.addWidget(button)
        dialog.exec()

    def _show_processed_image_preview(self) -> None:
        if not self._ensure_preview_frame():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("当前字幕事件处理图预览")
        dialog.resize(900, 720)
        layout = QHBoxLayout(dialog)
        previews = [
            ("style_job", self.style_job_widget.build_image_options()),
            ("text_job", self.text_job_widget.build_image_options()),
        ]
        for title, image_options in previews:
            _, image_b64 = self._preprocess_preview(image_options)
            image = QImage.fromData(base64.b64decode(image_b64))
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.addWidget(QLabel(f"{title} | {image_options.format_name} | max_edge={image_options.max_edge} | q={image_options.quality}"))
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setMinimumSize(320, 180)
            image_label.setPixmap(QPixmap.fromImage(image).scaled(420, 560, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            panel_layout.addWidget(image_label, 1)
            layout.addWidget(panel, 1)
        dialog.exec()

    def _render_preview_pixmap(self, frame_bytes: bytes) -> QPixmap | None:
        image = QImage.fromData(frame_bytes)
        if image.isNull():
            return None
        pixmap = QPixmap.fromImage(image)
        painter = QPainter(pixmap)
        painter.fillRect(0, 0, pixmap.width(), pixmap.height(), QColor(0, 0, 0, 80))
        region = self._subtitle_region_rect()
        left = int(pixmap.width() * region["x"] / 100)
        top = int(pixmap.height() * region["y"] / 100)
        right = int(pixmap.width() * (region["x"] + region["width"]) / 100)
        bottom = int(pixmap.height() * (region["y"] + region["height"]) / 100)
        painter.fillRect(left, top, max(1, right - left), max(1, bottom - top), QColor(255, 255, 255, 40))
        painter.setPen(QPen(QColor(255, 80, 80), 3))
        painter.drawRect(left, top, max(1, right - left), max(1, bottom - top))
        painter.end()
        return pixmap

    def _refresh_video_preview(self) -> None:
        video_path = self.video_edit.text().strip()
        self.preview_time_hint.setText(f"预览时间：{self.preview_time_slider.value() / 1000:.0%}")
        self.preview_frame_bytes = None
        if not video_path:
            self.preview_label.setText("选择视频后可预览字幕区域")
            self.preview_label.set_frame(None)
            return
        frame_bytes = extract_frame_ffmpeg(video_path, self._selected_preview_time_sec())
        if not frame_bytes:
            self.preview_label.setText("预览截图失败，请检查 ffmpeg 和视频路径")
            self.preview_label.set_frame(None)
            return
        self.preview_frame_bytes = frame_bytes
        image = QImage.fromData(frame_bytes)
        if image.isNull():
            self.preview_label.setText("预览图像解码失败")
            self.preview_label.set_frame(None)
            return
        self.preview_label.set_frame(image)

    # ------------------------------------------------------------------
    # Style preparation
    # ------------------------------------------------------------------

    def _has_style_keywords(self, text: str) -> bool:
        return any(keyword in text for keyword in STYLE_KEYWORDS_ZH)

    def _start_style_prep(self, prompt: str, images: list[tuple[str, str]], mode: str) -> None:
        if self.style_prep_worker is not None:
            return
        try:
            provider_config = self.style_job_widget.build_provider_config()
        except Exception as exc:
            QMessageBox.critical(self, WINDOW_TITLE, f"样式整理无法启动：{exc}")
            return
        if provider_config is None:
            QMessageBox.critical(self, WINDOW_TITLE, "请先启用并配置样式识别任务")
            return
        self.style_prep_mode = mode
        self.refine_style_button.setEnabled(False)
        self.analyze_style_image_button.setEnabled(False)
        self.style_prep_worker = StylePrepWorker(provider_config, prompt, images)
        self.style_prep_worker.finished_text.connect(self._on_style_prep_finished)
        self.style_prep_worker.failed.connect(self._on_style_prep_failed)
        self.style_prep_worker.finished.connect(self._clear_style_prep_worker)
        self.style_prep_worker.start()
        self._append_log(f"开始样式整理：{mode}")

    def _normalize_style_compact_text(self, raw: str) -> str:
        line = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        line = line.strip("` ")
        return line.replace("，", "; ").replace(",", "; ")

    def _refine_style_description(self) -> None:
        raw = self.style_description_edit.toPlainText().strip()
        if not raw:
            QMessageBox.critical(self, WINDOW_TITLE, "请先输入自然语言样式描述")
            return
        if not self._has_style_keywords(raw):
            QMessageBox.warning(
                self,
                WINDOW_TITLE,
                "这段描述里没有明显的字幕样式关键词，暂不发送给模型。请补充颜色、描边、阴影、单行/双行、字体等字幕特征后再试。",
            )
            return
        self._start_style_prep(build_style_refine_prompt(raw), [], "text")

    def _analyze_preview_style(self) -> None:
        if not self._ensure_preview_frame():
            return
        image_options = self.style_job_widget.build_image_options()
        image = self._preprocess_preview(image_options)
        self._start_style_prep(build_style_image_analysis_prompt(), [image], "image")

    def _dry_run_text_recognition(self) -> None:
        if self.style_prep_worker is not None:
            return
        if not self._ensure_preview_frame():
            return
        try:
            provider_config = self.text_job_widget.build_provider_config()
        except Exception as exc:
            QMessageBox.critical(self, WINDOW_TITLE, f"文字识别 dry run 无法启动：{exc}")
            return
        if provider_config is None:
            QMessageBox.warning(self, WINDOW_TITLE, "请先启用并配置文字提取任务，再执行 dry run。")
            return
        image_options = self.text_job_widget.build_image_options()
        image = self._preprocess_preview(image_options)
        prompt = build_text_dry_run_prompt(self.subtitle_language_combo.currentData() or "auto")
        self.style_prep_mode = "text_dry_run"
        self.refine_style_button.setEnabled(False)
        self.analyze_style_image_button.setEnabled(False)
        self.dry_run_text_button.setEnabled(False)
        self.style_prep_worker = StylePrepWorker(provider_config, prompt, [image])
        self.style_prep_worker.finished_text.connect(self._on_style_prep_finished)
        self.style_prep_worker.failed.connect(self._on_style_prep_failed)
        self.style_prep_worker.finished.connect(self._clear_style_prep_worker)
        self.style_prep_worker.start()
        self._append_log("开始文字识别 dry run")

    def _on_style_prep_finished(self, result: ProviderResponse) -> None:
        text = result.text.strip()
        usage_summary = self._format_usage_summary(result.usage)
        if self.style_prep_mode == "text_dry_run":
            self._append_log(f"文字识别 dry run 返回：{text} [{usage_summary}]")
            QMessageBox.information(self, WINDOW_TITLE, text or "无返回内容")
            return
        compact = self._normalize_style_compact_text(text)
        self.style_compact_edit.setText(compact)
        self._append_log(f"样式整理完成：{compact} [{usage_summary}]")

    def _on_style_prep_failed(self, message: str) -> None:
        self._append_log(f"样式整理失败：{message}")
        QMessageBox.critical(self, WINDOW_TITLE, message)

    def _clear_style_prep_worker(self) -> None:
        self.refine_style_button.setEnabled(True)
        self.analyze_style_image_button.setEnabled(True)
        self.dry_run_text_button.setEnabled(True)
        self.style_prep_worker = None
        self.style_prep_mode = ""

    # ------------------------------------------------------------------
    # Style table
    # ------------------------------------------------------------------

    def _set_style_cell(self, row: int, column: int, value: str) -> None:
        if column == 4:
            combo = QComboBox()
            combo.addItem("任意", "either")
            combo.addItem("单行", "single")
            combo.addItem("双行", "double")
            index = combo.findData(value or "either")
            combo.setCurrentIndex(index if index >= 0 else 0)
            self.styles_table.setCellWidget(row, column, combo)
            return
        item = QTableWidgetItem(value)
        self.styles_table.setItem(row, column, item)

    def _add_style_row(
        self,
        style_id: str = "",
        display_name: str = "",
        ass_style_name: str = "",
        feature_notes: str = "",
        layout_hint: str = "either",
        samples: list[StyleSample] | None = None,
    ) -> None:
        row = self.styles_table.rowCount()
        self.styles_table.insertRow(row)
        values = [style_id, display_name, ass_style_name, feature_notes, layout_hint]
        for column, value in enumerate(values):
            self._set_style_cell(row, column, value)
        self.styles_table.setVerticalHeaderItem(row, QTableWidgetItem("samples" if samples else ""))

    def _remove_selected_style_rows(self) -> None:
        selected_rows = sorted({index.row() for index in self.styles_table.selectedIndexes()}, reverse=True)
        for row in selected_rows:
            self.styles_table.removeRow(row)

    def _renumber_style_rows(self) -> None:
        next_id = 1
        for row in range(self.styles_table.rowCount()):
            has_content = False
            for column in range(1, self.styles_table.columnCount()):
                if column == 4:
                    combo = self.styles_table.cellWidget(row, column)
                    if combo is not None and combo.currentData() != "either":
                        has_content = True
                        break
                    continue
                item = self.styles_table.item(row, column)
                if item is not None and item.text().strip():
                    has_content = True
                    break
            if has_content:
                self._set_style_cell(row, 0, str(next_id))
                next_id += 1

    def _import_style_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "导入样式模板", "", "JSON files (*.json);;All files (*)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("样式模板 JSON 顶层必须是数组")
            self._load_style_rows(data)
            self._renumber_style_rows()
            self._set_imported_ass_style_section("", [], enabled=False)
            self._append_log(f"已导入样式模板：{path}")
        except Exception as exc:
            QMessageBox.critical(self, WINDOW_TITLE, str(exc))

    def _import_ass_styles(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "从ASS导入样式", "", "ASS files (*.ass);;All files (*)")
        if not path:
            return
        try:
            styles = parse_ass_styles(path)
            if not styles:
                QMessageBox.warning(self, WINDOW_TITLE, "该 ASS 文件中未找到 [V4+ Styles] 样式定义。")
                return
            style_section = extract_ass_style_section(path)
            self._load_style_rows(styles)
            self._renumber_style_rows()
            self._set_imported_ass_style_section(path, style_section, enabled=True)
            self._append_log(f"已从 ASS 导入 {len(styles)} 个样式，并启用输出样式段覆盖：{path}")
        except Exception as exc:
            QMessageBox.critical(self, WINDOW_TITLE, f"导入 ASS 样式失败：{exc}")

    def _set_imported_ass_style_section(self, path: str, style_section: list[str], *, enabled: bool) -> None:
        self.imported_ass_style_path = path
        self.imported_ass_style_section = list(style_section)
        has_section = bool(style_section)
        self.override_output_styles_check.setEnabled(has_section)
        self.override_output_styles_check.setChecked(enabled and has_section)
        if has_section:
            source_name = Path(path).name if path else "已保存样式段"
            style_count = sum(1 for line in style_section if line.strip().startswith("Style:"))
            self.override_output_styles_check.setText(f"输出时用导入 ASS 样式覆盖样式段（{source_name}，{style_count} 个样式）")
        else:
            self.override_output_styles_check.setText("输出时用导入 ASS 样式覆盖样式段")

    def _export_style_template(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "导出样式模板", "styles.json", "JSON files (*.json)")
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(self._dump_style_rows(), ensure_ascii=False, indent=2), encoding="utf-8")
            self._append_log(f"已导出样式模板：{path}")
        except Exception as exc:
            QMessageBox.critical(self, WINDOW_TITLE, str(exc))

    def _append_compact_style_row(self) -> None:
        compact = self.style_compact_edit.text().strip()
        if not compact:
            QMessageBox.critical(self, WINDOW_TITLE, "请先生成或填写紧凑样式描述")
            return
        next_id = self.styles_table.rowCount() + 1
        self._add_style_row(str(next_id), f"Style {next_id}", f"Style{next_id:02d}", compact, "either")
        self._renumber_style_rows()

    def _save_style_sample_backup(self, style_id: int, feature_notes: str) -> StyleSample | None:
        if not self.preview_frame_bytes:
            return None
        image_options = self.style_job_widget.build_image_options()
        _, sample_bytes = self._preprocess_preview(image_options)
        STYLE_SAMPLE_DIR.mkdir(exist_ok=True)
        timestamp_ms = int(round(self._selected_preview_time_sec() * 1000))
        extension = "jpg" if image_options.format_name.upper() == "JPEG" else image_options.format_name.lower()
        stem = f"style_{style_id:02d}_{timestamp_ms}"
        image_path = STYLE_SAMPLE_DIR / f"{stem}.{extension}"
        meta_path = STYLE_SAMPLE_DIR / f"{stem}.json"
        image_path.write_bytes(base64.b64decode(sample_bytes))
        metadata = {
            "style_id": style_id,
            "feature_notes": feature_notes,
            "video_path": self.video_edit.text().strip(),
            "subtitle_input_path": self.ass_input_edit.text().strip(),
            "timestamp_ms": timestamp_ms,
            "subtitle_language": self.subtitle_language_combo.currentData() or "auto",
            "subtitle_region_rect": self._subtitle_region_rect(),
            "image_format": image_options.format_name,
            "quality": image_options.quality,
            "max_edge": image_options.max_edge,
        }
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return StyleSample(image_path=str(image_path), timestamp_ms=timestamp_ms, note=feature_notes)

    def _lock_style_from_preview(self) -> None:
        if not self._ensure_preview_frame():
            return
        compact = self.style_compact_edit.text().strip()
        if not compact:
            QMessageBox.critical(self, WINDOW_TITLE, "请先生成紧凑样式描述后再锁定")
            return
        next_id = self.styles_table.rowCount() + 1
        sample = self._save_style_sample_backup(next_id, compact)
        self._add_style_row(str(next_id), f"Style {next_id}", f"Style{next_id:02d}", compact, "either", [sample] if sample else [])
        self._renumber_style_rows()
        if sample is not None:
            self._append_log(f"已从当前预览锁定一条样式样本，并备份到 {sample.image_path}")
        else:
            self._append_log("已从当前预览锁定一条样式样本。")

    def _parse_style_profiles(self) -> list[StyleProfile]:
        profiles: list[StyleProfile] = []
        for row in range(self.styles_table.rowCount()):
            values = []
            for column in range(self.styles_table.columnCount()):
                if column == 4:
                    combo = self.styles_table.cellWidget(row, column)
                    values.append(str(combo.currentData()) if combo is not None else "either")
                    continue
                item = self.styles_table.item(row, column)
                values.append(item.text().strip() if item is not None else "")
            if not any(values):
                continue
            style_id_text, display_name, ass_style_name, feature_notes, layout_hint = values
            if not style_id_text or not display_name or not ass_style_name:
                raise ValueError(f"第 {row + 1} 行样式缺少必填字段")
            if layout_hint and layout_hint not in LAYOUT_HINTS:
                raise ValueError(f"第 {row + 1} 行布局提示必须是：任意、单行或双行")
            profiles.append(
                StyleProfile(
                    style_id=int(style_id_text),
                    display_name=display_name,
                    ass_style_name=ass_style_name,
                    feature_notes=feature_notes,
                    layout_hint=layout_hint or "either",
                )
            )
        return profiles

    def _load_style_rows(self, data: list[dict]) -> None:
        self.styles_table.setRowCount(0)
        for item in data:
            samples = [StyleSample(**sample) for sample in item.get("samples", [])]
            self._add_style_row(
                str(item.get("style_id", "")),
                item.get("display_name", ""),
                item.get("ass_style_name", ""),
                item.get("feature_notes", ""),
                item.get("layout_hint", "either"),
                samples,
            )

    def _load_legacy_style_text(self, raw_text: str) -> None:
        self.styles_table.setRowCount(0)
        for raw_line in raw_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split("|", 4)]
            while len(parts) < 5:
                parts.append("")
            self._add_style_row(parts[0], parts[1], parts[2], parts[3], parts[4] or "either")

    def _dump_style_rows(self) -> list[dict]:
        rows = []
        for row_index, profile in enumerate(self._parse_style_profiles()):
            header_item = self.styles_table.verticalHeaderItem(row_index)
            samples = []
            if header_item is not None and header_item.text().strip() == "samples":
                stem = f"style_{profile.style_id:02d}_"
                for meta_path in sorted(STYLE_SAMPLE_DIR.glob(f"{stem}*.json")) if STYLE_SAMPLE_DIR.exists() else []:
                    data = json.loads(meta_path.read_text(encoding="utf-8"))
                    image_path = meta_path.with_suffix(".jpg")
                    if not image_path.exists():
                        webp_path = meta_path.with_suffix(".webp")
                        image_path = webp_path if webp_path.exists() else image_path
                    samples.append(
                        {
                            "image_path": str(image_path),
                            "timestamp_ms": int(data.get("timestamp_ms", 0)),
                            "note": data.get("feature_notes", ""),
                        }
                    )
            rows.append(
                {
                    "style_id": profile.style_id,
                    "display_name": profile.display_name,
                    "ass_style_name": profile.ass_style_name,
                    "feature_notes": profile.feature_notes,
                    "layout_hint": profile.layout_hint,
                    "samples": samples,
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Subtitle events
    # ------------------------------------------------------------------

    def _load_subtitle_events(self) -> None:
        self.loaded_events = []
        self.current_event_index = -1
        path = self.ass_input_edit.text().strip()
        if not path:
            self._populate_subtitle_jump_combo()
            return
        document = parse_subtitle_document(path)
        self.loaded_events = document.events
        self._populate_subtitle_jump_combo()
        if self.loaded_events:
            self.current_event_index = 0

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------

    def _validate_job_widget(self, widget: ProviderJobWidget) -> str | None:
        if not widget.enabled_checkbox.isChecked():
            return None
        provider_type = widget.provider_combo.currentText()
        model = widget.model_edit.text().strip()
        api_key = widget.api_key_edit.text().strip()
        base_url = widget.base_url_edit.text().strip()
        if not model:
            return f"{widget.title()} 缺少 model"
        if not api_key:
            return f"{widget.title()} 缺少 API key"
        if provider_type == "openai" and not base_url:
            return f"{widget.title()} 缺少 Base URL"
        return None

    def _run_startup_checks(self) -> None:
        self._append_log(f"启动自检：config={CONFIG_PATH.name}")
        if CONFIG_PATH.exists():
            self._append_log("[OK] 已找到 config.json")
        else:
            self._append_log("[WARN] 尚未生成 config.json，首次保存设置后会创建")

        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            self._append_log(f"[OK] ffmpeg: {ffmpeg_path}")
        else:
            self._append_log("[WARN] 未找到 ffmpeg，运行截图流程前请先安装并加入 PATH")

        for widget in [self.style_job_widget, self.text_job_widget]:
            issue = self._validate_job_widget(widget)
            if issue is None:
                state = "已启用并配置完整" if widget.enabled_checkbox.isChecked() else "未启用"
                self._append_log(f"[OK] {widget.title()}：{state}")
            else:
                self._append_log(f"[WARN] {issue}")

    def _selected_output_style_section(self) -> list[str] | None:
        if not self.override_output_styles_check.isChecked() or not self.imported_ass_style_section:
            return None
        return list(self.imported_ass_style_section)

    def _style_section_source_label(self) -> str:
        if not self.imported_ass_style_section:
            return ""
        style_count = sum(1 for line in self.imported_ass_style_section if line.strip().startswith("Style:"))
        source = self.imported_ass_style_path or "已保存的导入样式段"
        return f"{source}（{style_count} 个样式）"

    def _payload_event_count(self, payload: RunPayload) -> int:
        if payload.event_ids is not None:
            return len(payload.event_ids)
        if self.loaded_events:
            return len(self.loaded_events)
        try:
            return len(parse_subtitle_document(payload.ass_input_path).events)
        except Exception:
            return 0

    def _confirm_run_payload(self, payload: RunPayload) -> bool:
        retry_count = len(payload.event_ids) if payload.event_ids else 0
        dialog = RunSummaryDialog(
            self,
            payload,
            event_count=self._payload_event_count(payload),
            retry_count=retry_count,
            output_exists=Path(payload.ass_output_path).exists(),
            style_section_source=self._style_section_source_label(),
        )
        if dialog.exec() != QDialog.Accepted:
            return False
        if dialog.use_output_style_override and self.imported_ass_style_section:
            payload.output_style_section_lines = list(self.imported_ass_style_section)
            self._append_log(f"本次运行将覆盖输出 ASS 样式段：{self._style_section_source_label()}")
        else:
            payload.output_style_section_lines = None
            if self.imported_ass_style_section:
                self._append_log("本次运行不覆盖输出 ASS 样式段。")
        return True

    def _build_payload(self) -> RunPayload:
        video_path = self.video_edit.text().strip()
        ass_input_path = self.ass_input_edit.text().strip()
        ass_output_path = self.ass_output_edit.text().strip()
        if not video_path:
            raise ValueError("请先选择视频文件")
        if not ass_input_path:
            raise ValueError("请先选择字幕输入文件")
        if Path(ass_input_path).suffix.lower() not in {".ass", ".srt"}:
            raise ValueError("字幕输入目前仅支持 ASS 或 SRT")
        if not ass_output_path:
            raise ValueError("请先填写 ASS 输出文件")
        region_rect = self._subtitle_region_rect()
        if region_rect["width"] <= 0 or region_rect["height"] <= 0:
            raise ValueError("请在预览图中框选有效的字幕区域")
        start, end = self._subtitle_region_start_end()

        style_provider = self.style_job_widget.build_provider_config()
        text_provider = self.text_job_widget.build_provider_config()
        if text_provider is not None and text_provider.max_output_tokens < 512:
            text_provider.max_output_tokens = 512
        if style_provider is None and text_provider is None:
            raise ValueError("样式识别和文字提取至少要启用一个任务")

        style_profiles = self._parse_style_profiles()
        if style_provider is not None and not style_profiles:
            raise ValueError("启用样式识别时，至少要提供一个锁定样式")

        text_image_options = self.text_job_widget.build_image_options()
        if text_image_options.max_edge < 1280:
            text_image_options.max_edge = 1280

        return RunPayload(
            video_path=video_path,
            ass_input_path=ass_input_path,
            ass_output_path=ass_output_path,
            style_profiles=style_profiles,
            style_provider=style_provider,
            text_provider=text_provider,
            subtitle_region_start=start,
            subtitle_region_end=end,
            subtitle_region_rect=region_rect,
            frame_concurrency=self.frame_concurrency_spin.value(),
            subtitle_language=self.subtitle_language_combo.currentData() or "auto",
            style_image_options=self.style_job_widget.build_image_options(),
            text_image_options=text_image_options,
            event_ids=self.pending_retry_event_ids,
            include_samples=self.include_samples_check.isChecked(),
            failed_tasks_map=self.pending_retry_failed_tasks,
            output_style_section_lines=self._selected_output_style_section(),
        )

    def _start_run(self) -> None:
        if self.worker is not None:
            return
        try:
            payload = self._build_payload()
        except Exception as exc:
            QMessageBox.critical(self, WINDOW_TITLE, str(exc))
            return
        if not self._confirm_run_payload(payload):
            self.pending_retry_event_ids = None
            self.pending_retry_failed_tasks = None
            return
        self._save_settings()
        retry_count = len(self.pending_retry_event_ids) if self.pending_retry_event_ids else 0
        if retry_count:
            task_summary = {}
            if self.pending_retry_failed_tasks:
                for tasks in self.pending_retry_failed_tasks.values():
                    for t in tasks:
                        task_summary[t] = task_summary.get(t, 0) + 1
            self._append_log(f"开始复跑失败项… events={retry_count} 任务分布={task_summary}")
        else:
            self._append_log("开始运行 GVS2…")
        self.run_progress_bar.setValue(0)
        self.run_progress_label.setText("准备开始")
        self.run_button.setEnabled(False)
        self.worker = PipelineWorker(payload)
        self.worker.finished_ok.connect(self._on_run_finished)
        self.worker.failed.connect(self._on_run_failed)
        self.worker.progress_changed.connect(self._update_run_progress)
        self.worker.finished.connect(self._clear_worker)
        self.worker.start()

    def _clear_worker(self) -> None:
        self.run_button.setEnabled(True)
        self.retry_failed_button.setEnabled(self._has_retryable_job())
        self.pending_retry_event_ids = None
        self.pending_retry_failed_tasks = None
        self.worker = None

    def _update_run_progress(self, current: int, total: int, message: str) -> None:
        total = max(total, 1)
        value = int(current * 100 / total)
        self.run_progress_bar.setValue(max(0, min(100, value)))
        self.run_progress_label.setText(message)

    def _on_run_finished(self, results: list[EventJobResult]) -> None:
        self.run_progress_bar.setValue(100)
        self.run_progress_label.setText("处理完成")
        payload = self.worker.payload if self.worker is not None else None
        self.last_run_results = results
        self.last_run_style_profiles = list(payload.style_profiles) if payload is not None else self._parse_style_profiles()
        self.review_results_button.setEnabled(bool(results) and bool(self.ass_output_edit.text().strip()))
        style_hits = sum(1 for item in results if item.style_result and item.style_result.matched)
        review_hits = sum(1 for item in results if item.style_result and item.style_result.review_required)
        text_review_hits = sum(1 for item in results if item.text_result and item.text_result.review_required)
        text_hits = sum(1 for item in results if item.text_result and item.text_result.matched)
        no_subtitle = sum(1 for item in results if item.text_result and not item.text_result.matched)
        failed_events = [item for item in results if item.error_messages]
        usage_summary = self._summarize_results_usage(results)
        self._append_log(
            f"处理完成。events={len(results)} style_hits={style_hits} review_hits={review_hits} text_hits={text_hits} no_subtitle={no_subtitle} failed={len(failed_events)} {usage_summary} 输出={self.ass_output_edit.text().strip()}"
        )
        job_path = self._write_job_record(results)
        if job_path:
            self._append_log(f"任务状态已保存：{job_path}")
        if review_hits:
            self._append_log(f"其中 {review_hits} 条样式已标记为“需核查”。")
        if text_review_hits:
            self._append_log(f"其中 {text_review_hits} 条文字识别结果较长或疑似不完整，已进入“文字需核查”。")
        if no_subtitle:
            self._append_log(f"其中 {no_subtitle} 条文字识别结果为未识别到字幕。")
        if failed_events:
            self.retry_failed_button.setEnabled(True)
            self._append_log(f"其中 {len(failed_events)} 条字幕处理失败，已跳过并继续后续任务。")
            report_path = self._write_failed_events_report(failed_events)
            if report_path:
                self._append_log(f"失败事件清单已导出：{report_path}")
            for item in failed_events[:5]:
                self._append_log(f"失败 {item.event_id}: {' | '.join(item.error_messages)}")
        else:
            self.retry_failed_button.setEnabled(False)
        if review_hits or text_review_hits or any("text" in item.failed_tasks for item in results):
            QMessageBox.information(self, WINDOW_TITLE, "处理完成。可点击“审查结果”处理需核查样式或复跑文字失败。")
        else:
            QMessageBox.information(self, WINDOW_TITLE, "处理完成")

    def _on_run_failed(self, message: str) -> None:
        self.run_progress_label.setText("处理失败")
        self._append_log(f"处理失败：{message}")
        QMessageBox.critical(self, WINDOW_TITLE, message)

    def _write_job_record(self, results: list[EventJobResult]) -> str | None:
        if self.worker is None:
            return None
        payload = self.worker.payload
        if not payload.ass_output_path:
            return None
        event_lookup = self._job_event_lookup(payload.ass_output_path)
        record = self.job_store.save_run(
            video_path=payload.video_path,
            subtitle_input_path=payload.ass_input_path,
            ass_output_path=payload.ass_output_path,
            results=results,
            event_lookup=event_lookup,
            previous_record=self.job_store.load_latest_for_output(payload.ass_output_path),
        )
        return str(self.job_store.path_for(record["job_id"]))

    def _job_event_lookup(self, output_path: str) -> dict[str, SubtitleEvent]:
        try:
            if output_path and Path(output_path).exists():
                document = parse_subtitle_document(output_path)
                return {event.event_id: event for event in document.events}
        except Exception as exc:
            self._append_log(f"读取输出字幕用于任务记录失败，改用输入字幕事件：{exc}")
        return {event.event_id: event for event in self.loaded_events}

    def _open_result_review(self) -> None:
        output_path = self.ass_output_edit.text().strip()
        if not output_path or not Path(output_path).exists():
            QMessageBox.warning(self, WINDOW_TITLE, "请先完成一次运行并生成输出 ASS。")
            return
        results = self.last_run_results or self._results_from_latest_job_record()
        if not results:
            QMessageBox.warning(self, WINDOW_TITLE, "当前没有可审查的运行结果。")
            return
        style_profiles = self.last_run_style_profiles or self._parse_style_profiles()
        review_video_path = self.video_edit.text().strip()
        if not review_video_path:
            review_video_path = str(self._latest_job_record().get("video_path", "")).strip()
        dialog = ResultReviewDialog(
            self,
            output_path,
            results,
            style_profiles,
            video_path=review_video_path,
            subtitle_region_rect=self._subtitle_region_rect(),
        )
        dialog.exec()
        if dialog.retry_failed_tasks_map:
            self._start_retry_from_map(dialog.retry_failed_tasks_map)

    def _results_from_latest_job_record(self) -> list[EventJobResult]:
        record = self._latest_job_record()
        results: list[EventJobResult] = []
        for item in record.get("events", []):
            event_id = str(item.get("event_id", "")).strip()
            if not event_id:
                continue
            text_task = (item.get("tasks") or {}).get("text") or {}
            text_result = None
            if text_task.get("review_required") or text_task.get("parsed_text") or text_task.get("raw_response"):
                text_result = TextJobResult(
                    matched=text_task.get("status") == "success",
                    text=str(text_task.get("parsed_text", "")),
                    raw_response=str(text_task.get("raw_response", "")),
                    review_required=bool(text_task.get("review_required", False)),
                    review_reasons=[str(reason) for reason in text_task.get("review_reasons", [])],
                )
            results.append(
                EventJobResult(
                    event_id=event_id,
                    text_result=text_result,
                    error_messages=[str(message) for message in item.get("error_messages", [])],
                    failed_tasks=[str(task) for task in item.get("failed_tasks", [])],
                    final_action=str(item.get("final_action", "skip")),
                )
            )
        return results

    def _start_retry_from_map(self, failed_tasks_map: dict[str, list[str]]) -> None:
        if self.worker is not None:
            QMessageBox.warning(self, WINDOW_TITLE, "当前仍有任务运行中，无法启动复跑。")
            return
        event_ids = set(failed_tasks_map.keys())
        if not event_ids:
            QMessageBox.warning(self, WINDOW_TITLE, "没有可复跑的失败项。")
            return
        self.pending_retry_event_ids = event_ids
        self.pending_retry_failed_tasks = failed_tasks_map
        self._append_log(f"准备复跑失败项：{len(event_ids)} 条")
        self._start_run()

    def _latest_job_record(self) -> dict:
        output_path = self.ass_output_edit.text().strip()
        if not output_path:
            return {}
        return self.job_store.load_latest_for_output(output_path)

    def _has_retryable_job(self) -> bool:
        record = self._latest_job_record()
        if self.job_store.failed_tasks_map(record):
            return True
        report_path = self._failed_events_report_path()
        return report_path is not None and report_path.exists()

    def _write_failed_events_report(self, failed_events: list[EventJobResult]) -> str | None:
        output_path = self.ass_output_edit.text().strip()
        if not output_path or not failed_events:
            return None
        report_path = Path(output_path).with_name(Path(output_path).stem + "_failed_events.json")
        event_lookup = {event.event_id: event for event in self.loaded_events}
        payload = []
        for item in failed_events:
            event = event_lookup.get(item.event_id)
            payload.append(
                {
                    "event_id": item.event_id,
                    "start_ms": event.start_ms if event is not None else None,
                    "end_ms": event.end_ms if event is not None else None,
                    "text": event.text if event is not None else "",
                    "original_style": event.original_style if event is not None else "",
                    "final_action": item.final_action,
                    "error_messages": item.error_messages,
                    "failed_tasks": item.failed_tasks,
                }
            )
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(report_path)

    def _failed_events_report_path(self) -> Path | None:
        output_path = self.ass_output_edit.text().strip()
        if not output_path:
            return None
        return Path(output_path).with_name(Path(output_path).stem + "_failed_events.json")

    def _retry_failed_events(self) -> None:
        record = self._latest_job_record()
        failed_tasks_map = self.job_store.failed_tasks_map(record)
        if failed_tasks_map:
            event_ids = set(failed_tasks_map.keys())
        else:
            report_path = self._failed_events_report_path()
            if report_path is None or not report_path.exists():
                QMessageBox.warning(self, WINDOW_TITLE, "未找到失败事件清单，请先完成一次包含失败项的运行。")
                return
            try:
                data = json.loads(report_path.read_text(encoding="utf-8"))
                event_ids = {str(item.get("event_id", "")).strip() for item in data if str(item.get("event_id", "")).strip()}
                failed_tasks_map = {}
                for item in data:
                    eid = str(item.get("event_id", "")).strip()
                    if not eid:
                        continue
                    tasks = item.get("failed_tasks")
                    if tasks:
                        failed_tasks_map[eid] = tasks
                    else:
                        # Legacy report without failed_tasks: assume both tasks failed
                        failed_tasks_map[eid] = ["style", "text"]
            except Exception as exc:
                QMessageBox.critical(self, WINDOW_TITLE, f"读取失败事件清单失败：{exc}")
                return
        if not event_ids:
            QMessageBox.warning(self, WINDOW_TITLE, "未找到失败事件清单，请先完成一次包含失败项的运行。")
            return
        self._start_retry_from_map(failed_tasks_map)

    # ------------------------------------------------------------------
    # Usage & logging
    # ------------------------------------------------------------------

    def _format_usage_summary(self, usage: ProviderUsage) -> str:
        parts = [f"in={usage.total_input_tokens}", f"out={usage.output_tokens}"]
        if usage.estimated_cost_usd is not None:
            parts.append(f"cost=${usage.estimated_cost_usd:.6f}")
        return " ".join(parts)

    def _summarize_results_usage(self, results: list[EventJobResult]) -> str:
        input_tokens = 0
        output_tokens = 0
        estimated_cost = 0.0
        has_cost = False
        for item in results:
            for job in (item.style_result, item.text_result):
                if job is None:
                    continue
                input_tokens += job.usage.total_input_tokens
                output_tokens += job.usage.output_tokens
                if job.usage.estimated_cost_usd is not None:
                    estimated_cost += job.usage.estimated_cost_usd
                    has_cost = True
        summary = f"tokens in={input_tokens} out={output_tokens}"
        if has_cost:
            summary += f" cost=${estimated_cost:.6f}"
        return summary

    def _append_log(self, message: str) -> None:
        self.log_edit.appendPlainText(message)

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        data = self.settings_store.load().get("gvs2", {})
        self.video_edit.setText(data.get("video_path", ""))
        self.ass_input_edit.setText(data.get("ass_input_path", ""))
        self.ass_output_edit.setText(data.get("ass_output_path", ""))
        region_rect = data.get("subtitle_region_rect")
        if not isinstance(region_rect, dict):
            legacy_start = int(data.get("subtitle_region_start", 66))
            legacy_end = int(data.get("subtitle_region_end", 100))
            region_rect = {"x": 0.0, "y": float(legacy_start), "width": 100.0, "height": float(max(1, legacy_end - legacy_start))}
        self.preview_label.set_region(region_rect)
        self.subtitle_region_info_edit.setText(self._format_region_info(self._subtitle_region_rect()))
        subtitle_language = data.get("subtitle_language", "auto")
        language_index = self.subtitle_language_combo.findData(subtitle_language)
        self.subtitle_language_combo.setCurrentIndex(language_index if language_index >= 0 else 0)
        self.frame_concurrency_spin.setValue(int(data.get("frame_concurrency", 5)))
        self.preview_time_slider.setValue(int(data.get("preview_time_ratio", 500)))
        self.style_description_edit.setPlainText(data.get("style_description_text", ""))
        self.style_compact_edit.setText(data.get("style_compact_text", ""))
        style_rows = data.get("style_profiles") or []
        if style_rows:
            self._load_style_rows(style_rows)
        else:
            legacy_text = data.get("style_profiles_text", "")
            if legacy_text:
                self._load_legacy_style_text(legacy_text)
        self.style_job_widget.load_settings(data.get("style_job", {}))
        self.text_job_widget.load_settings(data.get("text_job", {}))
        self._enforce_text_ocr_budget(self.text_job_widget)
        self._style_job_history = data.get("style_job_history", [])
        self._text_job_history = data.get("text_job_history", [])
        self.include_samples_check.setChecked(data.get("include_samples", False))
        imported_style_section = data.get("imported_ass_style_section") or []
        self._set_imported_ass_style_section(
            data.get("imported_ass_style_path", ""),
            imported_style_section if isinstance(imported_style_section, list) else [],
            enabled=bool(data.get("override_output_styles", False)),
        )
        if self.ass_input_edit.text().strip():
            try:
                self._load_subtitle_events()
            except Exception as exc:
                self._append_log(f"字幕事件加载失败：{exc}")
        output_path = self.ass_output_edit.text().strip()
        self.review_results_button.setEnabled(bool(output_path and Path(output_path).exists() and self._latest_job_record()))
        QTimer.singleShot(0, self._refresh_video_preview)

    def _save_settings(self) -> None:
        self.settings_store.save(
            {
                "gvs2": {
                    "video_path": self.video_edit.text().strip(),
                    "ass_input_path": self.ass_input_edit.text().strip(),
                    "ass_output_path": self.ass_output_edit.text().strip(),
                    "subtitle_region_rect": self._subtitle_region_rect(),
                    "subtitle_language": self.subtitle_language_combo.currentData() or "auto",
                    "frame_concurrency": self.frame_concurrency_spin.value(),
                    "preview_time_ratio": self.preview_time_slider.value(),
                    "style_description_text": self.style_description_edit.toPlainText(),
                    "style_compact_text": self.style_compact_edit.text().strip(),
                    "style_profiles": self._dump_style_rows(),
                    "include_samples": self.include_samples_check.isChecked(),
                    "imported_ass_style_path": self.imported_ass_style_path,
                    "imported_ass_style_section": self.imported_ass_style_section,
                    "override_output_styles": self.override_output_styles_check.isChecked(),
                    "style_job": self.style_job_widget.dump_settings(),
                    "text_job": self.text_job_widget.dump_settings(),
                    "style_job_history": self._style_job_history,
                    "text_job_history": self._text_job_history,
                }
            }
        )
        self._append_log(f"设置已保存到 {CONFIG_PATH.name}。")

    def closeEvent(self, event: QCloseEvent) -> None:
        self._save_settings()
        super().closeEvent(event)


def launch() -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()
