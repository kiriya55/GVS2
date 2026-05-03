from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent, QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
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

from models.job_result import EventJobResult
from models.style_profile import StyleProfile, StyleSample
from providers.base import ProviderResponse, ProviderUsage
from providers.prompt_builder import (
    build_style_image_analysis_prompt,
    build_style_preview_prompt,
    build_style_refine_prompt,
    build_text_dry_run_prompt,
)
from services.image_preprocess import preprocess_for_llm
from services.media import extract_frame_ffmpeg, probe_video_duration_ffprobe
from services.subtitle_parser import parse_subtitle_document
from storage.settings_store import SettingsStore
from ui.widgets import (
    IMAGE_FORMATS,
    LAYOUT_HINTS,
    STYLE_KEYWORDS_ZH,
    SUBTITLE_LANGUAGES,
    ApiSettingsDialog,
    PipelineWorker,
    ProviderJobWidget,
    RunPayload,
    StylePrepWorker,
)


WINDOW_TITLE = "GVS2"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
ICON_PATH = Path(__file__).resolve().parents[1] / "ico.ico"
STYLE_SAMPLE_DIR = Path(__file__).resolve().parents[1] / "style_samples"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(980, 820)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self.settings_store = SettingsStore(str(CONFIG_PATH))
        self.worker: PipelineWorker | None = None
        self.style_prep_worker: StylePrepWorker | None = None
        self.style_prep_mode = ""
        self.preview_frame_bytes: bytes | None = None
        self.loaded_events = []
        self.current_event_index = -1
        self.pending_retry_event_ids: set[str] | None = None
        self.api_settings_dialog = ApiSettingsDialog(self)
        self.style_job_widget = self.api_settings_dialog.style_job_widget
        self.text_job_widget = self.api_settings_dialog.text_job_widget
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
        self.append_compact_style_button = QPushButton("追加当前紧凑描述")
        self.append_compact_style_button.clicked.connect(self._append_compact_style_row)
        self.lock_preview_style_button = QPushButton("从当前预览锁定样式")
        self.lock_preview_style_button.clicked.connect(self._lock_style_from_preview)
        styles_actions.addWidget(self.add_style_button)
        styles_actions.addWidget(self.remove_style_button)
        styles_actions.addWidget(self.renumber_style_button)
        styles_actions.addWidget(self.import_styles_button)
        styles_actions.addWidget(self.export_styles_button)
        styles_actions.addWidget(self.append_compact_style_button)
        styles_actions.addWidget(self.lock_preview_style_button)
        styles_layout.addLayout(styles_actions)
        self.include_samples_check = QCheckBox("运行时使用已保存的样本图作为视觉参考（few-shot）")
        self.include_samples_check.setToolTip("勾选后，已锁定样式的样本图会作为视觉示例发送给 LLM，提高识别准确率，但会增加 token 消耗")
        styles_layout.addWidget(self.include_samples_check)
        parent_layout.addWidget(styles_box, 1)

    def _setup_preview_group(self, parent_layout: QVBoxLayout) -> None:
        preview_box = QGroupBox("视频预览")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_label = QLabel("选择视频后可预览字幕区域")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(360, 220)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_time_slider = QSlider(Qt.Horizontal)
        self.preview_time_slider.setRange(0, 1000)
        self.preview_time_slider.setValue(500)
        self.preview_time_slider.valueChanged.connect(self._refresh_video_preview)
        self.preview_time_hint = QLabel("预览时间：50%")
        preview_nav = QHBoxLayout()
        self.prev_subtitle_button = QPushButton("上一个字幕时点")
        self.prev_subtitle_button.clicked.connect(lambda: self._jump_subtitle_event(-1))
        self.next_subtitle_button = QPushButton("下一个字幕时点")
        self.next_subtitle_button.clicked.connect(lambda: self._jump_subtitle_event(1))
        self.preview_processed_button = QPushButton("预览处理图")
        self.preview_processed_button.clicked.connect(self._show_processed_image_preview)
        self.dry_run_text_button = QPushButton("文字提取试运行")
        self.dry_run_text_button.clicked.connect(self._dry_run_text_recognition)
        preview_nav.addWidget(self.prev_subtitle_button)
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
        self.region_start_spin = QSpinBox()
        self.region_start_spin.setRange(0, 100)
        self.region_start_spin.setValue(66)
        self.region_end_spin = QSpinBox()
        self.region_end_spin.setRange(0, 100)
        self.region_end_spin.setValue(100)
        self.subtitle_language_combo = QComboBox()
        for value, label in SUBTITLE_LANGUAGES:
            self.subtitle_language_combo.addItem(label, value)
        self.region_start_spin.valueChanged.connect(self._refresh_video_preview)
        self.region_end_spin.valueChanged.connect(self._refresh_video_preview)
        region_form.addRow("起始百分比", self.region_start_spin)
        region_form.addRow("结束百分比", self.region_end_spin)
        region_form.addRow("字幕语言", self.subtitle_language_combo)
        region_box.setLayout(region_form)
        parent_layout.addWidget(region_box)

    def _setup_actions(self, parent_layout: QVBoxLayout) -> None:
        actions_layout = QHBoxLayout()
        self.refresh_preview_button = QPushButton("刷新预览")
        self.refresh_preview_button.clicked.connect(self._refresh_video_preview)
        self.run_button = QPushButton("运行当前勾选的工作")
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
        progress_actions.addStretch(1)
        progress_actions.addWidget(self.retry_failed_button)
        progress_layout.addWidget(self.run_progress_label)
        progress_layout.addWidget(self.run_progress_bar)
        progress_layout.addLayout(progress_actions)
        parent_layout.addWidget(progress_box)

    # ------------------------------------------------------------------
    # Menu & dialogs
    # ------------------------------------------------------------------

    def _open_api_settings(self) -> None:
        self.api_settings_dialog.load_settings(self.style_job_widget.dump_settings(), self.text_job_widget.dump_settings())
        if self.api_settings_dialog.exec() == QDialog.Accepted:
            style_job, text_job = self.api_settings_dialog.dump_settings()
            self.style_job_widget.load_settings(style_job)
            self.text_job_widget.load_settings(text_job)
            self._save_settings()
            self._run_startup_checks()

    def _with_browse(self, edit: QLineEdit, handler, *, save: bool = False) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit)
        button = QPushButton("浏览")
        button.clicked.connect(handler)
        row.addWidget(button)
        return container

    # ------------------------------------------------------------------
    # File browsing
    # ------------------------------------------------------------------

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.mkv *.avi *.mov);;All files (*)")
        if path:
            self.video_edit.setText(path)
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

    def _selected_preview_time_sec(self) -> float:
        video_path = self.video_edit.text().strip()
        duration = probe_video_duration_ffprobe(video_path) if video_path else None
        ratio = self.preview_time_slider.value() / 1000
        self.preview_time_hint.setText(f"预览时间：{ratio:.0%}")
        if duration and duration > 0:
            return duration * ratio
        return 0.0

    def _set_preview_to_time_ms(self, time_ms: int) -> None:
        video_path = self.video_edit.text().strip()
        duration = probe_video_duration_ffprobe(video_path) if video_path else None
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
        event = self.loaded_events[next_index]
        self._set_preview_to_time_ms(event.midpoint_ms)
        self._append_log(f"跳转到字幕事件 {next_index + 1}/{len(self.loaded_events)}：{event.start_ms}-{event.end_ms}ms")

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
        if not self.preview_frame_bytes:
            self._refresh_video_preview()
        if not self.preview_frame_bytes:
            QMessageBox.warning(self, WINDOW_TITLE, "请先生成视频预览图。")
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
            _, image_b64 = preprocess_for_llm(
                self.preview_frame_bytes,
                image_options,
                start_percent=self.region_start_spin.value(),
                end_percent=self.region_end_spin.value(),
            )
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
        top = int(pixmap.height() * self.region_start_spin.value() / 100)
        bottom = int(pixmap.height() * self.region_end_spin.value() / 100)
        painter.fillRect(0, top, pixmap.width(), max(1, bottom - top), QColor(255, 255, 255, 40))
        painter.setPen(QPen(QColor(255, 80, 80), 3))
        painter.drawRect(1, top, max(1, pixmap.width() - 2), max(1, bottom - top))
        painter.end()
        return pixmap

    def _refresh_video_preview(self) -> None:
        video_path = self.video_edit.text().strip()
        self.preview_time_hint.setText(f"预览时间：{self.preview_time_slider.value() / 1000:.0%}")
        self.preview_frame_bytes = None
        if not video_path:
            self.preview_label.setText("选择视频后可预览字幕区域")
            self.preview_label.setPixmap(QPixmap())
            return
        frame_bytes = extract_frame_ffmpeg(video_path, self._selected_preview_time_sec())
        if not frame_bytes:
            self.preview_label.setText("预览截图失败，请检查 ffmpeg 和视频路径")
            self.preview_label.setPixmap(QPixmap())
            return
        self.preview_frame_bytes = frame_bytes
        pixmap = self._render_preview_pixmap(frame_bytes)
        if pixmap is None:
            self.preview_label.setText("预览图像解码失败")
            self.preview_label.setPixmap(QPixmap())
            return
        scaled = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

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
        if not self.preview_frame_bytes:
            self._refresh_video_preview()
        if not self.preview_frame_bytes:
            QMessageBox.critical(self, WINDOW_TITLE, "请先生成视频预览图")
            return
        image_options = self.style_job_widget.build_image_options()
        image = preprocess_for_llm(
            self.preview_frame_bytes,
            image_options,
            start_percent=self.region_start_spin.value(),
            end_percent=self.region_end_spin.value(),
        )
        self._start_style_prep(build_style_image_analysis_prompt(), [image], "image")

    def _dry_run_text_recognition(self) -> None:
        if self.style_prep_worker is not None:
            return
        if not self.preview_frame_bytes:
            self._refresh_video_preview()
        if not self.preview_frame_bytes:
            QMessageBox.critical(self, WINDOW_TITLE, "请先生成视频预览图")
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
        image = preprocess_for_llm(
            self.preview_frame_bytes,
            image_options,
            start_percent=self.region_start_spin.value(),
            end_percent=self.region_end_spin.value(),
        )
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
            self._append_log(f"已导入样式模板：{path}")
        except Exception as exc:
            QMessageBox.critical(self, WINDOW_TITLE, str(exc))

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
        _, sample_bytes = preprocess_for_llm(
            self.preview_frame_bytes,
            image_options,
            start_percent=self.region_start_spin.value(),
            end_percent=self.region_end_spin.value(),
        )
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
            "subtitle_region_start": self.region_start_spin.value(),
            "subtitle_region_end": self.region_end_spin.value(),
            "image_format": image_options.format_name,
            "quality": image_options.quality,
            "max_edge": image_options.max_edge,
        }
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return StyleSample(image_path=str(image_path), timestamp_ms=timestamp_ms, note=feature_notes)

    def _lock_style_from_preview(self) -> None:
        if not self.preview_frame_bytes:
            self._refresh_video_preview()
        if not self.preview_frame_bytes:
            QMessageBox.critical(self, WINDOW_TITLE, "请先生成视频预览图")
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
            return
        document = parse_subtitle_document(path)
        self.loaded_events = document.events
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
        start = self.region_start_spin.value()
        end = self.region_end_spin.value()
        if start > end:
            raise ValueError("字幕区域起始百分比不能大于结束百分比")

        style_provider = self.style_job_widget.build_provider_config()
        text_provider = self.text_job_widget.build_provider_config()
        if style_provider is None and text_provider is None:
            raise ValueError("样式识别和文字提取至少要启用一个任务")

        style_profiles = self._parse_style_profiles()
        if style_provider is not None and not style_profiles:
            raise ValueError("启用样式识别时，至少要提供一个锁定样式")

        return RunPayload(
            video_path=video_path,
            ass_input_path=ass_input_path,
            ass_output_path=ass_output_path,
            style_profiles=style_profiles,
            style_provider=style_provider,
            text_provider=text_provider,
            subtitle_region_start=start,
            subtitle_region_end=end,
            subtitle_language=self.subtitle_language_combo.currentData() or "auto",
            style_image_options=self.style_job_widget.build_image_options(),
            text_image_options=self.text_job_widget.build_image_options(),
            event_ids=self.pending_retry_event_ids,
            include_samples=self.include_samples_check.isChecked(),
        )

    def _start_run(self) -> None:
        if self.worker is not None:
            return
        try:
            payload = self._build_payload()
        except Exception as exc:
            QMessageBox.critical(self, WINDOW_TITLE, str(exc))
            return
        self._save_settings()
        retry_count = len(self.pending_retry_event_ids) if self.pending_retry_event_ids else 0
        if retry_count:
            self._append_log(f"开始复跑失败项… events={retry_count}")
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
        self.retry_failed_button.setEnabled(self._failed_events_report_path() is not None and self._failed_events_report_path().exists())
        self.pending_retry_event_ids = None
        self.worker = None

    def _update_run_progress(self, current: int, total: int, message: str) -> None:
        total = max(total, 1)
        value = int(current * 100 / total)
        self.run_progress_bar.setValue(max(0, min(100, value)))
        self.run_progress_label.setText(message)

    def _on_run_finished(self, results: list[EventJobResult]) -> None:
        self.run_progress_bar.setValue(100)
        self.run_progress_label.setText("处理完成")
        style_hits = sum(1 for item in results if item.style_result and item.style_result.matched)
        review_hits = sum(1 for item in results if item.style_result and item.style_result.review_required)
        text_hits = sum(1 for item in results if item.text_result and item.text_result.matched)
        no_subtitle = sum(1 for item in results if item.text_result and not item.text_result.matched)
        failed_events = [item for item in results if item.error_messages]
        usage_summary = self._summarize_results_usage(results)
        self._append_log(
            f"处理完成。events={len(results)} style_hits={style_hits} review_hits={review_hits} text_hits={text_hits} no_subtitle={no_subtitle} failed={len(failed_events)} {usage_summary} 输出={self.ass_output_edit.text().strip()}"
        )
        if review_hits:
            self._append_log(f"其中 {review_hits} 条样式已标记为“需核查”。")
        if no_subtitle:
            self._append_log(f"其中 {no_subtitle} 条文字识别结果为未识别到字幕。")
        if failed_events:
            report_path = self._write_failed_events_report(failed_events)
            self.retry_failed_button.setEnabled(bool(report_path))
            self._append_log(f"其中 {len(failed_events)} 条字幕处理失败，已跳过并继续后续任务。")
            if report_path:
                self._append_log(f"失败清单已导出：{report_path}")
            for item in failed_events[:5]:
                self._append_log(f"失败 {item.event_id}: {' | '.join(item.error_messages)}")
        else:
            self.retry_failed_button.setEnabled(False)
        QMessageBox.information(self, WINDOW_TITLE, "处理完成")

    def _on_run_failed(self, message: str) -> None:
        self.run_progress_label.setText("处理失败")
        self._append_log(f"处理失败：{message}")
        QMessageBox.critical(self, WINDOW_TITLE, message)

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
        report_path = self._failed_events_report_path()
        if report_path is None or not report_path.exists():
            QMessageBox.warning(self, WINDOW_TITLE, "未找到失败事件清单，请先完成一次包含失败项的运行。")
            return
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            event_ids = {str(item.get("event_id", "")).strip() for item in data if str(item.get("event_id", "")).strip()}
        except Exception as exc:
            QMessageBox.critical(self, WINDOW_TITLE, f"读取失败事件清单失败：{exc}")
            return
        if not event_ids:
            QMessageBox.warning(self, WINDOW_TITLE, "失败事件清单为空，暂无可复跑项。")
            return
        self.pending_retry_event_ids = event_ids
        self._append_log(f"准备复跑失败项：{len(event_ids)} 条")
        self._start_run()

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
        self.region_start_spin.setValue(int(data.get("subtitle_region_start", 66)))
        self.region_end_spin.setValue(int(data.get("subtitle_region_end", 100)))
        subtitle_language = data.get("subtitle_language", "auto")
        language_index = self.subtitle_language_combo.findData(subtitle_language)
        self.subtitle_language_combo.setCurrentIndex(language_index if language_index >= 0 else 0)
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
        self.include_samples_check.setChecked(data.get("include_samples", False))
        if self.ass_input_edit.text().strip():
            try:
                self._load_subtitle_events()
            except Exception as exc:
                self._append_log(f"字幕事件加载失败：{exc}")
        self._refresh_video_preview()

    def _save_settings(self) -> None:
        self.settings_store.save(
            {
                "gvs2": {
                    "video_path": self.video_edit.text().strip(),
                    "ass_input_path": self.ass_input_edit.text().strip(),
                    "ass_output_path": self.ass_output_edit.text().strip(),
                    "subtitle_region_start": self.region_start_spin.value(),
                    "subtitle_region_end": self.region_end_spin.value(),
                    "subtitle_language": self.subtitle_language_combo.currentData() or "auto",
                    "preview_time_ratio": self.preview_time_slider.value(),
                    "style_description_text": self.style_description_edit.toPlainText(),
                    "style_compact_text": self.style_compact_edit.text().strip(),
                    "style_profiles": self._dump_style_rows(),
                    "include_samples": self.include_samples_check.isChecked(),
                    "style_job": self.style_job_widget.dump_settings(),
                    "text_job": self.text_job_widget.dump_settings(),
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
