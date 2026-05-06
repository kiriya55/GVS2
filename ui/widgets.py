from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from models.style_profile import StyleProfile
from providers.base import ProviderConfig
from providers.factory import build_provider
from providers.base import ProviderResponse
from services.image_preprocess import ImageEncodingOptions


IMAGE_FORMATS = {
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}
LAYOUT_HINTS = ["either", "single", "double"]
SUBTITLE_LANGUAGES = [
    ("auto", "自动识别"),
    ("zh-Hans", "简体中文"),
    ("zh-Hant", "繁體中文"),
    ("ja", "日语"),
    ("en", "英语"),
    ("ko", "韩语"),
    ("mixed", "中日英韩混合"),
]
STYLE_KEYWORDS_ZH = ["字幕", "描边", "边框", "填充", "红", "黄", "白", "黑", "蓝", "双行", "单行", "阴影", "加粗", "字体"]
ENV_DEFAULTS = {
    "openai": {
        "base_url": "GVS2_OPENAI_BASE_URL",
        "api_key": "GVS2_OPENAI_API_KEY",
        "model": "GVS2_OPENAI_MODEL",
    },
    "anthropic": {
        "base_url": "GVS2_ANTHROPIC_BASE_URL",
        "api_key": "GVS2_ANTHROPIC_API_KEY",
        "model": "GVS2_ANTHROPIC_MODEL",
    },
}

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


def browse_row(edit: QLineEdit, handler) -> QWidget:
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.addWidget(edit)
    button = QPushButton("浏览")
    button.clicked.connect(handler)
    row.addWidget(button)
    return container


def _read_env(name: str) -> str:
    return os.environ.get(name, "").strip()


@dataclass(slots=True)
class RunPayload:
    video_path: str
    ass_input_path: str
    ass_output_path: str
    style_profiles: list[StyleProfile]
    style_provider: ProviderConfig | None
    text_provider: ProviderConfig | None
    subtitle_region_start: int
    subtitle_region_end: int
    subtitle_language: str
    style_image_options: ImageEncodingOptions
    text_image_options: ImageEncodingOptions
    event_ids: set[str] | None = None
    include_samples: bool = False


class PipelineWorker(QThread):
    finished_ok = Signal(list)
    failed = Signal(str)
    progress_changed = Signal(int, int, str)

    def __init__(self, payload: RunPayload) -> None:
        super().__init__()
        self.payload = payload

    def run(self) -> None:
        from pipeline.runner import run_gvs2

        try:
            results = run_gvs2(
                video_path=self.payload.video_path,
                ass_input_path=self.payload.ass_input_path,
                ass_output_path=self.payload.ass_output_path,
                style_profiles=self.payload.style_profiles,
                style_provider=self.payload.style_provider,
                text_provider=self.payload.text_provider,
                subtitle_region_start=self.payload.subtitle_region_start,
                subtitle_region_end=self.payload.subtitle_region_end,
                subtitle_language=self.payload.subtitle_language,
                style_image_options=self.payload.style_image_options,
                text_image_options=self.payload.text_image_options,
                progress_callback=self.progress_changed.emit,
                event_ids=self.payload.event_ids,
                include_samples=self.payload.include_samples,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(results)


class StylePrepWorker(QThread):
    finished_text = Signal(object)
    failed = Signal(str)

    def __init__(self, provider_config: ProviderConfig, prompt: str, images: list[tuple[str, str]]) -> None:
        super().__init__()
        self.provider_config = provider_config
        self.prompt = prompt
        self.images = images

    def run(self) -> None:
        try:
            provider = build_provider(self.provider_config)
            result = provider.classify(self.prompt, self.images)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished_text.emit(result)


class ProviderJobWidget(QGroupBox):
    def __init__(self, title: str, *, enabled: bool, default_model: str, default_max_edge: int, default_max_output_tokens: int = 256) -> None:
        super().__init__(title)
        self._history: list[dict] = []

        self.enabled_checkbox = QCheckBox("启用")
        self.enabled_checkbox.setChecked(enabled)

        self.history_combo = QComboBox()
        self.history_combo.setMinimumWidth(180)
        self.history_combo.setPlaceholderText("历史记录…")
        self.history_combo.currentIndexChanged.connect(self._apply_history_selection)
        self.save_history_button = QPushButton("保存当前")
        self.save_history_button.setToolTip("将当前 Provider 配置保存到历史记录")
        self.save_history_button.clicked.connect(self._save_current_to_history)
        self.delete_history_button = QPushButton("删除")
        self.delete_history_button.setToolTip("删除选中的历史记录")
        self.delete_history_button.clicked.connect(self._delete_selected_history)

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["openai", "anthropic"])

        self.model_edit = QLineEdit(default_model)
        self.base_url_edit = QLineEdit()
        self.api_key_edit = QLineEdit()
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1, 8192)
        self.max_tokens_spin.setValue(default_max_output_tokens)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(1, 600)
        self.timeout_spin.setValue(180)
        self.concurrency_spin = QSpinBox()
        self.concurrency_spin.setRange(1, 8)
        self.concurrency_spin.setValue(1)
        self.image_format_combo = QComboBox()
        self.image_format_combo.addItems(["JPEG", "WEBP"])
        self.max_edge_spin = QSpinBox()
        self.max_edge_spin.setRange(64, 4096)
        self.max_edge_spin.setValue(default_max_edge)
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(75)
        self.disable_thinking_check = QCheckBox("禁用推理模式")
        self.disable_thinking_check.setToolTip("某些模型默认启用推理思考链，会产生大量推理token，启用此选项可禁用推理模式")

        # ---------- main layout ----------
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        layout.addWidget(self.enabled_checkbox)

        history_row = QHBoxLayout()
        history_row.addWidget(QLabel("历史"))
        history_row.addWidget(self.history_combo, 1)
        history_row.addWidget(self.save_history_button)
        history_row.addWidget(self.delete_history_button)
        layout.addLayout(history_row)

        basic_form = QFormLayout()
        basic_form.addRow("接口类型", self.provider_combo)
        basic_form.addRow("模型", self.model_edit)
        basic_form.addRow("Base URL", self.base_url_edit)
        basic_form.addRow("API Key", self.api_key_edit)
        basic_form.addRow("最大输出 token", self.max_tokens_spin)
        layout.addLayout(basic_form)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setText("高级选项")
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.RightArrow)
        layout.addWidget(self.advanced_toggle)

        self.advanced_widget = QWidget()
        adv_form = QFormLayout(self.advanced_widget)
        adv_form.setContentsMargins(0, 0, 0, 0)
        adv_form.addRow("超时（秒）", self.timeout_spin)
        adv_form.addRow("并发数", self.concurrency_spin)
        adv_form.addRow("图片格式", self.image_format_combo)
        adv_form.addRow("最大边长", self.max_edge_spin)
        adv_form.addRow("图片质量", self.quality_spin)
        adv_form.addRow(self.disable_thinking_check)
        self.advanced_widget.setVisible(False)
        layout.addWidget(self.advanced_widget)

        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        self.enabled_checkbox.toggled.connect(self._apply_enabled_state)
        self.provider_combo.currentTextChanged.connect(self._apply_provider_defaults)
        self._apply_provider_defaults(self.provider_combo.currentText())
        self._apply_enabled_state(self.enabled_checkbox.isChecked())

    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_widget.setVisible(checked)
        self.advanced_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def _apply_enabled_state(self, enabled: bool) -> None:
        for widget in [
            self.provider_combo,
            self.model_edit,
            self.base_url_edit,
            self.api_key_edit,
            self.max_tokens_spin,
            self.advanced_toggle,
            self.advanced_widget,
        ]:
            widget.setEnabled(enabled)

    def _apply_provider_defaults(self, provider_type: str) -> None:
        env = ENV_DEFAULTS[provider_type]
        env_model = _read_env(env["model"])
        env_base_url = _read_env(env["base_url"])
        env_api_key = _read_env(env["api_key"])

        if provider_type == "anthropic":
            if not self.model_edit.text().strip() or self.model_edit.text().startswith("gpt-"):
                self.model_edit.setText(env_model or "claude-opus-4-7")
            self.base_url_edit.setPlaceholderText(env_base_url or "例如 https://api.anthropic.com")
            self.model_edit.setPlaceholderText("例如 claude-opus-4-7")
        else:
            if not self.model_edit.text().strip() or self.model_edit.text().startswith("claude-"):
                self.model_edit.setText(env_model or "gpt-4o")
            self.base_url_edit.setPlaceholderText(env_base_url or "例如 https://api.openai.com/v1")
            self.model_edit.setPlaceholderText("例如 gpt-4o")

        self.api_key_edit.setPlaceholderText("例如 sk-... 或对应平台 API Key")
        if not self.base_url_edit.text().strip() and env_base_url:
            self.base_url_edit.setText(env_base_url)
        if not self.api_key_edit.text().strip() and env_api_key:
            self.api_key_edit.setText(env_api_key)

    # ---- provider history ----

    def _history_label(self, entry: dict) -> str:
        ptype = entry.get("provider_type", "?")
        model = entry.get("model", "?")
        return f"{ptype} / {model}"

    def _refresh_history_combo(self) -> None:
        self.history_combo.blockSignals(True)
        self.history_combo.clear()
        for entry in self._history:
            self.history_combo.addItem(self._history_label(entry))
        self.history_combo.setCurrentIndex(-1)
        self.history_combo.blockSignals(False)

    def _apply_history_selection(self, index: int) -> None:
        if index < 0 or index >= len(self._history):
            return
        entry = self._history[index]
        self.provider_combo.setCurrentText(entry.get("provider_type", "openai"))
        self.model_edit.setText(entry.get("model", ""))
        self.base_url_edit.setText(entry.get("base_url", ""))
        self.api_key_edit.setText(entry.get("api_key", ""))

    def _save_current_to_history(self) -> None:
        entry = {
            "provider_type": self.provider_combo.currentText(),
            "model": self.model_edit.text().strip(),
            "base_url": self.base_url_edit.text().strip(),
            "api_key": self.api_key_edit.text().strip(),
        }
        if not entry["model"]:
            QMessageBox.warning(self, self.title(), "请先填写模型名称再保存。")
            return
        for existing in self._history:
            if (existing.get("provider_type") == entry["provider_type"]
                    and existing.get("model") == entry["model"]
                    and existing.get("base_url") == entry["base_url"]):
                existing["api_key"] = entry["api_key"]
                self._refresh_history_combo()
                return
        self._history.append(entry)
        self._refresh_history_combo()

    def _delete_selected_history(self) -> None:
        index = self.history_combo.currentIndex()
        if index < 0 or index >= len(self._history):
            return
        self._history.pop(index)
        self._refresh_history_combo()

    def load_history(self, history: list[dict]) -> None:
        self._history = list(history)
        self._refresh_history_combo()

    def dump_history(self) -> list[dict]:
        return [dict(h) for h in self._history]

    def build_provider_config(self) -> ProviderConfig | None:
        if not self.enabled_checkbox.isChecked():
            return None
        provider_type = self.provider_combo.currentText()
        model = self.model_edit.text().strip()
        api_key = self.api_key_edit.text().strip()
        base_url = self.base_url_edit.text().strip()
        if not model:
            raise ValueError(f"{self.title()} 的 model 不能为空")
        if not api_key:
            raise ValueError(f"{self.title()} 的 API key 不能为空")
        if provider_type == "openai" and not base_url:
            raise ValueError(f"{self.title()} 使用 openai 接口时必须填写 Base URL")
        return ProviderConfig(
            provider_type=provider_type,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_sec=self.timeout_spin.value(),
            max_output_tokens=self.max_tokens_spin.value(),
            concurrency=self.concurrency_spin.value(),
            extra_params={"disable_thinking": self.disable_thinking_check.isChecked()},
        )

    def build_image_options(self) -> ImageEncodingOptions:
        format_name = self.image_format_combo.currentText()
        return ImageEncodingOptions(
            format_name=format_name,
            mime_type=IMAGE_FORMATS[format_name],
            quality=self.quality_spin.value(),
            max_edge=self.max_edge_spin.value(),
        )

    def load_settings(self, data: dict) -> None:
        self.enabled_checkbox.setChecked(data.get("enabled", self.enabled_checkbox.isChecked()))
        provider_type = data.get("provider_type")
        if provider_type:
            index = self.provider_combo.findText(provider_type)
            if index >= 0:
                self.provider_combo.setCurrentIndex(index)
        self.model_edit.setText(data.get("model", self.model_edit.text()))
        self.base_url_edit.setText(data.get("base_url", ""))
        self.api_key_edit.setText(data.get("api_key", ""))
        self.max_tokens_spin.setValue(int(data.get("max_output_tokens", self.max_tokens_spin.value())))
        self.timeout_spin.setValue(int(data.get("timeout_sec", self.timeout_spin.value())))
        self.concurrency_spin.setValue(int(data.get("concurrency", self.concurrency_spin.value())))
        format_name = data.get("image_format")
        if format_name:
            index = self.image_format_combo.findText(format_name)
            if index >= 0:
                self.image_format_combo.setCurrentIndex(index)
        self.max_edge_spin.setValue(int(data.get("max_edge", self.max_edge_spin.value())))
        self.quality_spin.setValue(int(data.get("quality", self.quality_spin.value())))
        self.disable_thinking_check.setChecked(data.get("disable_thinking", False))
        self._apply_enabled_state(self.enabled_checkbox.isChecked())

    def dump_settings(self) -> dict:
        return {
            "enabled": self.enabled_checkbox.isChecked(),
            "provider_type": self.provider_combo.currentText(),
            "model": self.model_edit.text().strip(),
            "base_url": self.base_url_edit.text().strip(),
            "api_key": self.api_key_edit.text().strip(),
            "max_output_tokens": self.max_tokens_spin.value(),
            "timeout_sec": self.timeout_spin.value(),
            "concurrency": self.concurrency_spin.value(),
            "image_format": self.image_format_combo.currentText(),
            "max_edge": self.max_edge_spin.value(),
            "quality": self.quality_spin.value(),
            "disable_thinking": self.disable_thinking_check.isChecked(),
        }


class ApiSettingsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("API 设置")
        self.resize(980, 560)
        layout = QVBoxLayout(self)

        jobs_layout = QHBoxLayout()
        self.style_job_widget = ProviderJobWidget("样式识别任务", enabled=True, default_model="claude-opus-4-7", default_max_edge=640, default_max_output_tokens=32)
        self.text_job_widget = ProviderJobWidget("文字提取任务", enabled=False, default_model="claude-opus-4-7", default_max_edge=768, default_max_output_tokens=256)
        jobs_layout.addWidget(self.style_job_widget)
        jobs_layout.addWidget(self.text_job_widget)
        layout.addLayout(jobs_layout)

        hint = QLabel("支持窗口直接输入配置并保存到 config.json；OpenAI-compatible 请填写完整 Base URL，例如 https://api.openai.com/v1；Anthropic 可填写 https://api.anthropic.com。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QHBoxLayout()
        self.save_button = QPushButton("保存并关闭")
        self.save_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

    def load_settings(self, style_job: dict, text_job: dict) -> None:
        self.style_job_widget.load_settings(style_job)
        self.text_job_widget.load_settings(text_job)

    def dump_settings(self) -> tuple[dict, dict]:
        return self.style_job_widget.dump_settings(), self.text_job_widget.dump_settings()

    def load_history(self, style_history: list[dict], text_history: list[dict]) -> None:
        self.style_job_widget.load_history(style_history)
        self.text_job_widget.load_history(text_history)

    def dump_history(self) -> tuple[list[dict], list[dict]]:
        return self.style_job_widget.dump_history(), self.text_job_widget.dump_history()
