from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from benchmark_app import load_samples
from services.benchmark import BenchmarkScorer
from storage.settings_store import SettingsStore


WINDOW_TITLE = "GVS2 Benchmark"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


class BenchmarkWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(840, 680)
        self.settings_store = SettingsStore(str(CONFIG_PATH))
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        file_box = QGroupBox("样本文件")
        file_form = QFormLayout()
        self.samples_path_edit = QLineEdit()
        file_form.addRow("JSON", self._with_browse(self.samples_path_edit, self._browse_samples))
        file_box.setLayout(file_form)
        layout.addWidget(file_box)

        options_box = QGroupBox("分组方式")
        options_layout = QHBoxLayout(options_box)
        self.custom_name_radio = QRadioButton("优先使用 llm_name")
        self.model_name_radio = QRadioButton("按 model_name 分组")
        self.custom_name_radio.setChecked(True)
        options_layout.addWidget(self.custom_name_radio)
        options_layout.addWidget(self.model_name_radio)
        layout.addWidget(options_box)

        buttons_layout = QHBoxLayout()
        self.load_button = QPushButton("加载样例 JSON")
        self.load_button.clicked.connect(self._preview_samples)
        self.score_button = QPushButton("生成匹配度表格")
        self.score_button.clicked.connect(self._score_samples)
        buttons_layout.addWidget(self.load_button)
        buttons_layout.addWidget(self.score_button)
        layout.addLayout(buttons_layout)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        self.preview_edit = QPlainTextEdit()
        self.preview_edit.setReadOnly(True)
        left_layout.addWidget(QLabel("样本预览"))
        left_layout.addWidget(self.preview_edit, 1)

        self.table_text = QPlainTextEdit()
        self.table_text.setReadOnly(True)
        left_layout.addWidget(QLabel("表格文本"))
        left_layout.addWidget(self.table_text, 1)

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(["LLM", "Samples", "StyleAcc", "TextExact", "TextSimAvg", "InTok", "OutTok", "AvgIn", "AvgOut", "CostUSD"])
        right_layout.addWidget(QLabel("汇总结果"))
        right_layout.addWidget(self.table, 1)

    def _with_browse(self, edit: QLineEdit, handler) -> QWidget:
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit)
        button = QPushButton("浏览")
        button.clicked.connect(handler)
        row.addWidget(button)
        return container

    def _browse_samples(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 benchmark JSON", "", "JSON files (*.json);;All files (*)")
        if path:
            self.samples_path_edit.setText(path)

    def _get_samples_path(self) -> str:
        path = self.samples_path_edit.text().strip()
        if not path:
            raise ValueError("请先选择 benchmark 样本 JSON")
        if not Path(path).exists():
            raise ValueError("样本 JSON 不存在")
        return path

    def _preview_samples(self) -> None:
        try:
            path = self._get_samples_path()
            raw = Path(path).read_text(encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, WINDOW_TITLE, str(exc))
            return
        self.preview_edit.setPlainText(raw)
        self._save_settings()

    def _score_samples(self) -> None:
        try:
            path = self._get_samples_path()
            samples = load_samples(path)
            scorer = BenchmarkScorer()
            use_custom_name = self.custom_name_radio.isChecked()
            summaries = scorer.summarize(samples, use_custom_name=use_custom_name)
            table_text = scorer.format_table(summaries)
        except Exception as exc:
            QMessageBox.critical(self, WINDOW_TITLE, str(exc))
            return

        self.table.setRowCount(len(summaries))
        for row, summary in enumerate(summaries):
            values = [
                summary.label,
                str(summary.total_samples),
                f"{summary.style_accuracy:.2%}",
                f"{summary.text_exact_accuracy:.2%}",
                f"{summary.text_similarity_average:.2%}",
                str(summary.total_input_tokens),
                str(summary.total_output_tokens),
                f"{summary.average_input_tokens:.1f}",
                f"{summary.average_output_tokens:.1f}",
                f"${summary.estimated_cost_usd:.6f}",
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table_text.setPlainText(table_text)
        self._save_settings()

    def _load_settings(self) -> None:
        data = self.settings_store.load().get("benchmark", {})
        self.samples_path_edit.setText(data.get("samples_path", ""))
        use_custom_name = data.get("use_custom_name", True)
        self.custom_name_radio.setChecked(bool(use_custom_name))
        self.model_name_radio.setChecked(not bool(use_custom_name))

    def _save_settings(self) -> None:
        self.settings_store.save(
            {
                "benchmark": {
                    "samples_path": self.samples_path_edit.text().strip(),
                    "use_custom_name": self.custom_name_radio.isChecked(),
                }
            }
        )


def launch() -> int:
    app = QApplication.instance() or QApplication([])
    window = BenchmarkWindow()
    window.show()
    return app.exec()
