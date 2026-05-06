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
from services.benchmark import BenchmarkScorer, extract_benchmark_samples
from storage.settings_store import SettingsStore
from ui.widgets import CONFIG_PATH, browse_row


WINDOW_TITLE = "GVS2 Benchmark"


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

        ass_compare_box = QGroupBox("ASS 对比导入")
        ass_compare_form = QFormLayout()
        self.gt_ass_edit = QLineEdit()
        self.result_ass_edit = QLineEdit()
        ass_compare_form.addRow("标准答案 ASS", self._with_browse(self.gt_ass_edit, self._browse_gt_ass))
        ass_compare_form.addRow("识别结果 ASS", self._with_browse(self.result_ass_edit, self._browse_result_ass))
        ass_llm_row = QHBoxLayout()
        self.ass_llm_name_edit = QLineEdit()
        self.ass_llm_name_edit.setPlaceholderText("自定义名称（可选）")
        self.ass_model_name_edit = QLineEdit()
        self.ass_model_name_edit.setPlaceholderText("模型名（可选）")
        ass_llm_row.addWidget(self.ass_llm_name_edit)
        ass_llm_row.addWidget(self.ass_model_name_edit)
        ass_compare_form.addRow("LLM / 模型", ass_llm_row)
        self.extract_button = QPushButton("提取对比样本")
        self.extract_button.clicked.connect(self._extract_from_ass)
        ass_compare_form.addRow(self.extract_button)
        ass_compare_box.setLayout(ass_compare_form)
        layout.addWidget(ass_compare_box)

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
        return browse_row(edit, handler)

    def _browse_samples(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 benchmark JSON", "", "JSON files (*.json);;All files (*)")
        if path:
            self.samples_path_edit.setText(path)

    def _browse_gt_ass(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择标准答案 ASS", "", "ASS files (*.ass);;All files (*)")
        if path:
            self.gt_ass_edit.setText(path)

    def _browse_result_ass(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择识别结果 ASS", "", "ASS files (*.ass);;All files (*)")
        if path:
            self.result_ass_edit.setText(path)

    def _get_samples_path(self) -> str:
        path = self.samples_path_edit.text().strip()
        if not path:
            raise ValueError("请先选择 benchmark 样本 JSON")
        if not Path(path).exists():
            raise ValueError("样本 JSON 不存在")
        return path

    def _extract_from_ass(self) -> None:
        gt_path = self.gt_ass_edit.text().strip()
        result_path = self.result_ass_edit.text().strip()
        if not gt_path or not result_path:
            QMessageBox.warning(self, WINDOW_TITLE, "请先选择标准答案和识别结果两个 ASS 文件。")
            return
        if not Path(gt_path).exists():
            QMessageBox.critical(self, WINDOW_TITLE, f"标准答案文件不存在：{gt_path}")
            return
        if not Path(result_path).exists():
            QMessageBox.critical(self, WINDOW_TITLE, f"识别结果文件不存在：{result_path}")
            return
        try:
            samples = extract_benchmark_samples(
                gt_path,
                result_path,
                llm_name=self.ass_llm_name_edit.text().strip(),
                model_name=self.ass_model_name_edit.text().strip(),
            )
        except Exception as exc:
            QMessageBox.critical(self, WINDOW_TITLE, f"提取失败：{exc}")
            return
        if not samples:
            QMessageBox.warning(self, WINDOW_TITLE, "未能匹配到任何字幕事件，请检查两个 ASS 文件的时间轴。")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "保存提取的样本", "benchmark_extracted.json", "JSON files (*.json)")
        if not save_path:
            return
        data = [
            {
                "sample_id": s.sample_id,
                "expected_style_id": s.expected_style_id,
                "expected_text": s.expected_text,
                "actual_style_id": s.actual_style_id,
                "actual_text": s.actual_text,
                "llm_name": s.llm_name,
                "model_name": s.model_name,
            }
            for s in samples
        ]
        Path(save_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.samples_path_edit.setText(save_path)
        self.preview_edit.setPlainText(json.dumps(data, ensure_ascii=False, indent=2))
        QMessageBox.information(self, WINDOW_TITLE, f"已提取 {len(samples)} 条对比样本，已保存并加载。")
        self._save_settings()

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
        self.gt_ass_edit.setText(data.get("gt_ass_path", ""))
        self.result_ass_edit.setText(data.get("result_ass_path", ""))
        self.ass_llm_name_edit.setText(data.get("ass_llm_name", ""))
        self.ass_model_name_edit.setText(data.get("ass_model_name", ""))
        use_custom_name = data.get("use_custom_name", True)
        self.custom_name_radio.setChecked(bool(use_custom_name))
        self.model_name_radio.setChecked(not bool(use_custom_name))

    def _save_settings(self) -> None:
        self.settings_store.save(
            {
                "benchmark": {
                    "samples_path": self.samples_path_edit.text().strip(),
                    "use_custom_name": self.custom_name_radio.isChecked(),
                    "gt_ass_path": self.gt_ass_edit.text().strip(),
                    "result_ass_path": self.result_ass_edit.text().strip(),
                    "ass_llm_name": self.ass_llm_name_edit.text().strip(),
                    "ass_model_name": self.ass_model_name_edit.text().strip(),
                }
            }
        )


def launch() -> int:
    app = QApplication.instance() or QApplication([])
    window = BenchmarkWindow()
    window.show()
    return app.exec()
