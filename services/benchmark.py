from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable


@dataclass(slots=True)
class BenchmarkSample:
    sample_id: str
    expected_style_id: int | None
    expected_text: str
    actual_style_id: int | None
    actual_text: str
    llm_name: str
    model_name: str
    style_input_tokens: int = 0
    style_output_tokens: int = 0
    text_input_tokens: int = 0
    text_output_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass(slots=True)
class BenchmarkSummary:
    label: str
    total_samples: int
    style_accuracy: float
    text_exact_accuracy: float
    text_similarity_average: float
    total_input_tokens: int
    total_output_tokens: int
    average_input_tokens: float
    average_output_tokens: float
    estimated_cost_usd: float


class BenchmarkScorer:
    def _normalize(self, text: str) -> str:
        return "".join(text.split()).lower()

    def _text_similarity(self, expected: str, actual: str) -> float:
        left = self._normalize(expected)
        right = self._normalize(actual)
        if not left and not right:
            return 1.0
        if not left or not right:
            return 0.0
        return SequenceMatcher(None, left, right).ratio()

    def summarize(self, samples: Iterable[BenchmarkSample], use_custom_name: bool = True) -> list[BenchmarkSummary]:
        grouped: dict[str, list[BenchmarkSample]] = {}
        for sample in samples:
            label = sample.llm_name if use_custom_name and sample.llm_name else sample.model_name
            grouped.setdefault(label, []).append(sample)

        summaries: list[BenchmarkSummary] = []
        for label, items in grouped.items():
            total = len(items)
            style_hits = 0
            text_exact_hits = 0
            similarity_sum = 0.0
            total_input_tokens = 0
            total_output_tokens = 0
            total_cost = 0.0
            for item in items:
                if item.expected_style_id is not None and item.expected_style_id == item.actual_style_id:
                    style_hits += 1
                similarity = self._text_similarity(item.expected_text, item.actual_text)
                similarity_sum += similarity
                if self._normalize(item.expected_text) == self._normalize(item.actual_text):
                    text_exact_hits += 1
                total_input_tokens += item.style_input_tokens + item.text_input_tokens
                total_output_tokens += item.style_output_tokens + item.text_output_tokens
                total_cost += item.estimated_cost_usd
            summaries.append(
                BenchmarkSummary(
                    label=label,
                    total_samples=total,
                    style_accuracy=style_hits / total if total else 0.0,
                    text_exact_accuracy=text_exact_hits / total if total else 0.0,
                    text_similarity_average=similarity_sum / total if total else 0.0,
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    average_input_tokens=total_input_tokens / total if total else 0.0,
                    average_output_tokens=total_output_tokens / total if total else 0.0,
                    estimated_cost_usd=total_cost,
                )
            )
        return sorted(summaries, key=lambda item: item.label.lower())

    def format_table(self, summaries: Iterable[BenchmarkSummary]) -> str:
        rows = [
            [
                "LLM",
                "Samples",
                "StyleAcc",
                "TextExact",
                "TextSimAvg",
                "InTok",
                "OutTok",
                "AvgIn",
                "AvgOut",
                "CostUSD",
            ]
        ]
        for summary in summaries:
            rows.append(
                [
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
            )
        widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
        formatted = []
        for idx, row in enumerate(rows):
            formatted.append(" | ".join(value.ljust(widths[i]) for i, value in enumerate(row)))
            if idx == 0:
                formatted.append("-+-".join("-" * width for width in widths))
        return "\n".join(formatted)
