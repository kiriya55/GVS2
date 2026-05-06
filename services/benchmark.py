from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, List


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


def _strip_ass_formatting(text: str) -> str:
    """Remove ASS override tags like {\\b1} from text."""
    import re
    return re.sub(r"\{[^}]*\}", "", text).replace("\\N", "\n").strip()


def extract_benchmark_samples(
    ground_truth_path: str,
    result_path: str,
    *,
    llm_name: str = "",
    model_name: str = "",
    time_tolerance_ms: int = 500,
) -> List[BenchmarkSample]:
    """Compare two ASS files event-by-event to produce benchmark samples.

    Matches events by start time within *time_tolerance_ms*.  The ground-truth
    file provides ``expected_*`` values; the result file provides ``actual_*``.
    """
    from services.subtitle_parser import parse_ass

    gt_doc = parse_ass(ground_truth_path)
    rs_doc = parse_ass(result_path)

    gt_events = sorted(gt_doc.events, key=lambda e: e.start_ms)
    rs_events = sorted(rs_doc.events, key=lambda e: e.start_ms)

    samples: list[BenchmarkSample] = []
    rs_index = 0

    for gt in gt_events:
        best = None
        best_diff = time_tolerance_ms + 1
        # advance result pointer close to gt start
        while rs_index < len(rs_events) - 1 and rs_events[rs_index].end_ms < gt.start_ms - time_tolerance_ms:
            rs_index += 1
        # scan nearby results
        for j in range(max(0, rs_index - 1), min(len(rs_events), rs_index + 3)):
            diff = abs(rs_events[j].start_ms - gt.start_ms)
            if diff < best_diff:
                best_diff = diff
                best = rs_events[j]
        if best is None or best_diff > time_tolerance_ms:
            continue

        samples.append(
            BenchmarkSample(
                sample_id=f"event-{gt.index + 1}",
                expected_style_id=None,
                expected_text=_strip_ass_formatting(gt.text),
                actual_style_id=None,
                actual_text=_strip_ass_formatting(best.text),
                llm_name=llm_name,
                model_name=model_name,
            )
        )

    return samples
