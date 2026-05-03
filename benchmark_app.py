from __future__ import annotations

import json
from pathlib import Path

from services.benchmark import BenchmarkSample, BenchmarkScorer

__all__ = ["load_samples", "main", "launch"]


def load_samples(path: str) -> list[BenchmarkSample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [BenchmarkSample(**item) for item in data]


def main(samples_path: str) -> None:
    scorer = BenchmarkScorer()
    samples = load_samples(samples_path)
    summaries = scorer.summarize(samples)
    print(scorer.format_table(summaries))


def launch() -> int:
    from ui.benchmark_window import launch as launch_ui

    return launch_ui()


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        raise SystemExit(launch())
