"""Combine standalone model metrics into one model-keyed result file."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_NUMBERS = range(1, 10)
RESULT_PATH = OUTPUT_DIR / "result.json"


def main() -> None:
    result = {}
    for number in MODEL_NUMBERS:
        metrics_path = OUTPUT_DIR / f"model_{number}_metrics.json"
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        result[f"model{number}"] = {
            aspect: {
                metric: values[metric]
                for metric in ("accuracy", "macro_f1", "weighted_f1")
            }
            for aspect, values in metrics.items()
        }

    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Combined results written to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
