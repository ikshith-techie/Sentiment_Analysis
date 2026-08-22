"""Baseline model: train one TF-IDF classifier per selected aspect.

Run from the project directory:
    py model_1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "1000_ds_Sentiment_analysis.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "model_1.joblib"
METRICS_PATH = OUTPUT_DIR / "model_1_metrics.json"

SELECTED_ASPECTS = {
    "food_quality": "FOOD#QUALITY",
    "ambience": "AMBIENCE#GENERAL",
    "prices": "FOOD#PRICES",
    "location": "LOCATION#GENERAL",
    "general": "RESTAURANT#GENERAL",
}


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""
    labels = {part.strip() for part in str(value).split(",") if part.strip()}
    if not labels:
        return ""
    return next(iter(labels)) if len(labels) == 1 else "mixed"


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train_aspect(data: pd.DataFrame, column: str) -> tuple[Pipeline, dict[str, object]]:
    labeled = data[["review", column]].copy()
    labeled["label"] = labeled[column].map(normalize_label)
    labeled = labeled[labeled["label"].ne("")]
    label_counts = labeled["label"].value_counts()
    stratify = labeled["label"] if label_counts.min() >= 2 else None
    train, test = train_test_split(
        labeled,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )
    model = make_pipeline()
    model.fit(train["review"], train["label"])
    predictions = model.predict(test["review"])
    metrics = {
        "labeled_rows": int(len(labeled)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "class_counts": {key: int(value) for key, value in label_counts.items()},
        "accuracy": round(float(accuracy_score(test["label"], predictions)), 4),
        "macro_f1": round(float(f1_score(test["label"], predictions, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(test["label"], predictions, average="weighted", zero_division=0)), 4),
        "report": classification_report(test["label"], predictions, output_dict=True, zero_division=0),
    }
    return model, metrics


def main() -> None:
    data = pd.read_csv(INPUT_PATH, low_memory=False)
    models = {}
    metrics = {}
    for aspect, column in SELECTED_ASPECTS.items():
        models[aspect], metrics[aspect] = train_aspect(data, column)
        print(
            f"{aspect}: {metrics[aspect]['labeled_rows']} labels, "
            f"macro F1={metrics[aspect]['macro_f1']:.3f}, "
            f"weighted F1={metrics[aspect]['weighted_f1']:.3f}"
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(models, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Models written to: {MODEL_PATH}")
    print(f"Metrics written to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
