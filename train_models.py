"""Benchmark five CPU-friendly text classifiers for the selected aspects."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "1000_ds_Sentiment_analysis.csv"
OUTPUT_DIR = ROOT / "outputs"
MODELS_PATH = OUTPUT_DIR / "model_benchmark.joblib"
BEST_MODELS_PATH = OUTPUT_DIR / "model_best.joblib"
METRICS_PATH = OUTPUT_DIR / "model_benchmark_metrics.json"

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
    labels = {part.strip().lower() for part in str(value).split(",") if part.strip()}
    if not labels:
        return ""
    return "mixed_neutral" if len(labels) > 1 or labels & {"mixed", "neutral"} else next(iter(labels))


def make_pipeline(classifier: object) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("classifier", classifier),
        ]
    )


def model_factories() -> dict[str, object]:
    return {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=42
        ),
        "linear_svm": lambda: LinearSVC(class_weight="balanced", random_state=42),
        "complement_naive_bayes": lambda: ComplementNB(alpha=0.5),
        "multinomial_naive_bayes": lambda: MultinomialNB(alpha=0.5),
        "sgd": lambda: SGDClassifier(
            loss="modified_huber", max_iter=1000, class_weight="balanced", random_state=42
        ),
        "ridge": lambda: RidgeClassifier(class_weight="balanced"),
        "passive_aggressive": lambda: SGDClassifier(
            loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0,
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "bernoulli_naive_bayes": lambda: BernoulliNB(alpha=0.5),
    }


def prepare_split(data: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, dict[str, int]]:
    labeled = data[["review", column]].copy()
    labeled["label"] = labeled[column].map(normalize_label)
    labeled = labeled[labeled["label"].ne("")]
    counts = labeled["label"].value_counts()
    stratify = labeled["label"] if len(counts) > 1 and counts.min() >= 2 else None
    train, test = train_test_split(
        labeled,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )
    return (
        train["review"],
        test["review"],
        train["label"],
        test["label"],
        {key: int(value) for key, value in counts.items()},
    )


def train_single_model(model_name: str, output_number: int) -> None:
    data = pd.read_csv(INPUT_PATH, low_memory=False)
    factory = model_factories()[model_name]
    models: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, object]] = {}

    for aspect, column in SELECTED_ASPECTS.items():
        train_text, test_text, train_labels, test_labels, class_counts = prepare_split(data, column)
        started = time.perf_counter()
        model = make_pipeline(factory())
        model.fit(train_text, train_labels)
        predictions = model.predict(test_text)
        elapsed = round(time.perf_counter() - started, 3)
        metrics[aspect] = {
            "model": model_name,
            "labeled_rows": int(len(train_labels) + len(test_labels)),
            "train_rows": int(len(train_labels)),
            "test_rows": int(len(test_labels)),
            "class_counts": class_counts,
            "accuracy": round(float(accuracy_score(test_labels, predictions)), 4),
            "macro_f1": round(float(f1_score(test_labels, predictions, average="macro", zero_division=0)), 4),
            "weighted_f1": round(float(f1_score(test_labels, predictions, average="weighted", zero_division=0)), 4),
            "seconds": elapsed,
        }
        models[aspect] = model
        print(f"{aspect}: macro F1={metrics[aspect]['macro_f1']:.3f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = OUTPUT_DIR / f"model_{output_number}.joblib"
    metrics_path = OUTPUT_DIR / f"model_{output_number}_metrics.json"
    joblib.dump(models, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Model written to: {model_path}")
    print(f"Metrics written to: {metrics_path}")


def main() -> None:
    data = pd.read_csv(INPUT_PATH, low_memory=False)
    factories = model_factories()
    trained_models: dict[str, dict[str, Pipeline]] = {name: {} for name in factories}
    metrics: dict[str, dict[str, dict[str, object]]] = {name: {} for name in factories}
    best_models: dict[str, Pipeline] = {}

    for aspect, column in SELECTED_ASPECTS.items():
        train_text, test_text, train_labels, test_labels, class_counts = prepare_split(data, column)
        aspect_results = []
        for model_name, factory in factories.items():
            started = time.perf_counter()
            model = make_pipeline(factory())
            model.fit(train_text, train_labels)
            predictions = model.predict(test_text)
            elapsed = round(time.perf_counter() - started, 3)
            result = {
                "labeled_rows": int(len(train_labels) + len(test_labels)),
                "train_rows": int(len(train_labels)),
                "test_rows": int(len(test_labels)),
                "class_counts": class_counts,
                "accuracy": round(float(accuracy_score(test_labels, predictions)), 4),
                "macro_f1": round(float(f1_score(test_labels, predictions, average="macro", zero_division=0)), 4),
                "weighted_f1": round(float(f1_score(test_labels, predictions, average="weighted", zero_division=0)), 4),
                "seconds": elapsed,
            }
            trained_models[model_name][aspect] = model
            metrics[model_name][aspect] = result
            aspect_results.append((result["macro_f1"], model_name))

        _, best_name = max(aspect_results)
        best_models[aspect] = trained_models[best_name][aspect]
        best_score = metrics[best_name][aspect]["macro_f1"]
        print(f"{aspect}: best={best_name}, macro F1={best_score:.3f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    joblib.dump(trained_models, MODELS_PATH)
    joblib.dump(best_models, BEST_MODELS_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"All models written to: {MODELS_PATH}")
    print(f"Best aspect models written to: {BEST_MODELS_PATH}")
    print(f"Metrics written to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
