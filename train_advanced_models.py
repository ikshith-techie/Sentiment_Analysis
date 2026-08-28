"""Fine-tune transformer classifiers for the selected restaurant aspects.

This script is intended for a CUDA-enabled machine. It trains one
sequence-classification model per aspect and writes Hugging Face model
directories below ``outputs/advanced_models``.

Examples:
    py train_advanced_models.py --model distilbert
    py train_advanced_models.py --model roberta --epochs 5
    py train_advanced_models.py --all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "1000_ds_Sentiment_analysis.csv"
OUTPUT_DIR = ROOT / "outputs" / "advanced_models"
SELECTED_ASPECTS = {
    "food_quality": "FOOD#QUALITY",
    "ambience": "AMBIENCE#GENERAL",
    "prices": "FOOD#PRICES",
    "location": "LOCATION#GENERAL",
    "general": "RESTAURANT#GENERAL",
}
MODEL_CHECKPOINTS = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "deberta": "microsoft/deberta-v3-base",
}


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return ""
    labels = {part.strip() for part in str(value).split(",") if part.strip()}
    return next(iter(labels)) if len(labels) == 1 else "mixed" if labels else ""


def prepare_split(data: pd.DataFrame, column: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    labeled = data[["review", column]].copy()
    labeled["label"] = labeled[column].map(normalize_label)
    labeled["review"] = labeled["review"].fillna("").astype(str)
    labeled = labeled[labeled["label"].ne("") & labeled["review"].str.strip().ne("")]
    labels = sorted(labeled["label"].unique())
    counts = labeled["label"].value_counts()
    stratify = labeled["label"] if len(labels) > 1 and counts.min() >= 2 else None
    train, test = train_test_split(
        labeled[["review", "label"]], test_size=0.2, random_state=42, stratify=stratify
    )
    return train.reset_index(drop=True), test.reset_index(drop=True), labels


def train_aspect(
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: list[str],
    checkpoint: str,
    aspect: str,
    epochs: float,
    batch_size: int,
    max_length: int,
    seed: int,
) -> dict[str, object]:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers import DataCollatorWithPadding, Trainer, TrainingArguments, set_seed
    except ImportError as error:
        raise SystemExit(
            "GPU model dependencies are not installed. On the college server, "
            "install the CUDA-compatible PyTorch build, then run: "
            "py -m pip install -r requirements.txt"
        ) from error

    set_seed(seed)
    label_to_id = {label: index for index, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    class ReviewDataset(torch.utils.data.Dataset):
        def __init__(self, frame: pd.DataFrame) -> None:
            self.encodings = tokenizer(
                frame["review"].tolist(), truncation=True, max_length=max_length
            )
            self.labels = [label_to_id[label] for label in frame["label"]]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, object]:
            item = {key: value[index] for key, value in self.encodings.items()}
            item["labels"] = self.labels[index]
            return item

    output_path = OUTPUT_DIR / checkpoint.replace("/", "_") / aspect
    output_path.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
    )
    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        fp16=use_cuda,
        seed=seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ReviewDataset(train),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    started = time.perf_counter()
    trainer.train()
    predictions = trainer.predict(ReviewDataset(test)).predictions.argmax(axis=-1)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    expected = test["label"].map(label_to_id)
    return {
        "checkpoint": checkpoint,
        "aspect": aspect,
        "labeled_rows": int(len(train) + len(test)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "labels": labels,
        "accuracy": round(float(accuracy_score(expected, predictions)), 4),
        "macro_f1": round(float(f1_score(expected, predictions, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(expected, predictions, average="weighted", zero_division=0)), 4),
        "seconds": round(time.perf_counter() - started, 3),
        "device": "cuda" if use_cuda else "cpu",
        "model_path": str(output_path.relative_to(ROOT)),
    }


def train_numbered_model(model_name: str, output_number: int, epochs: float = 4.0, batch_size: int = 8, max_length: int = 256, seed: int = 42) -> None:
    data = pd.read_csv(INPUT_PATH, low_memory=False)
    checkpoint = MODEL_CHECKPOINTS[model_name]
    metrics = {}
    for aspect, column in SELECTED_ASPECTS.items():
        train, test, labels = prepare_split(data, column)
        metrics[aspect] = train_aspect(
            train, test, labels, checkpoint, aspect,
            epochs, batch_size, max_length, seed,
        )
        print(f"model_{output_number}/{aspect}: macro F1={metrics[aspect]['macro_f1']:.3f}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR.parent / f"model_{output_number}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics written to: {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=MODEL_CHECKPOINTS)
    group.add_argument("--all", action="store_true", help="Train every checkpoint family")
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_names = list(MODEL_CHECKPOINTS) if args.all else [args.model]
    for model_name in model_names:
        train_numbered_model(
            model_name,
            10 + list(MODEL_CHECKPOINTS).index(model_name),
            args.epochs,
            args.batch_size,
            args.max_length,
            args.seed,
        )


if __name__ == "__main__":
    main()