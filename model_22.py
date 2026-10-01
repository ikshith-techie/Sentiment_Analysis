"""Model 22: RoBERTa-base with class-weighted cross-entropy."""

from train_advanced_models import train_numbered_model


if __name__ == "__main__":
    train_numbered_model("roberta-weighted", 22)