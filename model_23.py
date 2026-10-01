"""Model 23: DeBERTa-v3-base with focal loss."""

from train_advanced_models import train_numbered_model


if __name__ == "__main__":
    train_numbered_model("deberta-focal", 23)