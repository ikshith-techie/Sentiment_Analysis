"""Model 5: TF-IDF plus SGD with modified Huber loss."""

from train_models import train_single_model


if __name__ == "__main__":
    train_single_model("sgd", 5)
