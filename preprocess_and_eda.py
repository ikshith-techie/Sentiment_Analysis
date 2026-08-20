"""Preprocess review text and generate a first NLP EDA report.

Run from the project directory:
    py preprocess_and_eda.py

The raw CSV is never modified. Outputs are written below ``outputs/``.
"""

from __future__ import annotations

import html
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import WordCloud


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "synthetic_raw.csv"
OUTPUT_DIR = ROOT / "outputs"
STEMMER = SnowballStemmer("english")
STOPWORDS = set(ENGLISH_STOP_WORDS)

# Conservative chat normalization: preserve meaning while handling common shorthand.
CHAT_WORDS = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "idk": "i do not know",
    "dont": "do not",
    "cant": "cannot",
    "wont": "will not",
    "didnt": "did not",
    "wasnt": "was not",
    "isnt": "is not",
}


def replace_chat_words(text: str) -> str:
    return re.sub(
        r"\b[\w']+\b",
        lambda match: CHAT_WORDS.get(match.group(0), match.group(0)),
        text,
        flags=re.IGNORECASE,
    )


def emoji_to_words(text: str) -> str:
    """Convert common sentiment emoji to words without requiring an emoji package."""
    replacements = {
        "😀": " happy ", "😃": " happy ", "😄": " happy ", "😍": " love ",
        "😊": " happy ", "👍": " good ", "❤": " love ", "❤️": " love ",
        "😞": " sad ", "😔": " sad ", "😡": " angry ", "😠": " angry ",
        "👎": " bad ", "😭": " crying ", "😂": " funny ",
    }
    for symbol, word in replacements.items():
        text = text.replace(symbol, word)
    return text


def clean_text(value: object) -> str:
    """Normalize a review while retaining negation words and sentiment cues."""
    text = "" if pd.isna(value) else html.unescape(str(value))
    text = emoji_to_words(text).lower()
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text)
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = replace_chat_words(text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # coooool -> cool
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", text)


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Use WordNet when available; otherwise return tokens unchanged."""
    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokens]
    except LookupError:
        return tokens


def build_aspect_table(data: pd.DataFrame) -> pd.DataFrame:
    """Convert numbered wide labels into a future-friendly long annotation table."""
    rows = []
    for column in data.columns:
        match = re.fullmatch(r"labels/(\d+)/category", column)
        if not match:
            continue
        index = match.group(1)
        sentiment_column = f"labels/{index}/sentiment"
        aspect_column = f"labels/{index}/aspect_term"
        implicit_column = f"labels/{index}/is_implicit"
        subset = data[["review_id", aspect_column, column, sentiment_column, implicit_column]].copy()
        subset.columns = ["review_id", "aspect_term", "category", "sentiment", "is_implicit"]
        rows.append(subset.dropna(subset=["category", "sentiment"]))
    if not rows:
        return pd.DataFrame(columns=["review_id", "aspect_term", "category", "sentiment", "is_implicit"])
    return pd.concat(rows, ignore_index=True)


def save_eda(data: pd.DataFrame, tokens: list[str]) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.countplot(data=data, x="stars", ax=axes[0], color="#176b87")
    axes[0].set_title("Star distribution")
    axes[0].set_xlabel("Stars")
    axes[0].set_ylabel("Reviews")
    sns.histplot(data=data, x="word_count", bins=25, ax=axes[1], color="#e07a5f")
    axes[1].set_title("Review length")
    axes[1].set_xlabel("Words")
    sns.boxplot(data=data, x="stars", y="word_count", ax=axes[2], color="#81b29a")
    axes[2].set_title("Length by rating")
    axes[2].set_xlabel("Stars")
    axes[2].set_ylabel("Words")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "review_overview.png", dpi=160)
    plt.close(fig)

    frequencies = Counter(token for token in tokens if token not in STOPWORDS and len(token) > 2)
    pd.DataFrame(frequencies.most_common(50), columns=["term", "frequency"]).to_csv(
        OUTPUT_DIR / "top_terms.csv", index=False
    )
    if frequencies:
        cloud = WordCloud(width=1400, height=700, background_color="white", colormap="viridis")
        cloud.generate_from_frequencies(frequencies).to_file(str(OUTPUT_DIR / "wordcloud.png"))


def main() -> None:
    data = pd.read_csv(INPUT_PATH, low_memory=False)
    data["clean_text"] = data["text"].map(clean_text)
    data["tokens"] = data["clean_text"].map(tokenize)
    data["lemmas"] = data["tokens"].map(lemmatize_tokens)
    data["stems"] = data["lemmas"].map(lambda values: [STEMMER.stem(value) for value in values])
    data["word_count"] = data["tokens"].str.len()
    data["char_count"] = data["clean_text"].str.len()
    data["review_sentiment"] = data["stars"].map({1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"})

    aspects = build_aspect_table(data)
    data.drop(columns=["tokens", "lemmas", "stems"]).to_csv(OUTPUT_DIR / "reviews_preprocessed.csv", index=False)
    aspects.to_csv(OUTPUT_DIR / "aspect_annotations_long.csv", index=False)
    data[["stars", "word_count", "char_count"]].describe().to_csv(OUTPUT_DIR / "numeric_summary.csv")
    data["review_sentiment"].value_counts().rename_axis("sentiment").to_csv(OUTPUT_DIR / "sentiment_summary.csv")
    if not aspects.empty:
        aspects.groupby(["category", "sentiment"]).size().reset_index(name="count").to_csv(
            OUTPUT_DIR / "aspect_sentiment_summary.csv", index=False
        )
    save_eda(data, [token for row in data["lemmas"] for token in row])
    print(f"Processed {len(data):,} reviews and {len(aspects):,} aspect annotations.")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()