# Review Sentiment Analysis

This workflow reads `1000_ds_Sentiment_analysis.csv` and prepares its selected aspect labels for analysis.

## Run

```powershell
py -m pip install -r requirements.txt
py preprocess_and_eda.py
py model_1.py
```

## Model versions

`model_1.py` is the baseline TF-IDF plus Logistic Regression approach. It trains all five selected aspect classifiers together and writes `outputs/model_1.joblib` and `outputs/model_1_metrics.json`. Future advanced approaches should be added as `model_2.py`, with their own versioned artifacts and metrics.

The script writes `outputs/` containing:

- `reviews_preprocessed.csv`: cleaned text, lemmas, stems, text-length features, and the existing review star sentiment.
- `aspect_annotations_long.csv`: current numbered aspect columns reshaped into one row per review/aspect label.
- `review_overview.png`, `wordcloud.png`, and CSV summaries for the initial EDA.

Cleaning includes HTML decoding/removal, URL and email removal, lowercasing, conservative chat-word normalization, emoji sentiment conversion, repeated-character normalization, punctuation removal, tokenization, stopword-aware term analysis, stemming, and optional WordNet lemmatization. Automatic spelling correction is intentionally not enabled until the domain vocabulary and annotation policy are fixed.

The selected source columns are `FOOD#QUALITY`, `AMBIENCE#GENERAL`, `FOOD#PRICES`, `LOCATION#GENERAL`, and `RESTAURANT#GENERAL`. They become Food quality, Ambience, Prices, Location, and General in the long annotation table. Because this dataset has no overall star column, the EDA-only `stars` value is derived from the selected labels: positive = 5, negative = 1, and mixed/neutral = 3.

## Local interface

Start the review interface with:

```powershell
py app.py
```

Open `http://127.0.0.1:8000`. The current baseline scores Food quality, Ambience, Prices, Location, and General as a placeholder. Type the complete review, then press **Go** (or `Ctrl/Cmd+Enter`) to score it and upsert it into `outputs/predicted_reviews.csv`. Replace `predict()` in `app.py` with the trained aspect model after model training.