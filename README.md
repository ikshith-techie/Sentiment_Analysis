# Review Sentiment Analysis

This starter workflow keeps `synthetic_raw.csv` unchanged and prepares the review text for later aspect-level star prediction.

## Run

```powershell
py -m pip install -r requirements.txt
py preprocess_and_eda.py
```

The script writes `outputs/` containing:

- `reviews_preprocessed.csv`: cleaned text, lemmas, stems, text-length features, and the existing review star sentiment.
- `aspect_annotations_long.csv`: current numbered aspect columns reshaped into one row per review/aspect label.
- `review_overview.png`, `wordcloud.png`, and CSV summaries for the initial EDA.

Cleaning includes HTML decoding/removal, URL and email removal, lowercasing, conservative chat-word normalization, emoji sentiment conversion, repeated-character normalization, punctuation removal, tokenization, stopword-aware term analysis, stemming, and optional WordNet lemmatization. Automatic spelling correction is intentionally not enabled until the domain vocabulary and annotation policy are fixed.

When the new labels arrive, keep one sentiment column per aspect (for example `food_sentiment`, `ambience_sentiment`) and map each aspect's `positive`/`negative` label to its own 1–5 star target. The long annotation table is the compatible intermediate format.

## Local interface

Start the review interface with:

```powershell
py app.py
```

Open `http://127.0.0.1:8000`. The current baseline scores Food, Service, Ambience, Value, and Cleanliness as a placeholder. Type the complete review, then press **Go** (or `Ctrl/Cmd+Enter`) to score it and upsert it into `outputs/predicted_reviews.csv` with columns such as `food_stars`, `service_stars`, and `ambience_stars`. Replace `predict()` in `app.py` with the trained aspect model after the corrected labels are supplied.