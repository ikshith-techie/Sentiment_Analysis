# Review Sentiment Analysis

This workflow reads `1000_ds_Sentiment_analysis.csv` and prepares its selected aspect labels for analysis.

## Run

```powershell
py -m pip install -r requirements.txt
py preprocess_and_eda.py
py train_models.py
```

## Model versions

`model_1.py` remains the baseline TF-IDF plus Logistic Regression approach. The standalone model scripts are:

- `model_2.py`: Linear SVM
- `model_3.py`: Complement Naive Bayes
- `model_4.py`: Multinomial Naive Bayes
- `model_5.py`: SGD with modified Huber loss
- `model_6.py`: Logistic Regression
- `model_7.py`: Ridge Classifier
- `model_8.py`: Passive-Aggressive Classifier
- `model_9.py`: Bernoulli Naive Bayes

Run any standalone model with `py model_2.py` (replace the number as needed). Each writes its own `outputs/model_N.joblib` and `outputs/model_N_metrics.json`. `train_models.py` still benchmarks all five CPU-friendly classifiers, writes `outputs/model_benchmark.joblib`, selects the highest macro-F1 model per aspect in `outputs/model_best.joblib`, and records timing in `outputs/model_benchmark_metrics.json`. GPU-heavy neural models are intentionally not run here.

To combine all standalone metrics into one model-keyed dictionary, run:

```powershell
py result.py
```

This creates `outputs/result.json` with `model1` through `model9`. Each aspect contains only `accuracy`, `macro_f1`, and `weighted_f1`.

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