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
- `model_10.py`: DistilBERT transformer
- `model_11.py`: BERT transformer
- `model_12.py`: RoBERTa transformer
- `model_13.py`: DeBERTa transformer

Run any standalone CPU model with `py model_2.py` (replace the number as needed). Each writes its own `outputs/model_N.joblib` and `outputs/model_N_metrics.json`. `train_models.py` still benchmarks all five CPU-friendly classifiers, writes `outputs/model_benchmark.joblib`, selects the highest macro-F1 model per aspect in `outputs/model_best.joblib`, and records timing in `outputs/model_benchmark_metrics.json`.

## Advanced GPU models

`model_10.py` through `model_13.py` fine-tune transformer classifiers for the same five aspect labels. Each checkpoint is trained separately for each aspect. The Hugging Face model files are saved below `outputs/advanced_models/`, while metrics use the same numbered format as the earlier models.

On the college server, install the PyTorch build that matches its CUDA version first, then install the project dependencies:

```powershell
py -m pip install -r requirements.txt
py model_10.py
```

Run `py model_11.py`, `py model_12.py`, or `py model_13.py` for the other transformer checkpoints. These scripts download large checkpoints and require substantial GPU memory, so they are intended for the college server. The shared `train_advanced_models.py` module contains the implementation and optional CLI controls such as `--epochs`, `--batch-size`, and `--max-length`.

To combine all standalone metrics into one model-keyed dictionary, run:

```powershell
py result.py
```

This creates `outputs/result.json` with whichever numbered model metrics are available, including `model10` through `model13` after GPU training. Each aspect contains only `accuracy`, `macro_f1`, and `weighted_f1`.

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