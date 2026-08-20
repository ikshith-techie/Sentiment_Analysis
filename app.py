"""Local review scoring interface and CSV persistence layer."""

from __future__ import annotations

import csv
import html
import json
import re
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
DATA_PATH = ROOT / "outputs" / "predicted_reviews.csv"
HOST = "127.0.0.1"
PORT = 8000

ASPECTS = {
    "food": {"label": "Food", "terms": {"food", "dish", "meal", "taste", "flavor", "flavour", "menu", "pizza", "chicken", "ramen", "soup"}},
    "service": {"label": "Service", "terms": {"service", "staff", "waiter", "waitress", "manager", "host", "order", "wait", "rude"}},
    "ambience": {"label": "Ambience", "terms": {"ambience", "atmosphere", "music", "decor", "quiet", "loud", "cozy", "crowded", "place"}},
    "value": {"label": "Value", "terms": {"price", "priced", "cost", "value", "cheap", "expensive", "worth", "money", "portion"}},
    "cleanliness": {"label": "Cleanliness", "terms": {"clean", "dirty", "hygiene", "bathroom", "table", "floor", "smell", "sanitary"}},
}
POSITIVE = {"amazing", "awesome", "best", "delicious", "excellent", "good", "great", "happy", "love", "perfect", "pleasant", "recommend", "wonderful"}
NEGATIVE = {"awful", "bad", "disappoint", "dirty", "horrible", "poor", "rude", "sad", "terrible", "worst", "bland", "slow", "cold", "overpriced"}


def clean_text(value: str) -> str:
    text = html.unescape(value or "").lower()
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z\s]", " ", text)).strip()


def score_aspect(text: str, terms: set[str]) -> dict[str, object]:
    cleaned = clean_text(text)
    tokens = set(cleaned.split())
    context = tokens & terms
    evidence = cleaned
    raw_sentences = re.split(r"[.!?,;]+", html.unescape(text or "").lower())
    aspect_sentences = [clean_text(sentence) for sentence in raw_sentences if set(clean_text(sentence).split()) & terms]
    if aspect_sentences:
        evidence = " ".join(aspect_sentences)
    evidence_tokens = set(evidence.split())
    positive = len(evidence_tokens & POSITIVE)
    negative = len(evidence_tokens & NEGATIVE)
    raw_score = 3 + positive - negative
    if context and positive == 0 and negative == 0:
        raw_score = 3
    stars = max(1, min(5, raw_score))
    sentiment = "positive" if stars >= 4 else "negative" if stars <= 2 else "neutral"
    return {"stars": stars, "sentiment": sentiment, "confidence": "baseline", "matched_terms": sorted(context)}


def predict(text: str) -> dict[str, object]:
    return {key: score_aspect(text, settings["terms"]) for key, settings in ASPECTS.items()}


def save_review(review_id: str, text: str, results: dict[str, object]) -> None:
    DATA_PATH.parent.mkdir(exist_ok=True)
    row: dict[str, object] = {"review_id": review_id, "text": text}
    for aspect, result in results.items():
        row[f"{aspect}_stars"] = result["stars"]
        row[f"{aspect}_sentiment"] = result["sentiment"]
    fields = list(row)
    rows = []
    if DATA_PATH.exists():
        with DATA_PATH.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    existing = next((index for index, item in enumerate(rows) if item.get("review_id") == review_id), None)
    if existing is None:
        rows.append(row)
    else:
        rows[existing] = row
    with DATA_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class RequestHandler(BaseHTTPRequestHandler):
    def send_json(self, payload: dict[str, object], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/aspects":
            self.send_json({"aspects": {key: value["label"] for key, value in ASPECTS.items()}})
            return
        file_path = WEB_DIR / ("index.html" if path == "/" else path.removeprefix("/"))
        if file_path.is_file() and WEB_DIR in file_path.parents:
            content_type = "text/html; charset=utf-8" if file_path.suffix == ".html" else "text/css; charset=utf-8" if file_path.suffix == ".css" else "application/javascript; charset=utf-8"
            body = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_json({"error": "Not found"}, 404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self.send_json({"error": "Not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            text = str(payload.get("text", "")).strip()
            review_id = str(payload.get("review_id") or uuid.uuid4())
            results = predict(text)
            if text:
                save_review(review_id, text, results)
            self.send_json({"review_id": review_id, "results": results, "saved": bool(text)})
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, 400)

    def log_message(self, format: str, *args: object) -> None:
        return


if __name__ == "__main__":
    print(f"Review interface running at http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), RequestHandler).serve_forever()