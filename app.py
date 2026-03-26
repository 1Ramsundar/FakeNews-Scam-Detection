import os
import re
import pickle
import torch
import numpy as np
from urllib.parse import urlparse

from flask import Flask, render_template, request, jsonify
from PIL import Image
import pytesseract

# ---------- TESSERACT PATH (LOCAL WINDOWS) ----------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------- CREATE APP ----------
app = Flask(__name__)

# ---------- LOAD CLASSICAL MODELS ----------
with open("model/scam_model.pkl", "rb") as f:
    scam_model = pickle.load(f)

with open("model/scam_vectorizer.pkl", "rb") as f:
    scam_vectorizer = pickle.load(f)

with open("model/fake_news_model.pkl", "rb") as f:
    fake_news_model = pickle.load(f)

with open("model/fake_news_vectorizer.pkl", "rb") as f:
    fake_news_vectorizer = pickle.load(f)

# ---------- LAZY-LOAD BERT MODELS ----------
# Models are loaded only on the first BERT request to keep startup fast.
_bert_scam_tokenizer   = None
_bert_scam_model       = None
_bert_fake_tokenizer   = None
_bert_fake_model       = None
_bert_device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_bert_scam():
    global _bert_scam_tokenizer, _bert_scam_model
    if _bert_scam_tokenizer is None:
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        path = "model/bert_scam"
        _bert_scam_tokenizer = DistilBertTokenizerFast.from_pretrained(path)
        _bert_scam_model     = DistilBertForSequenceClassification.from_pretrained(path)
        _bert_scam_model.to(_bert_device).eval()
    return _bert_scam_tokenizer, _bert_scam_model

def get_bert_fake():
    global _bert_fake_tokenizer, _bert_fake_model
    if _bert_fake_tokenizer is None:
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        path = "model/bert_fake_news"
        _bert_fake_tokenizer = DistilBertTokenizerFast.from_pretrained(path)
        _bert_fake_model     = DistilBertForSequenceClassification.from_pretrained(path)
        _bert_fake_model.to(_bert_device).eval()
    return _bert_fake_tokenizer, _bert_fake_model

def bert_predict(tokenizer, model, text, max_len=128):
    """Return (predicted_label_str, confidence_int)."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True,
        padding=True, max_length=max_len
    )
    inputs = {k: v.to(_bert_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred   = int(np.argmax(probs))
    label  = model.config.id2label[pred]
    return label, int(max(probs) * 100)

# ---------- TEXT CLEANING ----------
def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------- URL HELPERS ----------
def normalize_url(url):
    return url.replace("[.]", ".")

def get_registered_domain(domain):
    parts = domain.split(".")
    if len(parts) >= 2:
        return parts[-2] + "." + parts[-1]
    return domain

# ==========================================================
#  HOME
# ==========================================================
@app.route("/")
def home():
    return render_template("index.html")

# ==========================================================
#  CLASSICAL — SCAM TEXT
# ==========================================================
@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"verdict": "No Input", "confidence": 0})

    vector = scam_vectorizer.transform([clean_text(text)])
    pred   = scam_model.predict(vector)[0]
    prob   = scam_model.predict_proba(vector)[0]

    return jsonify({
        "verdict":    "Scam Detected" if pred == "scam" else "Safe",
        "confidence": int(max(prob) * 100),
        "model_used": "TF-IDF + Naive Bayes",
        "indicators": ["TF-IDF", "Naive Bayes"]
    })

# ==========================================================
#  CLASSICAL — FAKE NEWS TEXT
# ==========================================================
@app.route("/analyze-fake-news-text", methods=["POST"])
def analyze_fake_news_text():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"verdict": "No Input", "confidence": 0})

    vector = fake_news_vectorizer.transform([clean_text(text)])
    pred   = fake_news_model.predict(vector)[0]
    prob   = fake_news_model.predict_proba(vector)[0]

    return jsonify({
        "verdict":    "Fake News" if pred == "fake" else "Real News",
        "confidence": int(max(prob) * 100),
        "model_used": "TF-IDF + Naive Bayes",
        "indicators": ["TF-IDF", "Fake News ML Model"]
    })

# ==========================================================
#  BERT — SCAM TEXT
# ==========================================================
@app.route("/analyze-text-bert", methods=["POST"])
def analyze_text_bert():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"verdict": "No Input", "confidence": 0})

    bert_model_dir = "model/bert_scam"
    if not os.path.isdir(bert_model_dir):
        return jsonify({
            "verdict":    "Model Not Found",
            "confidence": 0,
            "error":      "DistilBERT scam model not trained yet. Run model/train_scam_bert.py first."
        }), 503

    tokenizer, model = get_bert_scam()
    label, conf      = bert_predict(tokenizer, model, text, max_len=128)

    return jsonify({
        "verdict":    "Scam Detected" if label == "scam" else "Safe",
        "confidence": conf,
        "model_used": "DistilBERT",
        "indicators": ["DistilBERT", "Transformer Attention"]
    })

# ==========================================================
#  BERT — FAKE NEWS TEXT
# ==========================================================
@app.route("/analyze-fake-news-bert", methods=["POST"])
def analyze_fake_news_bert():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"verdict": "No Input", "confidence": 0})

    bert_model_dir = "model/bert_fake_news"
    if not os.path.isdir(bert_model_dir):
        return jsonify({
            "verdict":    "Model Not Found",
            "confidence": 0,
            "error":      "DistilBERT fake news model not trained yet. Run model/train_fake_news_bert.py first."
        }), 503

    tokenizer, model = get_bert_fake()
    label, conf      = bert_predict(tokenizer, model, text, max_len=256)

    return jsonify({
        "verdict":    "Fake News" if label == "fake" else "Real News",
        "confidence": conf,
        "model_used": "DistilBERT",
        "indicators": ["DistilBERT", "Transformer Attention"]
    })

# ==========================================================
#  URL ANALYSIS
# ==========================================================
@app.route("/analyze-url", methods=["POST"])
def analyze_url():
    data = request.get_json(force=True)
    url  = normalize_url(data.get("url", "").strip())
    if not url:
        return jsonify({"risk_level": "No Input", "risk_score": 0})

    score      = 0
    indicators = []

    parsed            = urlparse(url if "://" in url else "http://" + url)
    domain            = parsed.netloc.lower()
    registered_domain = get_registered_domain(domain)

    for brand in ["google", "paypal", "amazon", "sbi", "paytm", "phonepe"]:
        if brand in domain and brand not in registered_domain:
            score += 40
            indicators.append(f"Brand impersonation: {brand}")
            break

    if len(url) > 75:
        score += 20
        indicators.append("Long URL")

    if re.match(r"\d+\.\d+\.\d+\.\d+", domain):
        score += 30
        indicators.append("IP-based URL")

    score = min(score, 100)
    risk  = "Trusted" if score <= 30 else "Suspicious" if score <= 60 else "High Risk"

    return jsonify({"risk_level": risk, "risk_score": score, "indicators": indicators})

# ==========================================================
#  IMAGE ANALYSIS (Tesseract → Classical Model)
# ==========================================================
@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"verdict": "No Input", "confidence": 0})

    image_file     = request.files["image"]
    img            = Image.open(image_file).convert("L")
    extracted_text = pytesseract.image_to_string(img)

    if not extracted_text.strip():
        return jsonify({"verdict": "Unclear", "confidence": 0})

    vector = fake_news_vectorizer.transform([clean_text(extracted_text)])
    pred   = fake_news_model.predict(vector)[0]
    prob   = fake_news_model.predict_proba(vector)[0]

    return jsonify({
        "verdict":        "Fake News" if pred == "fake" else "Real News",
        "confidence":     int(max(prob) * 100),
        "model_used":     "TF-IDF + Naive Bayes",
        "indicators":     ["Tesseract OCR", "ML Classification"],
        "extracted_text": extracted_text[:300]
    })

# ==========================================================
#  RUN
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)