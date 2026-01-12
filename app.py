import os
import pickle
import re
from urllib.parse import urlparse

from flask import Flask, render_template, request, jsonify
from google.cloud import vision

app = Flask(__name__)

# ---------- LOAD ML MODELS ----------

with open("model/scam_model.pkl", "rb") as f:
    scam_model = pickle.load(f)

with open("model/scam_vectorizer.pkl", "rb") as f:
    scam_vectorizer = pickle.load(f)

with open("model/fake_news_model.pkl", "rb") as f:
    fake_news_model = pickle.load(f)

with open("model/fake_news_vectorizer.pkl", "rb") as f:
    fake_news_vectorizer = pickle.load(f)

# ---------- GOOGLE VISION CLIENT ----------
# IMPORTANT: No API key here
# Auth happens via GOOGLE_APPLICATION_CREDENTIALS
vision_client = vision.ImageAnnotatorClient()

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

# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")

# ---------- SCAM TEXT ANALYSIS ----------
@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "verdict": "No Input",
            "confidence": 0,
            "indicators": ["No text provided"]
        })

    cleaned = clean_text(text)
    vector = scam_vectorizer.transform([cleaned])

    pred = scam_model.predict(vector)[0]
    prob = scam_model.predict_proba(vector)[0]
    confidence = int(max(prob) * 100)

    verdict = "Scam Detected" if pred == "scam" else "Safe"

    return jsonify({
        "verdict": verdict,
        "confidence": confidence,
        "indicators": [
            "TF-IDF feature extraction",
            "Multinomial Naive Bayes classification"
        ]
    })

# ---------- FAKE NEWS TEXT ANALYSIS ----------
@app.route("/analyze-fake-news-text", methods=["POST"])
def analyze_fake_news_text():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({
            "verdict": "No Input",
            "confidence": 0,
            "indicators": ["No text provided"]
        })

    cleaned = clean_text(text)
    vector = fake_news_vectorizer.transform([cleaned])

    pred = fake_news_model.predict(vector)[0]
    prob = fake_news_model.predict_proba(vector)[0]
    confidence = int(max(prob) * 100)

    verdict = "Fake News" if pred == "fake" else "Real News"

    return jsonify({
        "verdict": verdict,
        "confidence": confidence,
        "indicators": [
            "TF-IDF feature extraction",
            "Fake news classification model"
        ]
    })

# ---------- URL ANALYSIS ----------
@app.route("/analyze-url", methods=["POST"])
def analyze_url():
    data = request.get_json(force=True)
    url = normalize_url(data.get("url", "").strip())

    if not url:
        return jsonify({
            "risk_level": "No Input",
            "risk_score": 0,
            "indicators": ["No URL provided"]
        })

    score = 0
    indicators = []

    parsed = urlparse(url if "://" in url else "http://" + url)
    domain = parsed.netloc.lower()
    registered_domain = get_registered_domain(domain)

    brands = [
        "google", "paypal", "microsoft", "facebook",
        "amazon", "apple", "sbi", "paytm", "phonepe"
    ]

    for brand in brands:
        if brand in domain and brand not in registered_domain:
            score += 40
            indicators.append(f"Brand impersonation detected: {brand}")
            break

    if len(url) > 75:
        score += 20
        indicators.append("Unusually long URL")

    if re.match(r"\d+\.\d+\.\d+\.\d+", domain):
        score += 30
        indicators.append("IP-based URL detected")

    for kw in ["login", "verify", "secure", "update", "claim", "free"]:
        if kw in url.lower():
            score += 10
            indicators.append(f"Phishing keyword detected: {kw}")

    score = min(score, 100)

    if score <= 30:
        risk = "Trusted"
    elif score <= 60:
        risk = "Suspicious"
    else:
        risk = "High Risk"

    return jsonify({
        "risk_level": risk,
        "risk_score": score,
        "indicators": indicators
    })

# ---------- IMAGE ANALYSIS (GOOGLE VISION OCR) ----------
@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({
            "verdict": "No Input",
            "confidence": 0,
            "indicators": ["No image uploaded"]
        })

    image_file = request.files["image"]
    content = image_file.read()

    image = vision.Image(content=content)
    response = vision_client.text_detection(image=image)

    if response.error.message:
        return jsonify({
            "verdict": "Unclear",
            "confidence": 0,
            "indicators": ["OCR processing failed"]
        })

    texts = response.text_annotations
    if not texts:
        return jsonify({
            "verdict": "Unclear",
            "confidence": 0,
            "indicators": ["No readable text detected"]
        })

    extracted_text = texts[0].description
    cleaned = clean_text(extracted_text)

    if not cleaned:
        return jsonify({
            "verdict": "Unclear",
            "confidence": 0,
            "indicators": ["Low OCR confidence"]
        })

    vector = fake_news_vectorizer.transform([cleaned])
    pred = fake_news_model.predict(vector)[0]
    prob = fake_news_model.predict_proba(vector)[0]
    confidence = int(max(prob) * 100)

    verdict = "Fake News" if pred == "fake" else "Real News"

    return jsonify({
        "verdict": verdict,
        "confidence": confidence,
        "indicators": [
            "Google Cloud Vision OCR",
            "Fake news ML classification applied"
        ],
        "extracted_text": extracted_text[:300]
    })

# ---------- RUN APP ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
