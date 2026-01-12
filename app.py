import os
import json
import tempfile
import pickle
import re
from urllib.parse import urlparse

from flask import Flask, render_template, request, jsonify
from google.cloud import vision

# ---------- GOOGLE CREDENTIALS SETUP (MUST BE FIRST) ----------
if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
    creds = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    temp.write(json.dumps(creds).encode())
    temp.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp.name

# ---------- CREATE APP ----------
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

# ---------- SCAM TEXT ----------
@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"verdict": "No Input", "confidence": 0, "indicators": ["No text provided"]})

    vector = scam_vectorizer.transform([clean_text(text)])
    pred = scam_model.predict(vector)[0]
    prob = scam_model.predict_proba(vector)[0]

    return jsonify({
        "verdict": "Scam Detected" if pred == "scam" else "Safe",
        "confidence": int(max(prob) * 100),
        "indicators": ["TF-IDF", "Naive Bayes"]
    })

# ---------- FAKE NEWS TEXT ----------
@app.route("/analyze-fake-news-text", methods=["POST"])
def analyze_fake_news_text():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"verdict": "No Input", "confidence": 0, "indicators": ["No text provided"]})

    vector = fake_news_vectorizer.transform([clean_text(text)])
    pred = fake_news_model.predict(vector)[0]
    prob = fake_news_model.predict_proba(vector)[0]

    return jsonify({
        "verdict": "Fake News" if pred == "fake" else "Real News",
        "confidence": int(max(prob) * 100),
        "indicators": ["TF-IDF", "Fake News ML Model"]
    })

# ---------- URL ANALYSIS ----------
@app.route("/analyze-url", methods=["POST"])
def analyze_url():
    data = request.get_json(force=True)
    url = normalize_url(data.get("url", "").strip())

    if not url:
        return jsonify({"risk_level": "No Input", "risk_score": 0, "indicators": ["No URL provided"]})

    score = 0
    indicators = []

    parsed = urlparse(url if "://" in url else "http://" + url)
    domain = parsed.netloc.lower()
    registered_domain = get_registered_domain(domain)

    for brand in ["google","paypal","amazon","sbi","paytm","phonepe"]:
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
    risk = "Trusted" if score <= 30 else "Suspicious" if score <= 60 else "High Risk"

    return jsonify({"risk_level": risk, "risk_score": score, "indicators": indicators})

# ---------- IMAGE ANALYSIS (GOOGLE OCR) ----------
@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"verdict": "No Input", "confidence": 0, "indicators": ["No image uploaded"]})

    image = vision.Image(content=request.files["image"].read())
    response = vision_client.text_detection(image=image)

    if response.error.message or not response.text_annotations:
        return jsonify({"verdict": "Unclear", "confidence": 0, "indicators": ["OCR failed"]})

    text = response.text_annotations[0].description
    vector = fake_news_vectorizer.transform([clean_text(text)])
    pred = fake_news_model.predict(vector)[0]
    prob = fake_news_model.predict_proba(vector)[0]

    return jsonify({
        "verdict": "Fake News" if pred == "fake" else "Real News",
        "confidence": int(max(prob) * 100),
        "indicators": ["Google Vision OCR", "ML Classification"],
        "extracted_text": text[:300]
    })

# ---------- RUN ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
