import os
import requests
import pickle
import re
from urllib.parse import urlparse
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ---------- LOAD MODELS ----------

with open("model/scam_model.pkl", "rb") as f:
    scam_model = pickle.load(f)

with open("model/scam_vectorizer.pkl", "rb") as f:
    scam_vectorizer = pickle.load(f)

with open("model/fake_news_model.pkl", "rb") as f:
    fake_news_model = pickle.load(f)

with open("model/fake_news_vectorizer.pkl", "rb") as f:
    fake_news_vectorizer = pickle.load(f)

# ---------- TEXT CLEANING ----------
def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------- URL HELPERS ----------
def get_registered_domain(domain):
    parts = domain.split(".")
    if len(parts) >= 2:
        return parts[-2] + "." + parts[-1]
    return domain

def normalize_url(url):
    return url.replace("[.]", ".")

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

    cleaned = clean_text(text)
    vector = scam_vectorizer.transform([cleaned])

    pred = scam_model.predict(vector)[0]
    prob = scam_model.predict_proba(vector)[0]
    confidence = int(max(prob) * 100)

    verdict = "Scam Detected" if pred == "scam" else "Safe"

    return jsonify({
        "verdict": verdict,
        "confidence": confidence,
        "indicators": ["TF-IDF feature extraction", "Naive Bayes classification"]
    })

# ---------- FAKE NEWS TEXT ----------
@app.route("/analyze-fake-news-text", methods=["POST"])
def analyze_fake_news_text():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"verdict": "No Input", "confidence": 0, "indicators": ["No text provided"]})

    cleaned = clean_text(text)
    vector = fake_news_vectorizer.transform([cleaned])

    pred = fake_news_model.predict(vector)[0]
    prob = fake_news_model.predict_proba(vector)[0]
    confidence = int(max(prob) * 100)

    verdict = "Fake News" if pred == "fake" else "Real News"

    return jsonify({
        "verdict": verdict,
        "confidence": confidence,
        "indicators": ["TF-IDF feature extraction", "Fake news ML model applied"]
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

    brands = ["google","paypal","microsoft","facebook","amazon","apple","sbi","paytm","phonepe"]
    for brand in brands:
        if brand in domain and brand not in registered_domain:
            score += 40
            indicators.append(f"Brand impersonation detected: {brand}")
            break

    if len(url) > 75:
        score += 20
        indicators.append("Long URL detected")

    if re.match(r"\d+\.\d+\.\d+\.\d+", domain):
        score += 30
        indicators.append("IP-based URL")

    for kw in ["login","verify","secure","update","claim","free"]:
        if kw in url.lower():
            score += 10
            indicators.append(f"Phishing keyword: {kw}")

    score = min(score, 100)
    risk = "Trusted" if score <= 30 else "Suspicious" if score <= 60 else "High Risk"

    return jsonify({"risk_level": risk, "risk_score": score, "indicators": indicators})

# ---------- IMAGE ANALYSIS (CLOUD OCR) ----------
@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({"verdict": "No Input", "confidence": 0, "indicators": ["No image uploaded"]})

    image_file = request.files["image"]

    api_key = os.environ.get("OCR_API_KEY", "helloworld")
    ocr_url = "https://api.ocr.space/parse/image"

    response = requests.post(
        ocr_url,
        files={"file": image_file},
        data={"apikey": api_key, "language": "eng"}
    )

    result = response.json()
    print(result)

    if result.get("IsErroredOnProcessing"):
        return jsonify({"verdict": "Unclear", "confidence": 0, "indicators": ["OCR failed"]})

    extracted_text = result["ParsedResults"][0]["ParsedText"]
    cleaned = clean_text(extracted_text)

    if not cleaned:
        return jsonify({"verdict": "Unclear", "confidence": 0, "indicators": ["No readable text"]})

    vector = fake_news_vectorizer.transform([cleaned])
    pred = fake_news_model.predict(vector)[0]
    prob = fake_news_model.predict_proba(vector)[0]
    confidence = int(max(prob) * 100)

    verdict = "Fake News" if pred == "fake" else "Real News"

    return jsonify({
        "verdict": verdict,
        "confidence": confidence,
        "indicators": ["Cloud OCR used", "ML classification applied"],
        "extracted_text": extracted_text[:300]
    })

# ---------- RUN ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)