from flask import Flask, render_template, request, jsonify
from PIL import Image
import pytesseract
import pickle
import re
from urllib.parse import urlparse

# Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)

# ---------- LOAD MODELS ----------

# Scam detection model (SMS)
with open("model/scam_model.pkl", "rb") as f:
    scam_model = pickle.load(f)

with open("model/scam_vectorizer.pkl", "rb") as f:
    scam_vectorizer = pickle.load(f)

# Fake news detection model (News articles)
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

# ---------- REGISTERED DOMAIN ----------
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

    prediction = scam_model.predict(vector)[0]
    probabilities = scam_model.predict_proba(vector)[0]
    confidence = int(max(probabilities) * 100)

    verdict = "Scam Detected" if prediction == "scam" else "Safe"

    return jsonify({
        "verdict": verdict,
        "confidence": confidence,
        "indicators": [
            "Text classified using TF-IDF features",
            "Naive Bayes probability-based decision"
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

    prediction = fake_news_model.predict(vector)[0]
    probabilities = fake_news_model.predict_proba(vector)[0]
    confidence = int(max(probabilities) * 100)

    verdict = "Fake News" if prediction == "fake" else "Real News"

    return jsonify({
        "verdict": verdict,
        "confidence": confidence,
        "indicators": [
            "Text classified using TF-IDF features",
            "Fake news detection model applied"
        ]
    })

def normalize_url(url):
    # Convert obfuscated URLs like [.] to .
    url = url.replace("[.]", ".")
    return url
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

    brand_names = [
        "google", "paypal", "microsoft", "facebook",
        "amazon", "apple", "samsung", "netflix",
        "instagram", "twitter", "whatsapp",
        "sbi", "hdfc", "icici", "axis",
        "flipkart", "phonepe", "paytm"
    ]

    for brand in brand_names:
        if brand in domain and brand not in registered_domain:
            score += 40
            indicators.append(f"Brand impersonation detected: '{brand}'")
            break

    if len(url) > 75:
        score += 20
        indicators.append("Unusually long URL")

    if re.match(r"\d+\.\d+\.\d+\.\d+", domain):
        score += 30
        indicators.append("IP address used instead of domain")

    suspicious_tlds = [".xyz", ".tk", ".top", ".info", ".buzz"]
    if any(domain.endswith(tld) for tld in suspicious_tlds):
        score += 25
        indicators.append("Suspicious top-level domain")

    if "@" in url or domain.count("-") > 2:
        score += 15
        indicators.append("Suspicious special characters in URL")

    keywords = ["login", "verify", "secure", "update", "claim", "free"]
    for kw in keywords:
        if kw in url.lower():
            score += 10
            indicators.append(f"Phishing keyword detected: '{kw}'")

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

# ---------- IMAGE ANALYSIS ----------
@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    if "image" not in request.files:
        return jsonify({
            "verdict": "No Input",
            "confidence": 0,
            "indicators": ["No image uploaded"]
        })

    image_file = request.files["image"]
    img = Image.open(image_file).convert("L")
    img = img.resize((img.width * 2, img.height * 2))

    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(img, config=custom_config)

    cleaned = clean_text(extracted_text)
    if not cleaned:
        return jsonify({
            "verdict": "Unclear",
            "confidence": 0,
            "indicators": ["Low image clarity for OCR"],
            "extracted_text": ""
        })

    vector = fake_news_vectorizer.transform([cleaned])
    prediction = fake_news_model.predict(vector)[0]
    probabilities = fake_news_model.predict_proba(vector)[0]
    confidence = int(max(probabilities) * 100)

    verdict = "Fake News" if prediction == "fake" else "Real News"

    return jsonify({
        "verdict": verdict,
        "confidence": confidence,
        "indicators": [
            "Text extracted using OCR",
            "Fake news classification applied"
        ],
        "extracted_text": extracted_text[:300]
    })

# ---------- RUN APP ----------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


