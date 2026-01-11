import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Load datasets
true_df = pd.read_csv("dataset/True.csv")
fake_df = pd.read_csv("dataset/Fake.csv")

# Add labels
true_df["label"] = "real"
fake_df["label"] = "fake"

# Combine datasets
df = pd.concat([true_df, fake_df], axis=0)

# Clean text
df["text"] = df["text"].apply(clean_text)

X = df["text"]
y = df["label"]

# Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)

X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Fake News Model Accuracy:", accuracy)

# Save model
pickle.dump(model, open("model/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/fake_news_vectorizer.pkl", "wb"))

print("Fake News model trained and saved successfully")
