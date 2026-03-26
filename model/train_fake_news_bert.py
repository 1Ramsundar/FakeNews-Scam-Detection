"""
train_fake_news_bert.py
Fine-tunes DistilBERT on the Fake/Real news dataset.
Saves the model + tokenizer to model/bert_fake_news/
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

# ─────────────────────────── CONFIG ────────────────────────────
MODEL_NAME      = "distilbert-base-uncased"
SAVE_PATH       = os.path.join(os.path.dirname(__file__), "bert_fake_news")
DATASET_DIR     = os.path.join(os.path.dirname(__file__), "..", "dataset")
TRUE_CSV        = os.path.join(DATASET_DIR, "True.csv")
FAKE_CSV        = os.path.join(DATASET_DIR, "Fake.csv")
MAX_LEN         = 128   # Reduced for faster training on CPU
BATCH_SIZE      = 16    # Increased batch size for speed
EPOCHS          = 3
SEED            = 42
MAX_SAMPLES     = 3000  # 3000 per class = 6000 total → ~2-3 hrs on CPU

# ─────────────────────────── UTILS ─────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)            # remove URLs
    text = re.sub(r"[^a-zA-Z0-9 .,!?']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ──────────────────────── DATASET CLASS ────────────────────────
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─────────────────────────── METRICS ───────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}


# ──────────────────────────── MAIN ─────────────────────────────
def main():
    print("=" * 60)
    print("  DistilBERT Fake News Detection — Fine-Tuning")
    print("=" * 60)

    # ── 1. Load & prepare dataset ──────────────────────────────
    print("\n[1/5] Loading datasets …")

    true_df = pd.read_csv(TRUE_CSV)
    fake_df = pd.read_csv(FAKE_CSV)

    # Use 'text' column if available, else combine title + text
    def get_text(df):
        if "text" in df.columns:
            col = df["text"]
        else:
            col = df.get("title", pd.Series([""] * len(df)))
        return col.astype(str).apply(clean_text)

    true_df = true_df.assign(text=get_text(true_df), label=0).sample(
        min(MAX_SAMPLES, len(true_df)), random_state=SEED
    )
    fake_df = fake_df.assign(text=get_text(fake_df), label=1).sample(
        min(MAX_SAMPLES, len(fake_df)), random_state=SEED
    )

    df = pd.concat([true_df[["text", "label"]], fake_df[["text", "label"]]], ignore_index=True)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)

    print(f"      Total samples : {len(df)}")
    print(f"      Real news     : {(df['label'] == 0).sum()}")
    print(f"      Fake news     : {(df['label'] == 1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=SEED,
        stratify=df["label"].tolist(),
    )

    # ── 2. Tokenise ────────────────────────────────────────────
    print("\n[2/5] Tokenising with DistilBert tokenizer …")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_LEN)
    test_enc  = tokenizer(X_test,  truncation=True, padding=True, max_length=MAX_LEN)

    train_dataset = NewsDataset(train_enc, y_train)
    test_dataset  = NewsDataset(test_enc,  y_test)

    # ── 3. Load pre-trained model ──────────────────────────────
    print("\n[3/5] Loading DistilBERT base model …")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "real", 1: "fake"},
        label2id={"real": 0, "fake": 1},
    )

    # ── 4. Train ───────────────────────────────────────────────
    print("\n[4/5] Training …")
    training_args = TrainingArguments(
        output_dir              = "./bert_fake_checkpoints",
        num_train_epochs        = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        warmup_steps            = 200,
        weight_decay            = 0.01,
        logging_dir             = "./bert_fake_logs",
        logging_steps           = 100,
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "accuracy",
        seed                    = SEED,
        fp16                    = torch.cuda.is_available(),
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = test_dataset,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # ── 5. Evaluate & save ─────────────────────────────────────
    print("\n[5/5] Evaluating & saving …")
    preds_output = trainer.predict(test_dataset)
    preds        = np.argmax(preds_output.predictions, axis=1)

    acc = accuracy_score(y_test, preds)
    print(f"\n  ✅  Test Accuracy : {acc * 100:.2f}%\n")
    print(classification_report(y_test, preds, target_names=["Real News", "Fake News"]))

    os.makedirs(SAVE_PATH, exist_ok=True)
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"\n  💾  Model saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
