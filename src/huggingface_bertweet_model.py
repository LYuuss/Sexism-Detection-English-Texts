# Pre trained model coming from https://huggingface.co/NLP-LTU/bertweet-large-sexism-detector

import pandas as pd
import torch

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from transformers import AutoTokenizer, AutoModelForSequenceClassification

"""
    We use a pre-trained / fine-tuned Hugging Face model:
    NLP-LTU/bertweet-large-sexism-detector

    This model directly predicts whether a text is:
        - not sexist
        - sexist

    We evaluate it on test.csv and print the results in the same
    format as the previous models.
"""

MODEL_NAME = "NLP-LTU/bertweet-large-sexism-detector"
DATA_PATH = "./dataset/test.csv"

# Load dataset
df = pd.read_csv(DATA_PATH)

texts = df["text"].fillna("").tolist()
y_true = df["label_sexist"].map({
    "not sexist": 0,
    "sexist": 1
}).tolist()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

def predict_texts(texts, batch_size=16, max_length=128):
    """
    Predict labels for a list of texts using the Hugging Face model.
    Returns a list of integers:
        0 -> not sexist
        1 -> sexist
    """
    predictions = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().tolist()

        predictions.extend(batch_preds)

        if (i // batch_size) % 50 == 0:
            print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

    return predictions

# Predict
y_pred = predict_texts(texts, batch_size=16, max_length=128)

# Evaluate
print(f"\n{'='*60}")
print("BERTweet-large sexism detector on train.csv")
print(f"{'='*60}")
print("Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification report:")
print(classification_report(y_true, y_pred, digits=4))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))