import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

"""
    This script visualizes transformer embeddings with t-SNE
    using the Hugging Face model:
        NLP-LTU/bertweet-large-sexism-detector

    Steps:
        - load texts from train.csv
        - tokenize them
        - extract one embedding per text from the last hidden state
        - reduce embeddings to 2D with t-SNE
        - plot the points colored by label

    Important:
        t-SNE is for visualization, not classification.
"""

MODEL_NAME = "NLP-LTU/bertweet-large-sexism-detector"
DATA_PATH = "./dataset/test.csv"

# To keep it light, use only a subset
MAX_SAMPLES = 1500
BATCH_SIZE = 16
MAX_LENGTH = 128
RANDOM_STATE = 42

# Load dataset
df = pd.read_csv(DATA_PATH)

df = df[["text", "label_sexist"]].dropna().copy()
df["label"] = df["label_sexist"].map({
    "not sexist": 0,
    "sexist": 1
})

if len(df) > MAX_SAMPLES:
    df, _ = train_test_split(
        df,
        train_size=MAX_SAMPLES,
        stratify=df["label"],
        random_state=RANDOM_STATE
    )

texts = df["text"].tolist()
labels = df["label"].to_numpy()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

def extract_embeddings(texts, batch_size=16, max_length=128):
    """
    Extract one embedding per text from the transformer's last hidden state.

    We use the first token embedding of the last hidden layer
    as a simple sentence-level representation.
    """
    all_embeddings = []

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
            outputs = model(**encodings, output_hidden_states=True)

            # Last hidden state: [batch_size, seq_len, hidden_dim]
            last_hidden_state = outputs.hidden_states[-1]

            # Take the first token embedding for each text
            batch_embeddings = last_hidden_state[:, 0, :]

        all_embeddings.append(batch_embeddings.cpu().numpy())

        print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

    return np.vstack(all_embeddings)

# Extract embeddings
embeddings = extract_embeddings(
    texts,
    batch_size=BATCH_SIZE,
    max_length=MAX_LENGTH
)

print("Embeddings shape:", embeddings.shape)

# t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=RANDOM_STATE,
    init="pca",
    learning_rate="auto"
)

embeddings_2d = tsne.fit_transform(embeddings)

# Plot
plt.figure(figsize=(10, 7))

for label_value, label_name in [(0, "not sexist"), (1, "sexist")]:
    mask = labels == label_value
    plt.scatter(
        embeddings_2d[mask, 0],
        embeddings_2d[mask, 1],
        label=label_name,
        alpha=0.7,
        s=20
    )

plt.title("t-SNE of BERTweet embeddings on test.csv (sample of 1500)")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.legend()
plt.tight_layout()
plt.show()