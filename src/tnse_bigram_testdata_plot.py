import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

test_data_path = "./dataset/test.csv"
test_df = pd.read_csv(test_data_path)

texts = test_df["text"].fillna("")
labels = test_df["label_sexist"].map({
    "not sexist": 0,
    "sexist": 1
})

# keep it light
MAX_SAMPLES = 600
if len(test_df) > MAX_SAMPLES:
    sampled_df = test_df.sample(n=MAX_SAMPLES, random_state=42)
    texts = sampled_df["text"].fillna("")
    labels = sampled_df["label_sexist"].map({
        "not sexist": 0,
        "sexist": 1
    })

vectorizer = CountVectorizer(
    ngram_range=(1, 2),
    min_df=3
)
X = vectorizer.fit_transform(texts)

X_dense = X.toarray()

pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_dense)

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
    init="pca"
)
X_tsne = tsne.fit_transform(X_pca)

plt.figure(figsize=(10, 7))

for class_value, class_name in [(0, "not sexist"), (1, "sexist")]:
    idx = labels == class_value
    plt.scatter(
        X_tsne[idx, 0],
        X_tsne[idx, 1],
        alpha=0.7,
        label=class_name
    )

plt.title("t-SNE visualization of test texts (CountVectorizer bigrams)")
plt.xlabel("t-SNE dimension 1")
plt.ylabel("t-SNE dimension 2")
plt.legend()
plt.tight_layout()
plt.show()