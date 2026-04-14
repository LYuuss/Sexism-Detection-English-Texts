from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from app.loadData import load_file
from app.models.huggingface_bertweet import extract_embeddings, load_model
from app.text_processing import map_data

DEFAULT_RANDOM_STATE = 42
DEFAULT_FIGSIZE = (10, 7)
LABEL_NAMES = {
    0: "not sexist",
    1: "sexist",
}


@dataclass
class TsneProjectionResult:
    coordinates: np.ndarray
    labels: np.ndarray
    texts: list[str]
    title: str
    sample_size: int


def _validate_sample_size(sample_size):
    if sample_size is None:
        return

    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer greater than or equal to 2.")

    if sample_size < 2:
        raise ValueError("sample_size must be greater than or equal to 2.")


def _prepare_visualization_frame(data=None, dataset_name="test"):
    frame = load_file(dataset_name).copy() if data is None else data.copy()

    required_columns = {"text", "label_sexist"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "data must contain the following columns: "
            f"{', '.join(sorted(required_columns))}."
        )

    frame = frame[["text", "label_sexist"]].copy()
    frame["text"] = frame["text"].fillna("")
    frame["label"] = map_data(frame)
    frame = frame.dropna(subset=["label"]).copy()

    if frame.empty:
        raise ValueError("No rows with valid labels were found in the dataset.")

    frame["label"] = frame["label"].astype(int)
    return frame.reset_index(drop=True)


def _sample_frame(frame, sample_size=None, stratify=False, random_state=DEFAULT_RANDOM_STATE):
    _validate_sample_size(sample_size)

    if sample_size is None or len(frame) <= sample_size:
        return frame.reset_index(drop=True)

    if stratify and frame["label"].nunique() > 1 and frame["label"].value_counts().min() >= 2:
        sampled_frame, _ = train_test_split(
            frame,
            train_size=sample_size,
            stratify=frame["label"],
            random_state=random_state
        )
        return sampled_frame.reset_index(drop=True)

    return frame.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def _resolve_pca_components(matrix, pca_components):
    if pca_components is None:
        return None

    if isinstance(pca_components, bool) or not isinstance(pca_components, int):
        raise TypeError("pca_components must be an integer greater than or equal to 1.")

    if pca_components < 1:
        raise ValueError("pca_components must be greater than or equal to 1.")

    return min(pca_components, matrix.shape[0], matrix.shape[1])


def _resolve_perplexity(n_samples, perplexity):
    if isinstance(perplexity, bool) or not isinstance(perplexity, (int, float)):
        raise TypeError("perplexity must be a positive number.")

    if perplexity <= 0:
        raise ValueError("perplexity must be greater than 0.")

    return min(float(perplexity), max(1.0, float(n_samples - 1)))


def _fit_tsne(matrix, perplexity=30, random_state=DEFAULT_RANDOM_STATE):
    if matrix.shape[0] < 2:
        raise ValueError("t-SNE requires at least 2 samples.")

    tsne = TSNE(
        n_components=2,
        perplexity=_resolve_perplexity(matrix.shape[0], perplexity),
        random_state=random_state,
        init="pca" if matrix.shape[1] >= 2 else "random",
        learning_rate="auto"
    )
    return tsne.fit_transform(matrix)


def plot_tsne_projection(
    projection,
    figure_size=DEFAULT_FIGSIZE,
    alpha=0.7,
    point_size=20,
    output_path=None,
    show=False
):
    figure, axis = plt.subplots(figsize=figure_size)

    for label_value, label_name in LABEL_NAMES.items():
        mask = projection.labels == label_value
        if not np.any(mask):
            continue

        axis.scatter(
            projection.coordinates[mask, 0],
            projection.coordinates[mask, 1],
            label=label_name,
            alpha=alpha,
            s=point_size
        )

    axis.set_title(projection.title)
    axis.set_xlabel("t-SNE dimension 1")
    axis.set_ylabel("t-SNE dimension 2")

    if len(axis.collections) > 0:
        axis.legend()

    figure.tight_layout()

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_file)

    if show:
        backend = plt.get_backend().lower()
        if "agg" in backend:
            print(
                "Matplotlib is using a non-interactive backend "
                f"({plt.get_backend()}). The figure was created but cannot be shown."
            )
        else:
            plt.show()

    return figure, axis


def build_bigram_tsne_projection(
    data=None,
    dataset_name="test",
    sample_size=600,
    ngram_range=(1, 2),
    min_df=3,
    pca_components=50,
    perplexity=30,
    random_state=DEFAULT_RANDOM_STATE
):
    frame = _prepare_visualization_frame(data=data, dataset_name=dataset_name)
    frame = _sample_frame(frame, sample_size=sample_size, random_state=random_state)

    texts = frame["text"].tolist()
    labels = frame["label"].to_numpy()

    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        min_df=min_df
    )
    features = vectorizer.fit_transform(texts).toarray()

    effective_pca_components = _resolve_pca_components(features, pca_components)
    if effective_pca_components is not None and effective_pca_components < features.shape[1]:
        features = PCA(n_components=effective_pca_components, random_state=random_state).fit_transform(features)

    coordinates = _fit_tsne(features, perplexity=perplexity, random_state=random_state)

    return TsneProjectionResult(
        coordinates=coordinates,
        labels=labels,
        texts=texts,
        title=f"t-SNE visualization of {dataset_name} texts (CountVectorizer bigrams)",
        sample_size=len(texts)
    )


def build_bertweet_tsne_projection(
    data=None,
    dataset_name="test",
    sample_size=1500,
    batch_size=16,
    max_length=128,
    perplexity=30,
    random_state=DEFAULT_RANDOM_STATE,
    tokenizer=None,
    model=None,
    device=None,
    debug=False
):
    frame = _prepare_visualization_frame(data=data, dataset_name=dataset_name)
    frame = _sample_frame(
        frame,
        sample_size=sample_size,
        stratify=True,
        random_state=random_state
    )

    texts = frame["text"].tolist()
    labels = frame["label"].to_numpy()

    if tokenizer is None or model is None or device is None:
        tokenizer, model, device = load_model(debug=debug)

    embeddings = extract_embeddings(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        debug=debug
    )

    coordinates = _fit_tsne(embeddings, perplexity=perplexity, random_state=random_state)

    return TsneProjectionResult(
        coordinates=coordinates,
        labels=labels,
        texts=texts,
        title=f"t-SNE of BERTweet embeddings on {dataset_name}.csv",
        sample_size=len(texts)
    )


def create_bigram_tsne_plot(
    data=None,
    dataset_name="test",
    sample_size=600,
    ngram_range=(1, 2),
    min_df=3,
    pca_components=50,
    perplexity=30,
    random_state=DEFAULT_RANDOM_STATE,
    figure_size=DEFAULT_FIGSIZE,
    alpha=0.7,
    point_size=20,
    output_path=None,
    show=False
):
    projection = build_bigram_tsne_projection(
        data=data,
        dataset_name=dataset_name,
        sample_size=sample_size,
        ngram_range=ngram_range,
        min_df=min_df,
        pca_components=pca_components,
        perplexity=perplexity,
        random_state=random_state
    )
    plot_tsne_projection(
        projection,
        figure_size=figure_size,
        alpha=alpha,
        point_size=point_size,
        output_path=output_path,
        show=show
    )
    return projection


def create_bertweet_tsne_plot(
    data=None,
    dataset_name="test",
    sample_size=1500,
    batch_size=16,
    max_length=128,
    perplexity=30,
    random_state=DEFAULT_RANDOM_STATE,
    tokenizer=None,
    model=None,
    device=None,
    debug=False,
    figure_size=DEFAULT_FIGSIZE,
    alpha=0.7,
    point_size=20,
    output_path=None,
    show=False
):
    projection = build_bertweet_tsne_projection(
        data=data,
        dataset_name=dataset_name,
        sample_size=sample_size,
        batch_size=batch_size,
        max_length=max_length,
        perplexity=perplexity,
        random_state=random_state,
        tokenizer=tokenizer,
        model=model,
        device=device,
        debug=debug
    )
    plot_tsne_projection(
        projection,
        figure_size=figure_size,
        alpha=alpha,
        point_size=point_size,
        output_path=output_path,
        show=show
    )
    return projection
