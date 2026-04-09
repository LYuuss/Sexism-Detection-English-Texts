from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .models.huggingface_bertweet import load_model, predict_texts
from .models.naive_bayes import build_pipeline
from .text_processing import map_data, missing_values_handling, preprocess_texts


SEXISM_LABELS = {0: "not sexist", 1: "sexist"}


@dataclass(frozen=True)
class MethodSpec:
    key: str
    name: str
    description: str
    kind: str
    vectorizer_type: str | None = None
    ngram_range: tuple[int, int] | None = None
    min_df: int | float | None = None


@dataclass(frozen=True)
class EvaluationReport:
    method_key: str
    method_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: list[list[int]]


@dataclass(frozen=True)
class PredictionResult:
    method_key: str
    method_name: str
    label: str


METHOD_SPECS = [
    MethodSpec(
        key="nb_count_unigram",
        name="Naive Bayes Count Unigram",
        description="Baseline bag-of-words with count vectors.",
        kind="naive_bayes",
        vectorizer_type="count",
        ngram_range=(1, 1),
    ),
    MethodSpec(
        key="nb_count_bigram",
        name="Naive Bayes Count Bigram",
        description="Count vectors with unigrams and bigrams.",
        kind="naive_bayes",
        vectorizer_type="count",
        ngram_range=(1, 2),
        min_df=3,
    ),
    MethodSpec(
        key="nb_tfidf_unigram",
        name="Naive Bayes TFIDF Unigram",
        description="TF-IDF vectorizer on unigrams.",
        kind="naive_bayes",
        vectorizer_type="tfidf",
        ngram_range=(1, 1),
    ),
    MethodSpec(
        key="nb_tfidf_bigram",
        name="Naive Bayes TFIDF Bigram",
        description="TF-IDF vectorizer with unigrams and bigrams.",
        kind="naive_bayes",
        vectorizer_type="tfidf",
        ngram_range=(1, 2),
        min_df=3,
    ),
    MethodSpec(
        key="bertweet_pretrained",
        name="BERTweet Pretrained",
        description="Pretrained Hugging Face sexism detector.",
        kind="bertweet",
    ),
]
METHODS_BY_KEY = {method.key: method for method in METHOD_SPECS}


class ModelWorkbench:
    def __init__(self, debug=False):
        self.debug = debug
        self._trained_models: dict[tuple[str, str], object] = {}
        self._bertweet_bundle: tuple[object, object, object] | None = None

    def set_debug(self, debug: bool) -> None:
        debug = bool(debug)
        if self.debug != debug:
            self.debug = debug
            self._bertweet_bundle = None

    def evaluate_methods(
        self,
        method_keys: list[str],
        dataset_paths: dict[str, Path],
    ) -> list[EvaluationReport]:
        reports = []
        for method_key in method_keys:
            spec = METHODS_BY_KEY[method_key]
            if spec.kind == "naive_bayes":
                reports.append(self._evaluate_naive_bayes(spec, dataset_paths))
            elif spec.kind == "bertweet":
                reports.append(self._evaluate_bertweet(spec, dataset_paths))
            else:
                raise ValueError(f"Unsupported method kind: {spec.kind}")

        return reports

    def predict_text(
        self,
        method_keys: list[str],
        text: str,
        dataset_paths: dict[str, Path],
    ) -> list[PredictionResult]:
        if not text.strip():
            raise ValueError("Input text must not be empty.")

        results = []
        for method_key in method_keys:
            spec = METHODS_BY_KEY[method_key]
            if spec.kind == "naive_bayes":
                model = self._get_or_train_naive_bayes(spec, dataset_paths["train"])
                processed_text = preprocess_texts([text], debug=self.debug)
                prediction = int(model.predict(processed_text)[0])
            elif spec.kind == "bertweet":
                tokenizer, bertweet_model, device = self._get_bertweet_bundle()
                prediction = int(
                    predict_texts(
                        texts=[text],
                        tokenizer=tokenizer,
                        model=bertweet_model,
                        device=device,
                        debug=self.debug,
                    )[0]
                )
            else:
                raise ValueError(f"Unsupported method kind: {spec.kind}")

            results.append(
                PredictionResult(
                    method_key=spec.key,
                    method_name=spec.name,
                    label=SEXISM_LABELS[prediction],
                )
            )

        return results

    def _evaluate_naive_bayes(
        self,
        spec: MethodSpec,
        dataset_paths: dict[str, Path],
    ) -> EvaluationReport:
        train_frame = self._read_dataset(dataset_paths["train"])
        test_frame = self._read_dataset(dataset_paths["test"])

        X_train = preprocess_texts(train_frame["text"], debug=self.debug)
        X_test = preprocess_texts(test_frame["text"], debug=self.debug)
        y_train = map_data(train_frame)
        y_test = map_data(test_frame)

        model = self._get_or_train_naive_bayes(spec, dataset_paths["train"], X_train, y_train)
        y_pred = model.predict(X_test)

        return self._build_report(spec, y_test, y_pred)

    def _evaluate_bertweet(
        self,
        spec: MethodSpec,
        dataset_paths: dict[str, Path],
    ) -> EvaluationReport:
        test_frame = self._read_dataset(dataset_paths["test"])
        texts = missing_values_handling(test_frame)
        y_test = map_data(test_frame)

        tokenizer, bertweet_model, device = self._get_bertweet_bundle()
        y_pred = predict_texts(
            texts=texts,
            tokenizer=tokenizer,
            model=bertweet_model,
            device=device,
            debug=self.debug,
        )

        return self._build_report(spec, y_test, y_pred)

    def _get_or_train_naive_bayes(
        self,
        spec: MethodSpec,
        train_path: Path,
        X_train: list[str] | None = None,
        y_train=None,
    ):
        cache_key = (spec.key, self._dataset_signature(train_path))
        if cache_key in self._trained_models:
            return self._trained_models[cache_key]

        if X_train is None or y_train is None:
            train_frame = self._read_dataset(train_path)
            X_train = preprocess_texts(train_frame["text"], debug=self.debug)
            y_train = map_data(train_frame)

        model = build_pipeline(
            vectorizer_type=spec.vectorizer_type or "count",
            ngram_range=spec.ngram_range,
            min_df=spec.min_df,
        )
        model.fit(X_train, y_train)
        self._trained_models[cache_key] = model
        return model

    def _get_bertweet_bundle(self) -> tuple[object, object, object]:
        if self._bertweet_bundle is None:
            self._bertweet_bundle = load_model(debug=self.debug)
        return self._bertweet_bundle

    def _read_dataset(self, path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        if "text" not in frame.columns or "label_sexist" not in frame.columns:
            raise ValueError(f"Dataset {path} must contain text and label_sexist columns.")
        return frame

    def _dataset_signature(self, path: Path) -> str:
        stats = path.stat()
        return f"{path.resolve()}::{stats.st_mtime_ns}"

    def _build_report(self, spec: MethodSpec, y_true, y_pred) -> EvaluationReport:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        confusion = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        return EvaluationReport(
            method_key=spec.key,
            method_name=spec.name,
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision,
            recall=recall,
            f1=f1,
            confusion=confusion,
        )
