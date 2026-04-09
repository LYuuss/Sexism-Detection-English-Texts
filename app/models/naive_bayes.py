from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def _validate_ngram_range(ngram_range):
    if ngram_range is None:
        return

    if not isinstance(ngram_range, tuple):
        raise TypeError("ngram_range must be a tuple of two integers, for example (1, 2).")

    if len(ngram_range) != 2:
        raise ValueError("ngram_range must contain exactly two integers: (min_n, max_n).")

    min_n, max_n = ngram_range
    if not isinstance(min_n, int) or not isinstance(max_n, int):
        raise TypeError("ngram_range must contain only integers.")

    if min_n < 1 or max_n < 1:
        raise ValueError("ngram_range values must be greater than or equal to 1.")

    if min_n > max_n:
        raise ValueError("ngram_range must satisfy min_n <= max_n.")


def _validate_min_df(min_df):
    if min_df is None:
        return

    if isinstance(min_df, bool):
        raise TypeError("min_df must be an integer >= 1 or a float between 0.0 and 1.0.")

    if isinstance(min_df, int):
        if min_df < 1:
            raise ValueError("min_df must be greater than or equal to 1 when it is an integer.")
        return

    if isinstance(min_df, float):
        if not 0.0 <= min_df <= 1.0:
            raise ValueError("min_df must be between 0.0 and 1.0 when it is a float.")
        return

    raise TypeError("min_df must be an integer >= 1 or a float between 0.0 and 1.0.")

vectorizer_types = ["count", "tfidf"]

def build_pipeline(vectorizer_type="count", ngram_range=None, min_df=None):
    _validate_ngram_range(ngram_range)
    _validate_min_df(min_df)

    if vectorizer_type not in vectorizer_types:
        raise ValueError(f"Invalid vectorizer type. Choose from {vectorizer_types}.")
    
    if vectorizer_type == "count":
        vectorizer_class = CountVectorizer
    elif vectorizer_type == "tfidf":
        vectorizer_class = TfidfVectorizer

    vectorizer_kwargs = {}
    if ngram_range is not None:
        vectorizer_kwargs["ngram_range"] = ngram_range
    if min_df is not None:
        vectorizer_kwargs["min_df"] = min_df

    pipeline = Pipeline([
        ("vectorizer", vectorizer_class(**vectorizer_kwargs)),
        ("classifier", MultinomialNB())
    ])

    return pipeline
