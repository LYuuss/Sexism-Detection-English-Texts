import pandas as pd

from nltk.stem import PorterStemmer

import string

from .stopwords import set_stopwords, download_nltk_stopwords, get_custom_stopwords

def map_data(data):
    return data["label_sexist"].map({
        "not sexist": 0,
        "sexist": 1
    })
    
def missing_values_handling(data):
    return data["text"].fillna("").tolist()

def _get_stop_words(language, debug=False):
    if language == "custom":
        return set(get_custom_stopwords())

    download_nltk_stopwords(debug=debug)
    return set_stopwords(language)


# Processing the inputs
def preprocess_texts(texts, language="english", stemming=True, debug=False):
    """
    Preprocess list of texts:
    - lowercase
    - remove punctuation
    - remove stopwords
    - optionally apply stemming
    - return cleaned texts as a list of strings
    """
    if not isinstance(stemming, bool):
        raise TypeError("stemming must be a boolean.")

    stop_words = _get_stop_words(language, debug=debug)

    punctuation = set(string.punctuation)
    stemmer = PorterStemmer() if stemming else None

    processed_texts = []

    for text in texts:
        if pd.isna(text):
            processed_texts.append("")
            continue

        cleaned = "".join(ch if ch not in punctuation else " " for ch in text.lower())

        tokens = cleaned.split()

        tokens = [word for word in tokens if word not in stop_words]

        if stemming:
            tokens = [stemmer.stem(word) for word in tokens]

        processed_texts.append(" ".join(tokens))

    return processed_texts
