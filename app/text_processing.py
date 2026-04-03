import pandas as pd

import nltk
from nltk.stem import PorterStemmer

import string

from .stopwords import set_stopwords, download_nltk_stopwords, get_custom_stopwords

def map_data(data):
    return data["label_sexist"].map({
        "not sexist": 0,
        "sexist": 1
    })

# Processing the inputs
def preprocess_texts(texts, language="english"):
    """
    Preprocess list of texts:
    - lowercase
    - remove punctuation
    - remove stopwords
    - apply stemming
    - return cleaned texts as a list of strings
    """
    if language != "custom":
        download_nltk_stopwords()
        stop_words = set_stopwords(language)
    else:
        stop_words = get_custom_stopwords()

    punctuation = set(string.punctuation)
    stemmer = PorterStemmer()

    processed_texts = []

    print("")
    for text in texts:
        if pd.isna(text):
            processed_texts.append("")
            continue

        cleaned = "".join(ch if ch not in punctuation else " " for ch in text.lower())

        tokens = cleaned.split()

        tokens = [word for word in tokens if word not in stop_words]

        if language != "custom":
            tokens = [stemmer.stem(word) for word in tokens]
        else:
            tokens = [word for word in tokens if word not in get_custom_stopwords()]

        processed_texts.append(" ".join(tokens))

    return processed_texts