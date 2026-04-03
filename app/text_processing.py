import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import string

def map_data(data):
    return data["label_sexist"].map({
        "not sexist": 0,
        "sexist": 1
    })
    
def download_nltk_stopwords():
    nltk.download("stopwords")
    
def set_stopwords(language):
    return set(stopwords.words(language))

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
    
    download_nltk_stopwords()
    stop_words = set_stopwords(language)

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

        tokens = [stemmer.stem(word) for word in tokens]

        processed_texts.append(" ".join(tokens))

    return processed_texts