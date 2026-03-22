import pandas as pd
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB

# Unquote if not installed locally, only once 
# nltk.download("stopwords")

"""
    We will first preprocess the inputs (raw text) : 
        - lowercase the texts
        - remove punctuation
        - remove stopwords
        - stemming

    We will then transform those into numerical vectors using words counts.
    Then we will train a Multinomial Naïve Bayes classifier on these counts.
    The training is done on the file named "train.csv" in the dataset directory.
    Tested on "test.csv" file.

    The pipeline automates the vectorization and classification steps.
"""

# Pipeline
model_preprocess_count =  Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

train_data_path = "./dataset/train.csv"
test_data_path = "./dataset/test.csv"

train_sexism_data = pd.read_csv(train_data_path)
test_sexism_data = pd.read_csv(test_data_path)

train_raw_text = train_sexism_data["text"]
test_raw_text = test_sexism_data["text"]

train_label = train_sexism_data["label_sexist"].map({
    "not sexist": 0,
    "sexist": 1
})

test_label = test_sexism_data["label_sexist"].map({
    "not sexist": 0,
    "sexist": 1
})

# Processing the inputs
def preprocess_texts(texts):
    """
    Preprocess a pandas Series / list of texts:
    - lowercase
    - remove punctuation
    - remove stopwords
    - apply stemming
    - return cleaned texts as a list of strings
    """
    stop_words = set(stopwords.words("english"))
    
    punctuation = set(string.punctuation)
    stemmer = PorterStemmer()

    processed_texts = []

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

train_processed_text = preprocess_texts(train_raw_text)
test_processed_text = preprocess_texts(test_raw_text)

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model

model_preprocess_count = evaluate_model(
    "Cleaned text + CountVectorizer + MultinomialNB",
    model_preprocess_count,
    train_processed_text, test_processed_text, train_label, test_label
)
