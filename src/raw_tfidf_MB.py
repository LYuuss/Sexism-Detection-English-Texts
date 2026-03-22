import pandas as pd
import string

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

"""
    We will keep the input texts unchanged.
    We will then transform those with Tfidf vectorize .
    Then we will train a Multinomial Naïve Bayes classifier on our processed data.
    The training is done on the file named "train.csv" in the dataset directory.
    Tested on "test.csv" file.

    The pipeline automates vectorization and classification steps.
"""

# Unquote if not installed locally, only once
# nltk.download("stopwords")

# Pipeline 
model_raw_tfidf = Pipeline([
    ("vectorizer", TfidfVectorizer()),
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

# Evaluate the model extrinsicly
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

model_raw_tfidf = evaluate_model(
    "Cleaned text + TfidfVectorizer + MultinomialNB",
    model_raw_tfidf,
    train_raw_text, test_raw_text, train_label, test_label
)