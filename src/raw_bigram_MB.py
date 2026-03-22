import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


"""
    
    We will use raw inputs (raw text) that we will transform into numerical vectors using word counts focusing on bigrams.
    Then we will train a Multinomial Naïve Bayes classifier on these counts.
    The training is done on the file named "train.csv" in the dataset directory.
    Tested on "test.csv" file.

    The pipeline automates the vectorization and classification steps.
"""

# Pipeline
model_raw_count = Pipeline([
    ("vectorizer", CountVectorizer(ngram_range=(1, 2), min_df=3)),
    ("classifier", MultinomialNB())
])

train_data_path = "./dataset/train.csv"
test_data_path = "./dataset/test.csv"

train_sexism_data = pd.read_csv(train_data_path)
test_sexism_data = pd.read_csv(test_data_path)

train_raw_text = train_sexism_data["text"]
test_raw_text = test_sexism_data["text"]

# map sexist -> 1 and non_sexist -> 0
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

model_raw_count = evaluate_model(
    "Raw text + CountVectorizer(1, 2) min_df = 3 + MultinomialNB",
    model_raw_count,
    train_raw_text, test_raw_text, train_label, test_label
)