import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


"""
    This approach is the simplest.
    We will use raw inputs (raw text) that we will transform into
    numerical vectors using words counts.
    Then we will train a Multinomial Naïve Bayes classifier on these counts.
    The training is done on the file named "train.csv" in the dataset directory.
    Tested on "test.csv" file.

    The pipeline automates the vectorization and classification steps.
"""

# Pipeline
model_raw_count = Pipeline([
    ("vectorizer", CountVectorizer()),
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
    "Raw text + CountVectorizer + MultinomialNB",
    model_raw_count,
    train_raw_text, test_raw_text, train_label, test_label
)

"""
Results

2877 non-sexist texts were correctly classified

153 non-sexist texts were wrongly predicted as sexist

360 sexist texts were correctly classified

610 sexist texts were missed and predicted as non-sexist

the model produces many false negatives on the sexist class.

accuracy = 0.80925 looks decent at first glance, but accuracy is a bit misleading here because the dataset is imbalanced: there are many more non-sexist texts than sexist ones.


precision = 0.7018
When the model says “sexist”, it is right about 70% of the time.

recall = 0.3711
It only detects about 37% of the actual sexist texts.

f1 = 0.4855
This is fairly low, and confirms that performance on the sexist class is the weak point.

It does not often accuse a non-sexist text of being sexist but it also fails to catch many sexist texts

So for a task like sexism detection, this baseline is useful, but not sufficient on its own.

The classifier is influenced by the majority class

Raw word counts may not capture more subtle sexist formulations

No preprocessing means the vocabulary is noisier

sexist texts may use varied wording, making them harder to detect

A simple Bag-of-Words Naïve Bayes model achieves acceptable overall accuracy, but struggles to recover the minority “sexist” class, as shown by the low recall.

The baseline model achieved an accuracy of 80.9%, but this score must be interpreted with caution due to class imbalance. 
While the classifier performs well on non-sexist texts (F1 = 0.8829), its performance on sexist texts is much weaker (recall = 0.3711, F1 = 0.4855). 
This indicates that the model tends to favor the majority class and misses a large number of sexist instances.






"""