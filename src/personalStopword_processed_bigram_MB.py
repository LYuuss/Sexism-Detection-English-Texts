import pandas as pd
import string

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

"""
    We will first preprocess the inputs (raw text) : 
        - lowercase the texts
        - remove punctuation
        - remove stopwords

    We will then transform those into numerical vectors using words counts focusing on bigram.
    Then we will train a Multinomial Naïve Bayes classifier on these counts.
    The training is done on the file named "train.csv" in the dataset directory.
    Tested on "test.csv" file.

    The pipeline automates the vectorization and classification steps.
"""

# Unquote if not installed locally, only once 
# nltk.download("stopwords")

# Pipeline
model_preprocess_count =  Pipeline([
    ("vectorizer", CountVectorizer(ngram_range=(1, 2), min_df=3)),
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

# list of stopword without things like "her", "she"...
custom_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                'through', 'during', 'before', 'after', 'above', 'below', 'from', 'up', 'down',
                'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                'there', 'when', 'where', 'why', 'how', 'any', 'both', 'each', 'few', 'more',
                'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so',
                'than', 'too', 'very', 's', 'will', 'just', 'don', 'now','br']

def preprocess_texts(texts):
    """
    Preprocess list of texts:
    - lowercase
    - remove punctuation
    - remove stopwords
    - return cleaned texts as a list of strings
    """

    punctuation = set(string.punctuation)

    processed_texts = []

    for text in texts:
        if pd.isna(text):
            processed_texts.append("")
            continue

        cleaned = "".join(ch if ch not in punctuation else " " for ch in text.lower())

        tokens = cleaned.split()

        tokens = [word for word in tokens if word not in custom_stopwords]

        processed_texts.append(" ".join(tokens))

    return processed_texts

train_processed_text = preprocess_texts(train_raw_text)
test_processed_text = preprocess_texts(test_raw_text)

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

model_preprocess_count = evaluate_model(
    "Cleaned text with personal stopword list + CountVectorizer(1,2) (min_df =3) + MultinomialNB",
    model_preprocess_count,
    train_processed_text, test_processed_text, train_label, test_label
)
