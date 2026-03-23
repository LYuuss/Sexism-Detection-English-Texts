import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

"""
The csv fields are as follow :
    rewire_id: Unique identifier for each entry.
    text: The English text for analysis.
    label_sexist: Classification indicating whether the text is sexist (not sexist, 75.7%).
    label_category: The category label, which is “none” in this filtered dataset (75.7%).
    label_vector: Represents a vectorization of the text; currently marked as “none” (75.7%).
    split: Indicates the dataset split; all entries belong to the training set (100%).
"""
train_data_path = "./dataset/train.csv"

sexism_data = pd.read_csv(train_data_path)

nb_sexist = sexism_data[sexism_data["label_sexist"] == "sexist" ]


text_cat = ["text", "label_sexist"]
raw_text = sexism_data["text"]


text_cat_data = sexism_data[text_cat]
print("Sample of the data:")
print(text_cat_data.head())

print("\nPartition of sexist texts:")
partition_of_sexism = str(len(nb_sexist)) + " of sexist texts over " + str(len(sexism_data)) + " texts"
print("\n"+partition_of_sexism+ "\n")

# Highlight frequents word in sexist section (without stopword)
sexist_texts = sexism_data[sexism_data["label_sexist"] == "sexist"]["text"]

vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(sexist_texts)

word_counts = np.asarray(X.sum(axis=0)).ravel()
feature_names = vectorizer.get_feature_names_out()

top_indices = word_counts.argsort()[::-1][:30]

print("Top 30 most frequent words in sexist texts:")
for idx in top_indices:
    print(feature_names[idx], word_counts[idx])