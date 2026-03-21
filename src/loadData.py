import pandas as pd 

"""
    rewire_id: Unique identifier for each entry.
    text: The English text for analysis.
    label_sexist: Classification indicating whether the text is sexist (not sexist, 75.7%).
    label_category: The category label, which is “none” in this filtered dataset (75.7%).
    label_vector: Represents a vectorization of the text; currently marked as “none” (75.7%).
    split: Indicates the dataset split; all entries belong to the training set (100%).
"""
train_data_path = "./dataset/train.csv"

sexism_data = pd.read_csv(train_data_path)

text_cat = ["text", "label_sexist"]
raw_text = sexism_data["text"]

text_cat_data = sexism_data[text_cat]

print(text_cat_data.head())

# map sexist -> 1 and non_sexist -> 0
label = text_cat_data["label_sexist"].map({
    "not sexist": 0,
    "sexist": 1
})


