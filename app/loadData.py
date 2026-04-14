import pandas as pd

train_data_path = "./dataset/train.csv"
test_data_path = "./dataset/test.csv"
dev_data_path = "./dataset/dev.csv"

def load_file(name_file):
    if name_file == "train":
        data = pd.read_csv(train_data_path)
    elif name_file == "test":
        data = pd.read_csv(test_data_path)
    elif name_file == "dev":
        data = pd.read_csv(dev_data_path)
    else:
        raise ValueError("Invalid file name. Use 'train', 'test', or 'dev'.")
    return data

def get_raw_text(data):
    return data["text"]

def get_sexist_rows(data):
    return data[data["label_sexist"] == "sexist"]

def get_not_sexist_rows(data):
    return data[data["label_sexist"] == "not sexist"]

def print_partition_of_sexism(data):
    only_sexist = len(get_sexist_rows(data))
    total_texts = len(data)
    partition_of_sexism = f"{only_sexist} of sexist texts over {total_texts} texts"
    print(partition_of_sexism)

