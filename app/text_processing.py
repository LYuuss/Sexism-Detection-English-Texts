import pandas as pd

def map_data(data):
    return data["label_sexist"].map({
        "not sexist": 0,
        "sexist": 1
    })