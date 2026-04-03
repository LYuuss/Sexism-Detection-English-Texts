from app.loadData import *
from app.text_processing import *

if __name__ == "__main__":
    data = load_file("train")
    print(get_raw_text(data).head())
    print(get_sexist_rows(data).head())
    
    print("\nPartition of sexist texts:")
    print_partition_of_sexism(data)
    
    print("\nMapped data:")
    mapped_data = map_data(data)
    print(mapped_data.head())
    
    print("\nPreprocessed texts:")
    preprocessed_texts = preprocess_texts(get_raw_text(data))
    print(preprocessed_texts[:25])