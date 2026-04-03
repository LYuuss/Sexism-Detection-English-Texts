from app.loadData import *
from app.text_processing import *
from app.stopwords import *

if __name__ == "__main__":
    data = load_file("train")
    print(get_raw_text(data).head())
    print(get_sexist_rows(data).head())
    
    print("\nPartition of sexist texts:")
    print_partition_of_sexism(data)
    
    print("\nMapped data:")
    mapped_data = map_data(data)
    print(mapped_data.head())
    
    add_custom_stopwords(["test1", "test2"])
    print(get_custom_stopwords())
    delete_custom_stopwords(["test1"])
    print(get_custom_stopwords())
    
    print("\nPreprocessed texts:")
    preprocessed_texts = preprocess_texts(get_raw_text(data))
    print(preprocessed_texts[:25])
    
    print("\nFinal preprocessed texts with custom stopwords:")
    final_preprocessed_texts = preprocess_texts(get_raw_text(data), "custom")
    print(final_preprocessed_texts[:25])