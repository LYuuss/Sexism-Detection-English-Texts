from app.loadData import *
from app.models.evaluate_model import evaluate
from app.models.huggingface_bertweet import load_model
from app.models.naive_bayes import build_pipeline
from app.text_processing import map_data, missing_values_handling, preprocess_texts


def example_evaluate_naive_bayes():
    train_data = load_file("train")
    test_data = load_file("test")

    train_texts = preprocess_texts(get_raw_text(train_data))
    test_texts = preprocess_texts(get_raw_text(test_data))
    y_train = map_data(train_data)
    y_test = map_data(test_data)

    naive_bayes_pipeline = build_pipeline(
        vectorizer_type="count",
        ngram_range=(1, 2),
        min_df=3
    )

    trained_pipeline, y_pred = evaluate(
        name="Naive Bayes example",
        model=naive_bayes_pipeline,
        X_train=train_texts,
        X_test=test_texts,
        y_train=y_train,
        y_test=y_test
    )

    return trained_pipeline, y_pred


def example_evaluate_bertweet():
    test_data = load_file("test")

    texts = missing_values_handling(test_data)
    y_true = map_data(test_data)

    tokenizer, bertweet_model, device = load_model(debug=True)

    loaded_model, y_pred = evaluate(
        name="BERTweet example",
        texts=texts,
        y_true=y_true,
        tokenizer=tokenizer,
        bertweet_model=bertweet_model,
        device=device,
        batch_size=16,
        max_length=128,
        debug=True
    )

    return loaded_model, y_pred


def example_bigram_tsne(output_path=None, show=False):
    from app.visualization.tsne import create_bigram_tsne_plot

    return create_bigram_tsne_plot(
        dataset_name="test",
        output_path=output_path,
        show=show
    )


def example_bertweet_tsne(output_path=None, show=False, debug=False):
    from app.visualization.tsne import create_bertweet_tsne_plot

    return create_bertweet_tsne_plot(
        dataset_name="test",
        output_path=output_path,
        show=show,
        debug=debug
    )


if __name__ == "__main__":
    print("Naive Bayes evaluation example:")
    example_evaluate_naive_bayes()

    print("\nBERTweet evaluation example:")
    example_evaluate_bertweet()
    
    example_bertweet_tsne(output_path="bertweet_tsne_plot.png", show=True, debug=True)
    
    example_bigram_tsne(output_path="bigram_tsne_plot.png", show=True)
