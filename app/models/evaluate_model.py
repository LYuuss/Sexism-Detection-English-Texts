from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Evaluate a trained model on a test set
def evaluate_bayes_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    _print_metrics(
        name,
        accuracy_score(y_test, y_pred),
        classification_report(y_test, y_pred, digits=4),
        confusion_matrix(y_test, y_pred)
    )

    return model, y_pred

def evaluate_huggingface_bertweet_model(
    texts,
    y_true,
    batch_size=16,
    max_length=128,
    debug=False,
    name="BERTweet-large sexism detector",
    tokenizer=None,
    bertweet_model=None,
    device=None
):
    from .huggingface_bertweet import load_model, predict_texts

    if tokenizer is None or bertweet_model is None or device is None:
        tokenizer, bertweet_model, device = load_model(debug=debug)

    y_pred = predict_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=bertweet_model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        debug=debug
    )

    _print_metrics(
        name,
        accuracy_score(y_true, y_pred),
        classification_report(y_true, y_pred, digits=4),
        confusion_matrix(y_true, y_pred)
    )

    return bertweet_model, y_pred

def evaluate(
    name=None, model=None,
    X_train=None, X_test=None,
    y_train=None, y_test=None,
    texts=None, y_true=None,
    batch_size=16, max_length=128,
    debug=False,
    tokenizer=None,
    bertweet_model=None,
    device=None
):
    if model is not None:

        required_args = {
            "name": name,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }
        missing_args = [arg_name for arg_name, value in required_args.items() if value is None]
        if missing_args:
            raise ValueError(
                "Bayes evaluation requires the following arguments: "
                f"{', '.join(missing_args)}."
            )

        return evaluate_bayes_model(name, model, X_train, X_test, y_train, y_test)

    if texts is not None or y_true is not None:
        required_args = {
            "texts": texts,
            "y_true": y_true,
        }
        missing_args = [arg_name for arg_name, value in required_args.items() if value is None]
        if missing_args:
            raise ValueError(
                "BERTweet evaluation requires the following arguments: "
                f"{', '.join(missing_args)}."
            )

        return evaluate_huggingface_bertweet_model(
            texts=texts,
            y_true=y_true,
            batch_size=batch_size,
            max_length=max_length,
            debug=debug,
            name=name or "BERTweet-large sexism detector",
            tokenizer=tokenizer,
            bertweet_model=bertweet_model,
            device=device
        )

    raise ValueError(
        "Unable to choose an evaluation method. "
        "Provide either a trained model with train/test data, "
        "or texts with y_true for BERTweet evaluation."
    )

def _print_metrics(name, accuracy, classification_report_str, confusion_matrix_array):
    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")
    print("Accuracy:", accuracy)
    print("\nClassification report:")
    print(classification_report_str)
    print("Confusion matrix:")
    print(confusion_matrix_array)
