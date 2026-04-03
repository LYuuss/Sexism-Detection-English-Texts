from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Evaluate a trained model on a test set
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

    return model, y_pred