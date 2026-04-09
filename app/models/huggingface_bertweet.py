import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "NLP-LTU/bertweet-large-sexism-detector"

def device_selection(debug=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if debug:
        print("Using device:", device)
    return device

def load_model(model_name=MODEL_NAME, debug=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = device_selection(debug=debug)

    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_texts(texts, tokenizer=None, model=None, device=None, batch_size=16, max_length=128, debug=False):
    if texts is None:
        raise ValueError("texts must not be None.")

    if not isinstance(texts, (list, tuple)):
        raise TypeError("texts must be a list or tuple of strings.")

    if any(not isinstance(text, str) for text in texts):
        raise TypeError("texts must contain only strings.")

    if batch_size < 1:
        raise ValueError("batch_size must be greater than or equal to 1.")

    if max_length < 1:
        raise ValueError("max_length must be greater than or equal to 1.")

    if tokenizer is None or model is None or device is None:
        tokenizer, model, device = load_model(debug=debug)
    
    predictions = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        encodings = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        encodings = {k: v.to(device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().tolist()

        predictions.extend(batch_preds)

    return predictions
