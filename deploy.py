from transformers import AutoTokenizer, AutoModelForTokenClassification
# from huggingface_hub import HfApi, HfFolder, Repository
import os
import torch
import modelbit
mb = modelbit.login()

def predict_labels(text):
    label_encoding = {0: "B-O", 1: "B-AC", 2: "B-LF", 3: "I-LF"}

    # Load the tokenizer and model from Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained("venkateshtata/nlp_cw")
    model = AutoModelForTokenClassification.from_pretrained("venkateshtata/nlp_cw")

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Perform inference
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the highest probability predictions
    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()

    # Convert predictions to labels
    predicted_labels = [label_encoding[pred] for pred in predictions]

    # Mapping tokens to their labels
    tokenized_sentence = tokenizer.tokenize(text)
    token_label_pairs = list(zip(tokenized_sentence, predicted_labels))

    return token_label_pairs

print("Predictions Test: ", predict_labels("For this purpose the Gothenburg Young Persons Empowerment, Scale, (, GYPES, ), was, developed"))

print("Deploying")
mb.deploy(predict_labels, python_packages=['transformers==4.34.1'])

print("Deployed")
