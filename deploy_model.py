mport os
import modelbit
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

def predict_labels(text):
    label_encoding = {0: 'B-O', 1: 'B-AC', 2: 'B-LF', 3: 'I-LF'}

    tokenizer = AutoTokenizer.from_pretrained('venkateshtata/nlp_cw')
    model = AutoModelForTokenClassification.from_pretrained('venkateshtata/nlp_cw')

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = outputs.logits.argmax(dim=-1).squeeze().tolist()
    predicted_labels = [label_encoding[pred] for pred in predictions]

    tokenized_sentence = tokenizer.tokenize(text)
    token_label_pairs = list(zip(tokenized_sentence, predicted_labels))

    return token_label_pairs

if __name__ == "__main__":
    os.environ['MB_API_KEY'] = os.getenv('MB_API_KEY')
    os.environ['MB_WORKSPACE_NAME'] = os.getenv('MB_WORKSPACE_NAME')
    mb = modelbit.login()
    mb.deploy(predict_labels, python_packages=['transformers==4.34.1'])

