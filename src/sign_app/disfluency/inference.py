import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification

MODEL_PATH = "disfluency_model" 

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH)
model = RobertaForTokenClassification.from_pretrained(MODEL_PATH)


id2label = {0: "O", 1: "DIS"}
label2id = {"O": 0, "DIS": 1}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()



def remove_disfluency(text: str) -> str:

    words = text.split()

    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()

    word_ids = inputs.word_ids(batch_index=0)

    word_predictions = {}
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id not in word_predictions:
            word_predictions[word_id] = predictions[idx]

    cleaned_words = [
        word for i, word in enumerate(words)
        if id2label[word_predictions.get(i, label2id["O"])] == "O"
    ]

    return " ".join(cleaned_words)


if __name__ == "__main__":
    text = "I uh want to go to the store"
    print(remove_disfluency(text))
