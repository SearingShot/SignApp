import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_PATH = "./speechCleaner_t5_model"

# Load tokenizer & model
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def remove_disfluency(text: str) -> str:
    inputs = tokenizer(
        "clean speech: " + text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
    cleaned_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return cleaned_text.strip()

# Test the disfluency removal on some example sentences
if __name__ == "__main__":
    text = "I uh want to go to the store"
    print(remove_disfluency(text))
