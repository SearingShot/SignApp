import re
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_PATH = "./sign_language_converter_model"

# Load tokenizer & model
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def convert_to_sign_friendly(text: str) -> list[str]:
    inputs = tokenizer(
        "convert to sign-friendly: " + text,
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
    sign_friendly_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sign_friendly_text = clean_gloss(sign_friendly_text)

    return sign_friendly_text

def clean_gloss(gloss: str) -> list[str]:
    words = gloss.split()

    cleaned = []
    for w in words:
        
        w = re.sub(r"[,.!]", "", w)

        if w.startswith("X-"):
            w = w.replace("X-", "")
        if w.startswith("DESC-"):
            w = w.replace("DESC-", "")

        cleaned.append(w)

    return cleaned

# Test the conversion on some example sentences
if __name__ == "__main__":
    text = "I want to go to the store"
    print(convert_to_sign_friendly(text))
    print(convert_to_sign_friendly("She is going to the park later"))
    print(convert_to_sign_friendly("Can you help me with this?"))
    print(convert_to_sign_friendly("The weather is nice today"))
