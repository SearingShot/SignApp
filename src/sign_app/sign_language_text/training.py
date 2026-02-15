from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import mlflow
import evaluate
import nltk

nltk.download('punkt')
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

base_model = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(base_model)
transformer_model = T5ForConditionalGeneration.from_pretrained(base_model)

sign_language_conversion_dataset = load_dataset("achrafothman/aslg_pc12")

sign_language_conversion_dataset = load_dataset("achrafothman/aslg_pc12")

# Split 90% train, 10% validation
sign_language_conversion_dataset = sign_language_conversion_dataset["train"].train_test_split(test_size=0.1)


def sign_friendly_mapping(example):
    return {
        "input_text": example["text"],
        "target_text": example["gloss"]
    }

Dataset = sign_language_conversion_dataset.map(
    sign_friendly_mapping,
    remove_columns=sign_language_conversion_dataset["train"].column_names
)

def is_valid(example):
    return (
        example["input_text"] is not None
        and example["target_text"] is not None
        and example["input_text"].strip() != ""
        and example["target_text"].strip() != ""
    )

Dataset = Dataset.filter(is_valid)

max_input_length = 256
max_target_length = 256

def tokenize_function(examples):
    inputs = ["convert to sign-friendly: " + text for text in examples["input_text"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")
    labels = tokenizer(
        examples["target_text"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = Dataset.map(
    tokenize_function, batched=True, remove_columns=Dataset["train"].column_names
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=transformer_model)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = [[label if label != -100 else tokenizer.pad_token_id for label in l] for l in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    bleu_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "bleu": bleu_result["bleu"],
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"]
    }

training_args = Seq2SeqTrainingArguments(
    output_dir="./sign_language_converter_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    predict_with_generate=True,
    save_total_limit=2,
    logging_steps=100,
    fp16=True, # set to True if using a GPU with mixed precision support
    report_to="mlflow",
)

trainer = Seq2SeqTrainer(
    model=transformer_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("Sign Language Text Conversion")
with mlflow.start_run(run_name="T5 Sign Language Converter"):
    trainer.train()
    trainer.save_model("./sign_language_converter_model")
    tokenizer.save_pretrained("./sign_language_converter_model")

# Test the trained model on some example sentences
def convert_to_sign_friendly(text: str) -> str:
    inputs = tokenizer("convert to sign-friendly: " + text, return_tensors="pt", truncation=True).input_ids.to(transformer_model.device)
    outputs = transformer_model.generate(inputs, max_length=max_target_length, num_beams=4, early_stopping=True)
    sign_friendly_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sign_friendly_text.strip()

if __name__ == "__main__":
    test_sentence = "I want to go to the store"
    print("Original:", test_sentence)
    print("Sign-friendly:", convert_to_sign_friendly(test_sentence))
