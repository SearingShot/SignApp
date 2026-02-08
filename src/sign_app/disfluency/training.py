from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import mlflow
import evaluate
import nltk
nltk.download('punkt')
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

base_model = "t5-small"  # You can choose a larger model like "t5-base" or "t5-large" if you have the resources

tokenizer = T5Tokenizer.from_pretrained(base_model)
transformer_model = T5ForConditionalGeneration.from_pretrained(base_model)

switchboard_dataset = load_dataset("amaai-lab/DisfluencySpeech")

def keep_only_text_columns(example):
    return {
        "input_text": example["transcript_a"],
        "target_text": example["transcript_c"]
    }

dataset = switchboard_dataset.map(
    keep_only_text_columns,
    remove_columns=switchboard_dataset["train"].column_names
)

def is_valid(example):
    return (
        example["input_text"] is not None
        and example["target_text"] is not None
        and example["input_text"].strip() != ""
        and example["target_text"].strip() != ""
    )

dataset = dataset.filter(is_valid)

encoding_max_length = 256
decoding_max_length = 256

def tokenize(sentences):
    inputs = ["clean speech: " + text for text in sentences["input_text"]]

    model_inputs = tokenizer(inputs, max_length=encoding_max_length, truncation=True, padding="max_length")
    labels = tokenizer(
        sentences["target_text"],
        max_length=decoding_max_length,
        truncation=True,
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(
    tokenize,batched=True,remove_columns=dataset["train"].column_names
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=transformer_model)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = [[label if label != -100 else tokenizer.pad_token_id for label in l] for l in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    bleu_result = bleu.compute(predictions=[pred.split() for pred in decoded_preds], references=[[labels.split()] for labels in decoded_labels])
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "bleu": bleu_result["bleu"],
        "rouge1": rouge_result["rouge1"],
        "rouge2": rouge_result["rouge2"],
        "rougeL": rouge_result["rougeL"]
    }

training_args = Seq2SeqTrainingArguments(
    output_dir="./speechCleaner_t5_model",
    eval_strategy="no",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_steps=100,
    save_total_limit=2,
    fp16=False,  # Set to True if you have a compatible GPU
    report_to="mlflow"
)

trainer = Seq2SeqTrainer(
    model=transformer_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("speechCleaner_t5_model")
with mlflow.start_run():
    trainer.train()
    trainer.save_model("./SpeechCleaner_t5_model")
    tokenizer.save_pretrained("./SpeechCleaner_t5_model")

def clean_text(text: str) -> str:
    inputs = tokenizer("clean speech: " + text, return_tensors="pt", truncation=True).input_ids.to(transformer_model.device)
    outputs = transformer_model.generate(inputs, max_length=decoding_max_length, num_beams=4, early_stopping=True)
    cleaned_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return cleaned_text

print(clean_text("Yeah uh I I don't work but I used to work when I had two children"))
print(clean_text("I want to go to the store um to buy some groceries"))
print(clean_text("So uh the meeting is scheduled for uh next Monday at 10 am"))
