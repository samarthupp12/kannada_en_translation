from datasets import load_dataset
from transformers import T5Tokenizer
from config import MODEL_NAME, MAX_LENGTH

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)

def preprocess_data(example):
    return {
        "input_text": "translate Kannada to English: " + example["tgt"],
        "target_text": example["src"]
    }

def tokenize_data(example):
    model_inputs = tokenizer(example["input_text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example["target_text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def get_dataset(train_size, val_size):
    dataset = load_dataset("ai4bharat/Samanantar", "kn")
    dataset = dataset.map(preprocess_data, remove_columns=["idx", "src", "tgt"])
    tokenized_dataset = dataset.map(tokenize_data, batched=True)

    train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(train_size))
    val_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(val_size))

    return train_dataset, val_dataset
