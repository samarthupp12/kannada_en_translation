import torch
from transformers import MT5ForConditionalGeneration, Trainer, TrainingArguments, T5Tokenizer
from dataset_preparation import get_dataset
from config import MODEL_NAME, DEVICE, TRAIN_SAMPLE_SIZE, VAL_SAMPLE_SIZE, EPOCHS, LEARNING_RATE, BATCH_SIZE

# Load model & tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

# Load dataset
train_dataset, val_dataset = get_dataset(TRAIN_SAMPLE_SIZE, VAL_SAMPLE_SIZE)

# Training args
training_args = TrainingArguments(
    output_dir="./mt5-kannada-en",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_dir="./logs",
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    save_total_limit=2,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained("./mt5-kannada-en")
tokenizer.save_pretrained("./mt5-kannada-en")
