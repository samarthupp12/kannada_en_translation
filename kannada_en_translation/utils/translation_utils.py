import torch
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from config import MODEL_NAME, DEVICE

tokenizer = T5Tokenizer.from_pretrained("./mt5-kannada-en", legacy=False)
model = MT5ForConditionalGeneration.from_pretrained("./mt5-kannada-en").to(DEVICE)

def translate_kn_to_en(text):
    input_text = "translate Kannada to English: " + text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(DEVICE)
    output_ids = model.generate(input_ids, max_length=128)
    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translation
