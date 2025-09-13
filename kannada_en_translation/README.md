# Kannada-to-English Multimodal Translator

## 📌 Overview
This project translates Kannada to English using *three input modes*:
1. 📷 Images (OCR via Tesseract)
2. 🎤 Audio (ASR via Whisper)
3. 📝 Text (direct input)

The translation is powered by a *fine-tuned MT5 model* on the ai4bharat/Samanantar dataset.

---

## 🚀 Setup

bash
git clone https://github.com/yourusername/kannada_en_translation.git
cd kannada_en_translation
pip install -r requirements.txt
sudo apt-get install -y tesseract-ocr-kan
sudo apt install ffmpeg
 
---

## Model training
Run the training script to fine-tune the MT5 model on the Samanantar dataset
bash
python train_mt5.py

Model checkpoints will be saved in ./mt5-kannada-en.
You can configure dataset size, learning rate, and epochs in config.py.

## 🔮 Run Inference

bash
python inference.py


## 🌐 Multimodal Translation
Translate from image, audio, or text inputs with a single command:
```bash
python multimodal_pipeline.py