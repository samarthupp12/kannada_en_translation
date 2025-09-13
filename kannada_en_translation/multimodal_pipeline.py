from utils.ocr_utils import extract_text_from_image
from utils.asr_utils import speech_to_text
from utils.translation_utils import translate_kn_to_en

def multimodal_translation(input_path, input_type):
    if input_type == "image":
        kannada_text = extract_text_from_image(input_path)
    elif input_type == "audio":
        kannada_text = speech_to_text(input_path)
    elif input_type == "text":
        kannada_text = input_path
    else:
        raise ValueError("Invalid input type. Choose from 'image', 'audio', or 'text'.")

    translated_text = translate_kn_to_en(kannada_text)
    return translated_text

if __name__ == "__main__":
    print("Image Translation:", multimodal_translation("samples/new.jpg", "image"))
    print("Audio Translation:", multimodal_translation("samples/Recording.m4a", "audio"))
    print("Text Translation:", multimodal_translation("ನಾನು ವಿದ್ಯಾರ್ಥಿ", "text"))
