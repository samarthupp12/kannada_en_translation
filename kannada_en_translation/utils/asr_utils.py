import whisper

model = whisper.load_model("large")

def speech_to_text(audio_path):
    result = model.transcribe(audio_path, language="kn")
    return result["text"]
