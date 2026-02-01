import os
import shutil
import sys
from fastapi import FastAPI, UploadFile, File
import whisper



app = FastAPI()

whisper_model = whisper.load_model("small")

upload_dir = "uploads"

# run the fastapi app by uvicorn main:app --reload


@app.post("/voice-to-text/")
def voice_to_text_endpoint(file: UploadFile = File(...)):
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as audio_file:
        shutil.copyfileobj(file.file, audio_file)

    result = whisper_model.transcribe(file_path)
    lang = result["language"]
    transcription = result["text"]
    return {"language": lang, "transcription": transcription}


def log_mel_spectrogram(file_path: str):
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    return {"mel_spectrogram_tensor": mel}  # Just returning the shape for simplicity

