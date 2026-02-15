import os
import shutil
from fastapi import FastAPI, UploadFile, File
import whisper

from disfluency.inference import remove_disfluency
from sign_app.sign_language_text.inference import convert_to_sign_friendly

app = FastAPI()

whisper_model = whisper.load_model("small")

UPLOAD_DIR = "././uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/voice-to-text/")
def voice_to_text_endpoint(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as audio_file:
        shutil.copyfileobj(file.file, audio_file)

    # Whisper transcription
    result = whisper_model.transcribe(file_path)
    transcription = result["text"]
    language = result["language"]

    # Remove disfluencies
    cleaned_text = remove_disfluency(transcription)

    sign_friendly_text = convert_to_sign_friendly(cleaned_text)

    return {
        "language": language,
        "raw_transcription": transcription,
        "cleaned_transcription": cleaned_text,
        "sign_friendly_text": sign_friendly_text
    }
