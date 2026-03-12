import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pymongo import MongoClient

load_dotenv()

# ── App ────────────────────────────────────────────────────────────
app = FastAPI(title="SignApp", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ─────────────────────────────────────────────────────────
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

UI_DIR = Path(__file__).parent / "ui"

# ── MongoDB ────────────────────────────────────────────────────────
client = MongoClient(MONGODB_URI)
db = client["SignApp"]
sign_rules_col = db["sign_rules"]
fingerspell_col = db["fingerspelling"]

# ── Lazy-loaded models ─────────────────────────────────────────────
_whisper_model = None
_disfluency_fn = None


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    return _whisper_model


def get_disfluency_fn():
    global _disfluency_fn
    if _disfluency_fn is None:
        from .disfluency.inference import remove_disfluency
        _disfluency_fn = remove_disfluency
    return _disfluency_fn


# Always import the new gloss converter (it's lightweight)
from .sign_language_text.gloss_converter import convert_to_sign_gloss


# ── Schemas ────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str


# ── Helpers ────────────────────────────────────────────────────────
def build_sign_sequence(gloss_tokens: list[str]) -> list[dict]:
    """Look up each gloss token in MongoDB sign_rules, fall back to fingerspelling."""
    sign_sequence = []

    for word in gloss_tokens:
        rule = sign_rules_col.find_one({"sign": word})

        if rule:
            sign_sequence.append({
                "type": "sign",
                "gloss": word,
                "handshape": rule["handshape"],
                "location": rule["location"],
                "movement": rule["movement"],
            })
        else:
            # Fingerspell each letter
            for letter in word:
                finger = fingerspell_col.find_one({"letter": letter.upper()})
                if finger:
                    sign_sequence.append({
                        "type": "fingerspell",
                        "letter": letter.upper(),
                        "handshape": finger["handshape"],
                        "location": "neutral_space",
                        "movement": finger.get("movement") or "none",
                    })

    return sign_sequence


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/voice-to-text/")
def voice_to_text_endpoint(file: UploadFile = File(...)):
    """Full pipeline: audio → transcription → gloss → sign sequence."""
    file_path = UPLOAD_DIR / (file.filename or "recording.webm")

    try:
        with open(file_path, "wb") as audio_file:
            shutil.copyfileobj(file.file, audio_file)

        # Whisper transcription
        whisper_model = get_whisper()
        result = whisper_model.transcribe(str(file_path), language="en")
        transcription = result["text"]
        language = result["language"]

        # Disfluency removal
        disfluency_fn = get_disfluency_fn()
        cleaned_text = disfluency_fn(transcription)

        # Gloss conversion (new hybrid converter)
        sign_friendly_text = convert_to_sign_gloss(cleaned_text)

        # MongoDB lookup
        sign_sequence = build_sign_sequence(sign_friendly_text)

        return {
            "language": language,
            "raw_transcription": transcription,
            "cleaned_transcription": cleaned_text,
            "sign_friendly_text": sign_friendly_text,
            "sign_sequence": sign_sequence,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up uploaded file
        if file_path.exists():
            file_path.unlink()


@app.post("/text-to-sign/")
def text_to_sign_endpoint(body: TextInput):
    """Text-only pipeline (skip audio/whisper). Useful for testing."""
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    # Disfluency removal
    disfluency_fn = get_disfluency_fn()
    cleaned_text = disfluency_fn(text)

    # Gloss conversion
    sign_friendly_text = convert_to_sign_gloss(cleaned_text)

    # MongoDB lookup
    sign_sequence = build_sign_sequence(sign_friendly_text)

    return {
        "cleaned_transcription": cleaned_text,
        "sign_friendly_text": sign_friendly_text,
        "sign_sequence": sign_sequence,
    }


# ── Static file serving for UI ─────────────────────────────────────

@app.get("/")
def serve_ui():
    return FileResponse(UI_DIR / "index.html")


# Mount static files AFTER the routes so "/" route takes priority
app.mount("/", StaticFiles(directory=str(UI_DIR)), name="ui")
