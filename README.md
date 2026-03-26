# SignApp — Speech & Text to 3D Sign Language Avatar

An AI-powered pipeline that converts **spoken or typed English** into **ASL (American Sign Language)** and renders it on a **3D avatar** in real time.

## 📋 Overview

SignApp bridges the communication gap between hearing and Deaf communities through a multi-stage NLP + 3D animation pipeline:

```
Voice / Text Input
       │
       ▼
┌──────────────┐     ┌────────────────┐     ┌──────────────────┐
│   Whisper    │ ──▶ │   Disfluency   │ ──▶ │  Gloss Converter │
│  (Speech→    │     │   Removal      │     │  (English → ASL  │
│   Text)      │     │   (T5 Model)   │     │   Gloss Tokens)  │
└──────────────┘     └────────────────┘     └──────────────────┘
                                                     │
                                                     ▼
                                            ┌──────────────────┐
                                            │  MongoDB Lookup  │
                                            │  (Sign Rules +   │
                                            │   Fingerspelling) │
                                            └──────────────────┘
                                                     │
                                                     ▼
                                            ┌──────────────────┐
                                            │  3D Avatar       │
                                            │  (Three.js +     │
                                            │   GLB Model)     │
                                            └──────────────────┘
```

## 🚀 Features

- **Voice Input** — Record via microphone with Voice Activity Detection (auto-stop on silence)
- **Text Input** — Type messages directly as an alternative to voice
- **Speech-to-Text** — OpenAI Whisper model with automatic language detection
- **Disfluency Removal** — Fine-tuned T5 model removes filler words, stutters, and false starts
- **Hybrid Gloss Converter** — Rule-based grammar transforms + NLTK lemmatization + 300+ word glossary + phrase matching
- **MongoDB Sign Database** — 100+ ASL sign rules with handshape, location, and movement data
- **Fingerspelling Fallback** — Unknown words are finger-spelled letter by letter (A-Z)
- **3D Avatar Animation** — Three.js + GLB model with per-bone rotation, smooth lerp interpolation, and idle breathing
- **Sign Playback** — Replay previously signed sequences
- **Docker Support** — Containerized deployment with Docker Compose

## 📁 Project Structure

```
SignApp/
├── src/
│   └── sign_app/                    # Main application package
│       ├── __init__.py
│       ├── api.py                   # FastAPI app & routes
│       ├── seed_signs.py            # MongoDB seeder (100+ signs, A-Z fingerspelling)
│       │
│       ├── audio/                   # Audio processing module
│       │   └── __init__.py
│       │
│       ├── disfluency/              # Disfluency removal module
│       │   ├── __init__.py
│       │   ├── inference.py         # T5 disfluency removal inference
│       │   └── training.py          # Model fine-tuning pipeline
│       │
│       ├── sign_language_text/      # English → ASL conversion
│       │   ├── __init__.py
│       │   ├── gloss_converter.py   # Hybrid rule-based + NLP gloss converter
│       │   ├── inference.py         # Sign language model inference
│       │   └── training.py          # Sign language model training
│       │
│       └── ui/                      # Frontend (served by FastAPI)
│           ├── __init__.py
│           ├── index.html           # Main UI page
│           ├── main.js              # Three.js scene, avatar, recording & UI logic
│           ├── signEngine.js        # Sign playback engine (movements, fingerspelling)
│           ├── fingerspellDictionary.js  # A-Z finger bone rotation data
│           └── avatar.glb           # 3D humanoid avatar model
│
├── tests/
│   ├── __init__.py
│   └── test_gloss_converter.py      # Gloss converter unit tests
│
├── docs/
│   ├── API.md                       # API endpoint reference
│   ├── DEVELOPMENT.md               # Development guide
│   └── INSTALLATION.md              # Setup instructions
│
├── .github/workflows/               # CI/CD pipelines
├── Dockerfile                       # Container build
├── docker-compose.yml               # Docker Compose (backend + volumes)
├── pyproject.toml                   # Project metadata & dependencies
├── requirements.txt                 # pip requirements
├── CONTRIBUTING.md                  # Contribution guidelines
├── LICENSE                          # MIT License
└── README.md                       # This file
```

## 🔧 Setup

### Prerequisites

- Python 3.12+
- MongoDB (local or Atlas)
- CUDA 11.8+ *(optional, for GPU acceleration)*

### Installation

1. **Clone & enter the project**:
   ```bash
   git clone <repository-url>
   cd SignApp
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**: Create a `.env` file in the project root:
   ```env
   MONGODB_URI=mongodb://localhost:27017/
   WHISPER_MODEL=small
   HOST=0.0.0.0
   PORT=8000
   ```

5. **Seed the database** (populates 100+ sign rules, A-Z fingerspelling, handshapes, locations, movements):
   ```bash
   python -m src.sign_app.seed_signs
   ```

### Running the App

```bash
uvicorn src.sign_app.api:app --reload
```

Open **http://localhost:8000** to use the UI. You can:
- 🎤 **Click the mic** to record a voice message (auto-stops after 2s of silence)
- ⌨️ **Type text** in the input field and press Enter / Send
- 🔄 **Replay** the last signed sequence

### Docker

```bash
docker-compose up --build
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/` | Serve the UI |
| `POST` | `/voice-to-text/` | Full pipeline: audio → transcription → gloss → sign sequence |
| `POST` | `/text-to-sign/` | Text-only pipeline: text → gloss → sign sequence |

### Example — Text to Sign

```bash
curl -X POST http://localhost:8000/text-to-sign/ \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, I want to help you"}'
```

Response:
```json
{
  "cleaned_transcription": "Hello, I want to help you",
  "sign_friendly_text": ["HELLO", "I", "WANT", "HELP", "YOU"],
  "sign_sequence": [
    {"type": "sign", "gloss": "HELLO", "handshape": "B", "location": "forehead", "movement": "wave"},
    {"type": "sign", "gloss": "I", "handshape": "POINT", "location": "chest", "movement": "tap"},
    ...
  ]
}
```

## 🧠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI + Uvicorn |
| **Speech-to-Text** | OpenAI Whisper (small, 774M params) |
| **Disfluency Removal** | Fine-tuned T5-base (223M params) |
| **Gloss Conversion** | NLTK WordNet Lemmatizer + rule-based transforms |
| **Sign Database** | MongoDB (sign rules, fingerspelling, handshapes) |
| **3D Rendering** | Three.js + GLB avatar model |
| **Containerization** | Docker + Docker Compose |

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Training the Disfluency Model

```bash
python -m src.sign_app.disfluency.training
```
Downloads the DisfluencySpeech dataset, fine-tunes T5, logs metrics to MLflow, and saves the model locally.

## ❓ Troubleshooting

| Issue | Solution |
|-------|---------|
| GPU out of memory | Use `WHISPER_MODEL=tiny` in `.env` |
| Model download errors | Check internet connection; models cache in `~/.cache/huggingface/` |
| MongoDB connection fails | Verify `MONGODB_URI` in `.env`; ensure MongoDB is running |
| No signs appear | Run `python -m src.sign_app.seed_signs` to populate the database |

## 📝 License

MIT — see [LICENSE](LICENSE)

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Note**: This project is in active development. Features being explored include multi-agent LLM translation, real-time WebSocket streaming, and enhanced 3D avatar expressions.
