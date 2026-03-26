# SignApp — Project Structure

## Directory Layout

```
SignApp/
│
├── src/                                 # Source code
│   └── sign_app/                        # Main application package
│       ├── __init__.py                  # Package initialization
│       ├── api.py                       # FastAPI app, routes, MongoDB lookups
│       ├── seed_signs.py                # Database seeder (100+ sign rules, A-Z fingerspelling)
│       │
│       ├── audio/                       # Audio processing module
│       │   └── __init__.py
│       │
│       ├── disfluency/                  # Speech disfluency removal
│       │   ├── __init__.py
│       │   ├── inference.py             # T5-based disfluency removal (inference)
│       │   └── training.py              # Fine-tuning pipeline with MLflow tracking
│       │
│       ├── sign_language_text/          # English → ASL conversion
│       │   ├── __init__.py
│       │   ├── gloss_converter.py       # Hybrid gloss converter (rules + NLTK + glossary)
│       │   ├── inference.py             # Sign language model inference
│       │   └── training.py              # Sign language model training
│       │
│       └── ui/                          # Frontend (served as static files by FastAPI)
│           ├── __init__.py
│           ├── index.html               # Main UI page (mic button, text input, avatar viewport)
│           ├── main.js                  # Three.js scene, avatar loading, recording, UI state machine
│           ├── signEngine.js            # Sign playback engine (handshape, location, movement animations)
│           ├── fingerspellDictionary.js  # A-Z fingerspelling bone rotation data (26 letters)
│           └── avatar.glb              # 3D humanoid avatar model (GLB format)
│
├── tests/                               # Test suite
│   ├── __init__.py
│   └── test_gloss_converter.py          # Unit tests for the gloss converter
│
├── docs/                                # Documentation
│   ├── API.md                           # REST API endpoint reference
│   ├── DEVELOPMENT.md                   # Development guide & architecture notes
│   └── INSTALLATION.md                  # Detailed setup instructions
│
├── .github/
│   └── workflows/                       # CI/CD pipelines
│       ├── tests.yml                    # Automated test runner
│       └── lint.yml                     # Code quality checks
│
├── Configuration
│   ├── pyproject.toml                   # Project metadata, dependencies, tool config
│   ├── requirements.txt                 # pip requirements
│   ├── Dockerfile                       # Container image (Python 3.12-slim + ffmpeg)
│   ├── docker-compose.yml               # Docker Compose (ports, env, volumes, healthcheck)
│   ├── .gitignore                       # Git ignore rules
│   └── .env                             # Environment variables (MongoDB URI, Whisper model, etc.)
│
└── Metadata
    ├── README.md                        # Project overview & usage guide
    ├── CONTRIBUTING.md                  # Contribution guidelines
    └── LICENSE                          # MIT License
```

## Module Descriptions

### `api.py` — FastAPI Backend
The central entrypoint. Defines four routes:
- `GET /` — serves the UI (`index.html`)
- `GET /health` — health check
- `POST /voice-to-text/` — full voice pipeline (Whisper → disfluency removal → gloss → sign sequence)
- `POST /text-to-sign/` — text-only pipeline (disfluency removal → gloss → sign sequence)

Uses lazy-loaded models (Whisper, disfluency T5) and MongoDB for sign rule lookups. Serves the `ui/` directory as static files.

### `sign_language_text/gloss_converter.py` — Hybrid Gloss Converter
Converts English text to ASL gloss tokens via a multi-phase pipeline:
1. Normalize text (lowercase, strip punctuation)
2. Match multi-word phrases (`"thank you"` → `THANK-YOU`)
3. Word-level glossary lookup (300+ entries)
4. NLTK POS-tag + WordNet lemmatization (`"going"` → `"go"` → `GO`)
5. Unknown words pass through uppercase (fingerspelled by the frontend)
6. Drop filler words, articles, copulas, and auxiliaries

### `disfluency/` — Speech Cleanup
- **`inference.py`** — Loads the fine-tuned T5 model and removes disfluencies from transcriptions
- **`training.py`** — Fine-tunes T5-base on the DisfluencySpeech dataset with MLflow tracking

### `seed_signs.py` — Database Seeder
Populates MongoDB with:
- 100+ ASL sign rules (handshape + location + movement)
- 26 fingerspelling letters
- Handshape names, location names, movement names

### `ui/` — 3D Avatar Frontend
- **`main.js`** — Three.js scene (camera, lights, avatar loading), voice recording with VAD, text input, UI state machine, handshape/location/fingerspell systems
- **`signEngine.js`** — Sign playback orchestrator: sequences signs and fingerspelling, applies bone rotations and animated movements (tap, wave, circle, forward, etc.)
- **`fingerspellDictionary.js`** — Per-letter finger bone rotation data for all 26 ASL letters
- **`avatar.glb`** — Humanoid 3D model with named bones (RightHand, RightForeArm, finger joints)
