# SignApp

A comprehensive audio processing pipeline for sign language recognition that leverages state-of-the-art models for speech-to-text conversion and disfluency removal.

## 📋 Overview

SignApp processes audio input through multiple stages:

1. **Audio Transcription**: Converts audio to text using OpenAI's Whisper model
2. **Disfluency Removal**: Cleans transcriptions by removing speech disfluencies (filler words, stutters, etc.)
3. **Sign Language Processing**: Further processes cleaned text for sign language recognition (in progress)

## 🚀 Features

- FastAPI-based REST API for easy integration
- Automatic audio transcription with language detection
- Intelligent disfluency removal using transformer models
- GPU acceleration support
- MLflow integration for experiment tracking

## 📁 Project Structure

```
SignApp/
├── src/
│   └── sign_app/              # Main application package
│       ├── __init__.py
│       ├── api.py             # FastAPI application and routes
│       ├── audio/             # Audio processing module
│       │   └── __init__.py
│       ├── disfluency/        # Disfluency removal module
│       │   ├── __init__.py
│       │   ├── inference.py   # Disfluency removal inference
│       │   └── training.py    # Model training scripts
│       └── ui/                # UI interfaces (in progress)
│           └── __init__.py
├── tests/                      # Test suite
│   └── __init__.py
├── docs/                       # Documentation
├── uploads/                    # Temporary audio file storage
├── pyproject.toml             # Project configuration
├── README.md                  # This file
└── .gitignore                 # Git ignore rules
```

## 🔧 Installation

### Prerequisites

- Python 3.12+
- CUDA 11.8+ (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd SignApp
```

2. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📚 Usage

### Running the API Server

```bash
uvicorn src.sign_app.api:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Voice to Text
Convert audio file to text with automatic language detection:

```bash
curl -X POST "http://localhost:8000/voice-to-text/" \
  -F "file=@audio.mp3"
```

Response:
```json
{
  "language": "en",
  "transcription": "your transcribed text here"
}
```

### Training Disfluency Removal Model

```bash
python src/sign_app/disfluency/training.py
```

This script:
- Downloads the DisfluencySpeech dataset
- Fine-tunes a T5 model for disfluency removal
- Logs metrics to MLflow
- Saves the model locally

### Using Disfluency Removal

```python
from src.sign_app.disfluency.inference import remove_disfluency

text = "Yeah uh I I don't work but I used to work"
cleaned = remove_disfluency(text)
print(cleaned)  # "I don't work but I used to work"
```

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Structure

- `api.py`: FastAPI endpoints and audio handling
- `disfluency/inference.py`: Disfluency removal inference logic
- `disfluency/training.py`: Model training pipeline with MLflow tracking

### Dependencies

- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `openai-whisper`: Speech-to-text model
- `transformers`: Disfluency removal models
- `mlflow`: Experiment tracking
- `torch`: Deep learning framework

## 📊 Model Information

### Whisper Model
- Size: Small (774M parameters)
- Trained on: 680,000 hours of multilingual audio data
- Supports: 99 languages

### Disfluency Removal
- Base Model: T5-base (223M parameters)
- Fine-tuned on: DisfluencySpeech dataset
- Metrics: BLEU, ROUGE

## 🚀 Deployment

For production deployment:

1. Update API configuration (CORS, security, etc.)
2. Use a production ASGI server (Gunicorn + Uvicorn)
3. Set up environment variables for model paths
4. Configure MLflow tracking server for experiment management

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## ❓ Troubleshooting

### GPU Memory Issues
- Use smaller model variant: `whisper.load_model("tiny")`
- Reduce batch size in training configurations

### Model Download Errors
- Ensure stable internet connection during first run
- Models are cached in `~/.cache/huggingface/`

## 📧 Contact

[Add contact information here]

---

**Note**: This project is currently in active development. Additional features and sign language processing capabilities are being added.
