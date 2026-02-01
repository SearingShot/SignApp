# Installation & Setup Guide

## Prerequisites

- **Python**: 3.12 or higher
- **pip**: Latest version
- **Virtual Environment**: Recommended
- **Git**: For version control
- **CUDA 11.8+** (Optional): For GPU acceleration with torch/transformers

## Step-by-Step Installation

### 1. Clone Repository

```bash
git clone <repository-url>
cd SignApp
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:
```bash
pip install fastapi uvicorn openai-whisper torch transformers datasets mlflow evaluate nltk
```

### 4. Verify Installation

```bash
python -c "import sign_app; print('Installation successful!')"
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Model paths (optional)
WHISPER_MODEL=small
DISFLUENCY_MODEL_PATH=./disfluency_model

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# MLflow
MLFLOW_TRACKING_URI=file:./mlruns
```

### First Run

On first execution, the following models will be downloaded automatically:
- **Whisper Small**: ~775MB
- **Hugging Face models**: Variable size (cached in `~/.cache/huggingface/`)

Ensure you have stable internet during first run.

## Running the Application

### Start API Server

```bash
uvicorn src.sign_app.api:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at: `http://localhost:8000`

Access interactive API docs: `http://localhost:8000/docs`

### Train Models (Optional)

```bash
python src/sign_app/disfluency/training.py
```

This will:
- Download the DisfluencySpeech dataset
- Fine-tune the T5 model
- Log results to MLflow
- Save models to `./SpeechCleaner_t5_model/`

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'sign_app'`:

```bash
# Ensure you're in the project root
cd SignApp
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Linux/Mac
# or
set PYTHONPATH=%PYTHONPATH%;%cd%\src          # Windows
```

### CUDA/GPU Issues

If you encounter GPU-related errors:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Memory Issues

For systems with limited RAM/VRAM:

1. Use smaller Whisper model: `whisper.load_model("tiny")`
2. Reduce batch sizes in training scripts
3. Enable CPU offloading in transformers

### Port Already in Use

If port 8000 is already in use:

```bash
uvicorn src.sign_app.api:app --port 8001
```

## Next Steps

- Read the [README.md](../README.md) for usage examples
- Check [DEVELOPMENT.md](./DEVELOPMENT.md) for architecture details
- Review API documentation at `http://localhost:8000/docs`

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4GB | 8GB+ |
| VRAM (GPU) | 2GB | 6GB+ |
| Disk Space | 10GB | 30GB+ |
| Python | 3.12 | 3.12+ |

## Support

For issues or questions, please refer to the main README or create an issue in the repository.
