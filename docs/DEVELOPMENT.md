# Development Guide

## Architecture Overview

### Module Organization

```
sign_app/
├── api.py           # FastAPI routes and request handling
├── audio/           # Audio processing with Whisper
├── disfluency/      # Disfluency removal models
│   ├── inference.py # Run inference on text
│   └── training.py  # Train/fine-tune models
└── ui/              # Frontend interfaces (future)
```

## Module Descriptions

### API Module (`api.py`)
- FastAPI application setup
- Endpoint: `/voice-to-text/` - Convert audio to text
- Handles file uploads and model inference
- Returns transcription with language detection

### Audio Module
- Whisper model initialization and usage
- Audio loading and preprocessing
- Multi-language transcription support

### Disfluency Module

#### Inference (`inference.py`)
- Load pre-trained disfluency removal model
- Remove filler words and speech artifacts
- Support for batch processing

#### Training (`training.py`)
- Dataset loading from HuggingFace
- Model fine-tuning pipeline
- Evaluation metrics (BLEU, ROUGE)
- MLflow experiment tracking
- Model checkpointing and saving

## Development Workflow

### Adding New Features

1. Create appropriate module file in `src/sign_app/`
2. Import in corresponding `__init__.py`
3. Add unit tests in `tests/`
4. Update documentation

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_audio.py -v

# Run with coverage
pytest tests/ --cov=src/sign_app
```

### Code Standards

- Python 3.12+
- Follow PEP 8 style guidelines
- Type hints for all functions
- Docstrings for modules and functions

## Deployment Considerations

- Model persistence and versioning
- GPU/CPU resource allocation
- API rate limiting
- Error handling and logging
- Security: CORS, authentication (future)

## Troubleshooting

### Common Issues

1. **Model not found errors**: Check model paths and cache directories
2. **CUDA out of memory**: Reduce batch sizes or use smaller models
3. **API not starting**: Verify all dependencies are installed

### Debugging

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- [ ] User authentication
- [ ] Model versioning system
- [ ] Web-based UI
- [ ] Real-time streaming support
- [ ] Advanced sign language recognition
- [ ] Multi-GPU training
