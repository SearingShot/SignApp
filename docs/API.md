# API Reference

## Base URL
```
http://localhost:8000
```

## Endpoints

### Voice to Text Conversion

Convert audio file to text with automatic language detection.

**Endpoint:** `POST /voice-to-text/`

**Request:**
- **Content-Type:** `multipart/form-data`
- **Parameters:**
  - `file` (File, required): Audio file (mp3, wav, m4a, etc.)

**Response:** `200 OK`
```json
{
  "language": "en",
  "transcription": "string with the converted text from audio"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/voice-to-text/" \
  -F "file=@path/to/audio.mp3"
```

**Python Example:**
```python
import requests

with open('audio.mp3', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/voice-to-text/', files=files)
    result = response.json()
    print(f"Language: {result['language']}")
    print(f"Transcription: {result['transcription']}")
```

### Supported Audio Formats
- MP3
- WAV
- M4A
- FLAC
- OGG
- OPUS
- AAC

## Error Responses

### 400 Bad Request
```json
{
  "detail": "No file provided or invalid file format"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Error processing audio file"
}
```

## Rate Limiting

Currently no rate limiting is enforced. For production deployments, implement:
- Request throttling
- Authentication
- User quotas

## OpenAPI/Swagger Documentation

Interactive API documentation available at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## Future Endpoints (In Development)

- `POST /remove-disfluency/` - Remove filler words from text
- `GET /supported-languages/` - List supported languages
- `POST /batch-transcribe/` - Process multiple files

---

For integration examples and more details, see the main README.
