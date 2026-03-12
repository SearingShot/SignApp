# ── Build Stage ─────────────────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

# Install system deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY .env .env
COPY uploads/ uploads/

# Copy model files (if present)
COPY speechCleaner_t5_model/ speechCleaner_t5_model/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run
CMD ["uvicorn", "src.sign_app.api:app", "--host", "0.0.0.0", "--port", "8000"]
