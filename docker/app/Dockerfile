# Detection service image: the pipeline plus the web API.
# Build from the repo root: docker build -f docker/app/Dockerfile .
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models/hf

WORKDIR /app

# Dependencies first so this layer is cached across code changes.
# The default PyPI torch wheel bundles the CUDA runtime.
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm \
 && python -m spacy download uk_core_news_sm \
 && python -m spacy download ru_core_news_sm \
 && python -m spacy download zh_core_web_sm

COPY . .

# Non-root user; volume mount points are created with its ownership so new
# named volumes inherit it.
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app/cache /app/logs /app/var /models/hf \
 && chown -R app:app /app/cache /app/logs /app/var /models

USER app

EXPOSE 8000
CMD ["python", "-m", "api"]
