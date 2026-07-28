# The cluster image: booking API, agent sessions (text), /metrics and the demo page.
# Speech extras (faster-whisper, Piper voices) are left out: they add over a gigabyte and CI has no
# audio to feed them. Audio turns need the local install (see README).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PARLEY_ASR_MODEL=none \
    PARLEY_TTS=none

WORKDIR /app

# deploy/requirements.txt is `uv export --frozen --no-dev --no-emit-project`, so the image installs
# exactly the versions in uv.lock.
COPY deploy/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

COPY api ./api
COPY speech ./speech
COPY dialogue ./dialogue
COPY web ./web

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin parley
USER 10001

EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
