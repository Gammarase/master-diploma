#!/bin/sh
# Start the Ollama server and pull OLLAMA_MODEL on first start.
# Models live in a volume, so later starts skip the download.
set -eu

: "${OLLAMA_MODEL:?OLLAMA_MODEL must be set}"

ollama serve &
pid=$!
trap 'kill -TERM "$pid" 2>/dev/null; wait "$pid"; exit 0' TERM INT

until ollama list >/dev/null 2>&1; do sleep 1; done

if ! ollama show "$OLLAMA_MODEL" >/dev/null 2>&1; then
  echo "Pulling model $OLLAMA_MODEL ..."
  ollama pull "$OLLAMA_MODEL"
fi

wait "$pid"
