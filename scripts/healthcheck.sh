#!/bin/sh

PORT="${HYPERCORN_PORT:-8000}"

curl -f "http://localhost:$PORT/health" || exit 1
