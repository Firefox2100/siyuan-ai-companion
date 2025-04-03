#!/bin/sh

# Default bind address and port
HOST="${HYPERCORN_HOST:-0.0.0.0}"
PORT="${HYPERCORN_PORT:-8000}"

exec hypercorn siyuan_ai_companion.asgi:application --bind "$HOST:$PORT"
