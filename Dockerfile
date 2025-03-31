FROM python:3.10-slim

# Install curl and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    g++ \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system appgroup && \
    useradd --system --no-create-home --gid appgroup appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SIYUAN_URL="http://siyuan:6806"
ENV QDRANT_LOCATION="qdrant:6333"
ENV QDRANT_COLLECTION_NAME="siyuan_ai_companion"
ENV OPENAI_URL="https://api.openai.com/v1/"

# Set the working directory and change ownership to the non-root user
WORKDIR /app
RUN chown appuser:appgroup /app

# Copy source code and configuration files, then change ownership
COPY . /app/siyuan-ai-companion

RUN python -m pip install --no-cache-dir \
  torch --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --no-cache-dir /app/siyuan-ai-companion[hypercorn] \
    && chown -R appuser:appgroup /app

# Switch to the non-root user
USER appuser

# Expose port
EXPOSE 8000

# Run the application
CMD ["hypercorn", "siyuan_ai_companion.asgi:application"]

# Healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1
