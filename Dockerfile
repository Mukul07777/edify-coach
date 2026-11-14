FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY server.py .
ENV PYTHONUNBUFFERED=1
# Maritime injects $PORT and keeps /data across restarts, redeploys and sleep
CMD ["python", "server.py"]
