FROM python:3.11-slim

WORKDIR /app

# Install system deps for Playwright and lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Install Playwright Chromium
RUN playwright install --with-deps chromium

# Copy app code
COPY . .

# Create data dir (volume mount point)
RUN mkdir -p /data

ENV PUNTY_DB_PATH=/data/punty.db
ENV PUNTY_DEBUG=false

EXPOSE 8000

CMD ["uvicorn", "punty.main:app", "--host", "0.0.0.0", "--port", "8000"]
