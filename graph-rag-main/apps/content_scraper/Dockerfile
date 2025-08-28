# Use playwright base image with pre-installed browsers
FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

WORKDIR /app

# Install system dependencies (minimize layers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium-browser \
    libglib2.0-0 \
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libgtk-3-0 \
    fonts-liberation \
    libappindicator3-1 \
    xdg-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY apps/content_scraper/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    gino==1.0.0 \
    SQLAlchemy==1.3.0 \
    playwright

# Copy application code
COPY libs/ ./libs/
COPY apps/content_scraper/ ./apps/content_scraper/

# Set environment variables
ENV PYTHONPATH=/app:$PYTHONPATH

# pwuser already exists in the base image, just use it
USER pwuser

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "apps.content_scraper.app:app", "--host", "0.0.0.0", "--port", "8099", "--workers", "1"]