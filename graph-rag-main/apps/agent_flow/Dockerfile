# Multi-stage build for optimization
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY apps/agent_flow/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt && \
    pip install --user --no-cache-dir openai==1.43.0

# Download NLTK data
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local
COPY --from=builder /usr/local/share/nltk_data /usr/local/share/nltk_data

# Copy application code
COPY libs/ ./libs/
COPY apps/agent_flow/ ./apps/agent_flow/

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app:$PYTHONPATH \
    WORKERS=4 \
    AGENT_FLOW_PORT=8096

# Make entrypoint executable and change ownership
RUN chmod +x apps/agent_flow/entrypoint.sh && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Use entrypoint script
ENTRYPOINT ["apps/agent_flow/entrypoint.sh"]