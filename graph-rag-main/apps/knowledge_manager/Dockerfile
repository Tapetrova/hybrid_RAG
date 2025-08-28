# Multi-stage build for optimization
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies including Rust
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

# Add Rust to PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy requirements and install Python dependencies
COPY apps/knowledge_manager/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt && \
    pip install --user --no-cache-dir \
    gino==1.0.0 \
    SQLAlchemy==1.3.0

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY libs/ ./libs/
COPY apps/knowledge_manager/ ./apps/knowledge_manager/

# Set environment variables
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONPATH=/app:$PYTHONPATH

# Change ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "apps.knowledge_manager.src.main:app", "--host", "0.0.0.0", "--port", "8098", "--workers", "1"]