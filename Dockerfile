# ============================================================
# Gemini Editor — Production Dockerfile
# ============================================================
# Multi-stage build to keep final image lean.
# Stage 1: Install Python dependencies with full build tools.
# Stage 2: Runtime image — copies only the installed site-packages
#           and the repo code; no compiler toolchain in prod.
# ============================================================

# ── Stage 1: Builder ────────────────────────────────────────
FROM python:3.11-slim AS builder

# Build-time system deps (compilers, libs required by some pip packages)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    libsndfile1-dev \
    libopus-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    poppler-utils \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python deps into a prefix so Stage 2 can copy them cleanly
COPY requirements.txt .
RUN pip install --upgrade pip wheel && \
    pip install --no-cache-dir --prefix=/install \
    # Exclude Windows-only packages — they will fail on Linux
    $(grep -v "sys_platform == 'win32'" requirements.txt | grep -v "^#" | grep -v "^$")


# ── Stage 2: Runtime ────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system deps only (no compilers)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    # FFmpeg with drawtext + libharfbuzz (full GPL build via BtbN)
    xz-utils \
    poppler-utils \
    libsndfile1 \
    libopus0 \
    libgomp1 \
    && curl -sL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz \
       | tar -xJ -C /tmp \
    && mv /tmp/ffmpeg-*-linux64-gpl/bin/ffmpeg /usr/local/bin/ \
    && mv /tmp/ffmpeg-*-linux64-gpl/bin/ffprobe /usr/local/bin/ \
    && rm -rf /tmp/ffmpeg-*-linux64-gpl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy project source (excluding secrets & local dev files via .dockerignore)
COPY . .

# Create writable directories that the batch process needs
# data/ is mounted read-only at runtime; downloads/ is user-scoped volume
RUN mkdir -p /app/downloads /app/cache /app/logs /app/temp /app/output

# Runtime env — override via docker run -e
ENV PYTHONUNBUFFERED=1 \
    BATCH_ONLY_MODE=1 \
    TZ=Asia/Kolkata \
    REPO_ROOT=/app

# Default entrypoint: run the batch for a specific user.
# Pass --batch-user <chat_id> to target a user; omit for admin/system pool.
ENTRYPOINT ["python", "-u", "main.py"]
CMD ["--batch"]
