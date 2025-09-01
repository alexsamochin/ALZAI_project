# ---- Base Image ----
FROM python:3.11-slim

# ---- Environment ----
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ---- System Dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---- Python Dependencies ----
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- Copy Entire Application ----
COPY . /app

# ---- Entrypoint ----
# Default: run training
ENTRYPOINT ["python", "train.py"]
