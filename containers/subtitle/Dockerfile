FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Systeem tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Stap 1: Upgrade pip en installeer basis tools
RUN pip install --upgrade pip setuptools wheel

# Stap 2: Installeer PyTorch (CRUCIAAL: eerst dit doen voor CUDA 12.1)
RUN pip install --no-cache-dir torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Stap 3: Installeer WhisperX direct van de source
RUN pip install --no-cache-dir git+https://github.com/m-bain/whisperX.git

# Stap 4: Overige benodigdheden voor je pipeline
RUN pip install --no-cache-dir \
    transformers \
    sentencepiece \
    pysrt \
    deep-translator \
    requests

# Stap 5: Kopieer je code (voor productie-gebruik)
COPY ./app /app

CMD ["python", "main.py"]