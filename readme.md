# Subtitle Mike (Transcribe to SRT)

**Subtitle Mike** is a professional-grade, AI-powered transcription tool that converts video and audio files into perfectly formatted SRT subtitles. By leveraging **WhisperX** for word-level alignment and **Pyannote** for speaker identification, it delivers "streaming-ready" results optimized for readability.

## Hardware & Performance

This tool is built for speed and requires an **NVIDIA GPU**.

* **Verified Hardware:** Optimized and tested on the **NVIDIA GeForce RTX 5080 (16GB VRAM)**.
* **Technology:** Leverages CUDA 12.x and FP16 compute to process long-form content in minutes.
* **Quiet Execution:** Custom filters suppress redundant library warnings (Pyannote/Torch) for clean, actionable logs.

## Features

* **Smart Formatting:**
* **Balanced Lines:** Splits subtitles at natural pauses, avoiding "orphan words."
* **Dialogue Handling:** Automatically adds dashes (`-`) when speaker changes are detected within a block.
* **Streaming Standards:** Adheres to the 42 characters-per-line (CPL) standard (Netflix/BBC style).


* **Batch Processing:** Handles single files or entire nested folder structures recursively.
* **Efficient Skip Logic:** Automatically skips files that already have an existing subtitle to save time.
* **Model Caching:** Supports local caching of AI models to prevent re-downloading on every run.

---

## Installation & Setup

### Prerequisites

1. **Docker & Docker Compose**
2. **NVIDIA Container Toolkit** installed on your host. This is required to bridge your physical GPU to the Docker container.
3. **Hugging Face Account & Model Access:** This tool uses Pyannote for speaker diarization. You must accept the user conditions for the following models on Hugging Face (while logged in):
* Accept terms for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
* Accept terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)


4. **Hugging Face Token:** Create a Read-access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 1. Prepare Environment

Create a local folder structure for caching AI models and prepare your `.env` file:

```bash
# Create cache directories
mkdir -p models/huggingface models/whisper models/torch

# Create .env file and add your token
echo "HF_TOKEN=your_huggingface_token_here" > .env

```

### 2. Docker Compose (Recommended)

Using Docker Compose is the easiest way to manage volumes and environment variables.

**Create a `docker-compose.yml` file:**

```yaml
services:
  subtitle-mike:
    image: mikedaspike/subtitle-mike:latest
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - NVIDIA_VISIBLE_DEVICES=all
      - PYTHONWARNINGS=ignore
    volumes:
      - /path/to/your/media/library:/data
      # Cache AI models locally
      - ./models/huggingface:/root/.cache/huggingface
      - ./models/whisper:/root/.cache/whisper
      - ./models/torch:/root/.cache/torch

```

### 3. Usage Examples

**Process a single file:**

```bash
docker compose run --rm subtitle-mike python main.py --input "/data/MyVideo.mp4"

```

**Process an entire folder recursively:**

```bash
docker compose run --rm subtitle-mike python main.py --input "/data/TV_Shows" --recursive

```

---

## Configuration & CLI Arguments

### Basic Usage

The only mandatory argument is `--input`.

| Argument | Type | Default | Status | Description |
| --- | --- | --- | --- | --- |
| `--input` | Path | - | **Required** | Path to the file or directory inside the container. |
| `--output_dir` | Path | `None` | Optional | Where to save SRTs. Defaults to the input folder. |
| `--recursive` | Flag | `False` | Optional | Scan all subdirectories for media files. |
| `--overwrite` | Flag | `False` | Optional | Re-process and replace existing `.srt` files. |
| `--lang` | String | `en` | Optional | Audio language code (en, nl, de, fr, etc.). |
| `--model` | String | `large-v3` | Optional | Whisper model size (see Model section). |

### Advanced Tuning

| Argument | Default | Description |
| --- | --- | --- |
| `--initial_prompt` | `""` | Provide context or specific names to improve AI accuracy. |
| `--blacklist` | `¶¶,...` | Comma-separated list of phrases to automatically remove. |
| `--beam` | `1` | Beam size. Higher is more precise but significantly slower. |
| `--onset` | `0.02` | VAD sensitivity. Adjust if speech detection is too aggressive. |
| `--max_cpl` | `42` | Maximum characters per line (Netflix/BBC standard). |
| `--min_duration` | `1.0` | Minimum display time for a subtitle block (seconds). |
| `--max_gap` | `0.5` | Merge threshold: joins segments if the gap is smaller than this. |
| `--debug` | `False` | Saves a `.debug.json` with raw timestamps and speaker scores. |

---

## Understanding Whisper Models

Subtitle Mike supports all OpenAI Whisper model sizes. For **NVIDIA RTX 5080** users, `large-v3` is the highly recommended default.

| Model | Parameters | VRAM Required | Accuracy | Recommendation |
| --- | --- | --- | --- | --- |
| **tiny** | 39 M | ~1 GB | Lowest | Only for very low-end hardware. |
| **base** | 74 M | ~1 GB | Low | Fast, but struggles with accents. |
| **small** | 244 M | ~2 GB | Moderate | Good balance for older GPUs. |
| **medium** | 769 M | ~5 GB | High | Great results, slightly faster than large. |
| **large-v3** | 1550 M | ~10 GB | **Highest** | **Default.** Best for RTX 30/40/50-series. |

---

## Supported Extensions

* **Video:** `.mp4`, `.mkv`, `.avi`, `.mov`
* **Audio:** `.wav`, `.mp3`

## License

[MIT](https://choosealicense.com/licenses/mit/)

---
