# Subtitle Mike: AI-Powered Transcribe & Contextual Translation

![Docker Pulls](https://img.shields.io/docker/pulls/mikedaspike/subtitle-mike?style=flat-square&color=blue)
![Docker Image Size](https://img.shields.io/docker/image-size/mikedaspike/subtitle-mike/latest?style=flat-square)
![GitHub License](https://img.shields.io/github/license/MikedaSpike/subtitle-mike?style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/MikedaSpike/subtitle-mike?style=flat-square)

Subtitle Mike is a professional‑grade, AI‑powered transcription tool that converts video and audio files into clean, readable SRT subtitles.  
It uses WhisperX for word‑level alignment and Pyannote for speaker identification, producing streaming‑ready results optimized for readability and timing accuracy.

The system offers a flexible translation engine, allowing users to optionally translate subtitles into multiple languages. You can choose between LibreTranslate for fast, reliable processing or leverage Ollama LLM (e.g., Mistral-Nemo) for high-quality, context-aware translations that capture nuances, idioms, and natural sentence flow better than standard engines.


​Designed for creators, archivists, and automation workflows, Subtitle Mike bridges the gap between raw AI transcription and polished, multi-language subtitle delivery.

---

## Hardware & Performance

This tool requires an NVIDIA GPU.

* Verified Hardware: NVIDIA GeForce RTX 5080 (16GB VRAM)  
* Uses CUDA 12.x and FP16 compute  
* Suppresses redundant library warnings for clean logs  

---

## Features

* WhisperX transcription with alignment  
* Optional Pyannote diarization  
* Energy‑based segment clamping  
* Genre‑based presets for VAD, ASR, and subtitle behavior  
* Subtitle formatting with CPL/CPS limits  
* Automatic dialogue dash insertion  
* Recursive folder processing  
* Skip logic for existing subtitles  
* Local model caching  
* Optional translation via LibreTranslate  
* JSON import/export for manual segment editing  
* Debug JSON output for every processing stage  
* Clean logging with CLI override detection  
* Model validation before loading  

---

# Installation & Setup

## Prerequisites

1. **Docker & Docker Compose**
2. **NVIDIA Container Toolkit** installed on your host. This is required to bridge your physical GPU to the Docker container.
3. **Hugging Face Account & Model Access:** This tool uses Pyannote for speaker diarization. You must accept the user conditions for the following models on Hugging Face (while logged in):
* Accept terms for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
* Accept terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
4. **Hugging Face Token:** Create a Read-access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

## Environment Setup

```bash
mkdir -p models/huggingface models/whisper models/torch
echo "HF_TOKEN=your_huggingface_token_here" > .env
```

---

## Docker Compose

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
    env_file:
      - .env
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - PYTHONWARNINGS=ignore
    volumes:
      - /path/to/your/media/library:/data
      - ./models/huggingface:/root/.cache/huggingface
      - ./models/whisper:/root/.cache/whisper
      - ./models/torch:/root/.cache/torch
```

---

# Optional: Automated Translation

Subtitle Mike supports automated translation using either **LibreTranslate** (fast, local) or an **LLM** (Ollama/Llama 3.1) for high-quality, context-aware results.

### Configuration: LibreTranslate
Add these to your environment variables in your `docker-compose.yml` or `.env` file to use the local translation engine:

```yaml
environment:
  - TRANSLATE_TO_LANG=nl
  - LT_LOAD_ONLY=en,nl

```

### Configuration: LLM (Ollama)

Use these settings for high-quality, context-aware translations via Ollama:

```yaml
environment:
  - TRANSLATE_TO_LANG=nl
  - LLM_URL=http://ollama-subtitle:11434
  - LLM_MODEL=llama3.1

```

### Deployment Examples

Choose the configuration that fits your workflow by using one of our example files:

* **[Standard / LibreTranslate](docker-compose.translate_libre.example.yml)** – Fast, local translation using dedicated models.
* **[High-Quality / LLM](docker-compose.translate_LLM.example.yml)** – Advanced, context-aware translation using Ollama.

### Translation CLI Arguments
You can switch between strategies using the `--translator` flag.

| Argument | Default | Description |
|----------|----------|-------------|
| `--translate` | False | Enables translation and generates a second SRT |
| `--translator_url` | http://translator:5000 | LibreTranslate endpoint |
| `--llm_url` | http://localhost:11434 | Ollama API endpoint (for `llm` strategy) |
| `--llm_prompt` | None | Specific instructions for the LLM (e.g. 'Medical drama context') |

---

# Usage Examples

### Process a single file

```bash
docker compose run --rm subtitle-mike python main.py --input "/data/MyVideo.mp4"
```

### Process a folder recursively

```bash
docker compose run --rm subtitle-mike python main.py --input "/data/Shows" --recursive
```

### Process and translate

```bash
docker compose run --rm subtitle-mike python main.py --input "/data/MyVideo.mp4" --translate
```

---

# Configuration & CLI Arguments

Below is the complete and updated list of CLI options supported by the current pipeline.

## Basic Usage

| Argument | Default | Description |
|----------|----------|-------------|
| `--input` | Required | File or directory to process |
| `--output_dir` | None | Custom output directory |
| `--recursive` | False | Scan subdirectories |
| `--overwrite` | False | Replace existing SRT files |
| `--genre` | sitcom | Selects a preset profile (see Genre Profiles) |
| `--model` | large‑v3 | Whisper model name (validated before loading) |
| `--lang` | en | Language code |
| `--initial_prompt` | "" | Optional prompt for Whisper |
| `--translate` | False | Generate translated SRT |
| `--debug` | False | Save debug JSON files |
| `--dump_json` | False | Export processed segments to JSON |
| `--from_json` | False | Load segments from JSON instead of audio |
| `--blacklist` | "" | Comma‑separated list of phrases to remove |

---

# Genre Profiles

Subtitle Mike includes a set of genre presets that automatically configure:

- VAD sensitivity  
- ASR beam size and no‑speech threshold  
- Subtitle timing rules  
- CPS/CPL limits  
- Duration constraints  
- Gap merging behavior  
- Semantic break preferences  

Available genres:

```
sitcom
drama
netflix
film
talkshow
podcast
music
documentary
youtube
asmr
action
```

Each genre defines:

### 1. Profile  
Determines base subtitle rules (CPL, CPS, durations).  
Profiles:  
- BroadcastProfile  
- NetflixProfile  

### 2. VAD Settings  
Includes:  
- threshold  
- min_silence_duration_ms  
- speech_pad_ms  
- vad_onset  
- vad_offset  

### 3. ASR Settings  
Includes:  
- beam_size  
- no_speech_threshold  

### 4. Subtitle Overrides  
May include:  
- max_gap  
- min_duration  
- max_duration  
- prefer_semantic_breaks  
- allow_sentence_merge  
- max_cps  

### Important: CLI overrides always take priority  
Any genre‑defined value can be overridden using:

```
--beam
--no_speech
--vad_onset
--vad_offset
--vad_threshold
--vad_min_silence_ms
--vad_speech_pad_ms
--max_cpl
--max_lines
--min_segment_duration
--max_segment_duration
--min_cps
--max_cps
--max_gap
```

---

# Subtitle Parameters (Overrides)

| Flag | Description |
|------|-------------|
| `--max_cpl` | Max characters per line |
| `--max_lines` | Max lines per subtitle block |
| `--min_segment_duration` | Minimum segment duration |
| `--max_segment_duration` | Maximum segment duration |
| `--min_cps` | Minimum characters per second |
| `--max_cps` | Maximum characters per second |
| `--max_gap` | Merge gap threshold |

---

# ASR Parameters (Overrides)

| Flag | Description |
|------|-------------|
| `--beam` | Beam size |
| `--no_speech` | No‑speech threshold |

---

# VAD Parameters (Overrides)

| Flag | Description |
|------|-------------|
| `--vad_onset` | Speech onset sensitivity |
| `--vad_offset` | Speech offset sensitivity |
| `--vad_threshold` | VAD threshold |
| `--vad_min_silence_ms` | Minimum silence duration |
| `--vad_speech_pad_ms` | Padding around speech |

---

# JSON Import/Export

Subtitle Mike supports exporting and re‑using transcription data through two options:

### `--dump_json`
Exports all processed segments (timestamps, text, speakers, word‑timing, clamped segments, etc.) to:

```
<file>.<lang>.segments.json
```

### `--from_json`
Loads the previously exported JSON instead of processing audio.  
Useful when:

- subtitles need manual correction  
- you want to test different subtitle settings  
- you want to regenerate subtitles without re‑transcribing  
- you want to run translation without GPU load  

Workflow:

1. Run once with `--dump_json`  
2. Edit the JSON if needed  
3. Rebuild subtitles with `--from_json`  

---

# Supported Extensions

* Video: `.mp4`, `.mkv`, `.avi`, `.mov`  
* Audio: `.wav`, `.mp3`  

---

# Support & Contributions

This project is maintained for personal use and shared as‑is.  
Issues are disabled, but pull requests are welcome.

Steps:

1. Fork  
2. Implement  
3. Submit PR  

---

# License

[MIT](https://choosealicense.com/licenses/mit/)

---

