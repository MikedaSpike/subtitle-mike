import os
import sys
import json
import time
import copy
import argparse
import logging
import warnings
import librosa

from pathlib import Path
from contextlib import contextmanager

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

from pipeline.subtitle import SubtitleEngine
from pipeline.translator import SubtitleTranslator
from pipeline.genres import GENRE_PROFILES
from pipeline.energy_clamp import clamp_segments_energy
from utils.logging import setup_logger

# ---------------------------------------------------------------------------
# ENV / TORCH SAFETY PATCH (PyTorch ≥2.6)
# ---------------------------------------------------------------------------

_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

try:
    import omegaconf
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    torch.serialization.add_safe_globals([
        ListConfig,
        DictConfig,
        omegaconf.base.ContainerMetadata,
        omegaconf.base.Node,
    ])
except Exception:
    pass

# ---------------------------------------------------------------------------
# LOGGING / WARNINGS
# ---------------------------------------------------------------------------

warnings.filterwarnings("ignore", message="Model was trained with")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=FutureWarning)

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)

logger = setup_logger()

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = {".mp4", ".mkv", ".avi", ".wav", ".mp3", ".mov"}

SUBTITLE_PARAM_KEYS = [
    "max_cpl",
    "max_lines",
    "min_duration",
    "max_duration",
    "min_cps",
    "max_cps",
    "max_gap",
]

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_out
            sys.stderr = old_err


def collect_files(input_path, recursive=False):
    p = Path(input_path)
    if p.is_file():
        return [p]

    files = []
    pattern = "**/*" if recursive else "*"
    for f in p.glob(pattern):
        if f.suffix.lower() in SUPPORTED_EXTS:
            files.append(f)
    return sorted(files)


def override(cli_value, preset_value):
    return cli_value if cli_value is not None else preset_value


def log_param_block(title, params):
    logger.info(title)
    for k in SUBTITLE_PARAM_KEYS:
        if k in params:
            logger.info(f"  - {k:15}: {params[k]}")

def log_transcribe_params(params):
    logger.info("Effective transcription parameters:")
    for k, v in params.items():
        logger.info(f"  - {k:20}: {v}")
        
def log_cli_overrides(args):
    logger.info("CLI overrides:")
    used = False

    argv = set(sys.argv)

    for k, v in vars(args).items():
        flag = f"--{k}"

        if flag in argv:
            logger.info(f"  - {k}: {v}")
            used = True

    if not used:
        logger.info("  (none)")
        

def save_debug_json(base_path: Path, data, tag):
    debug_path = base_path.with_suffix(f".debug_{tag}.json")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"      debug -> {debug_path.name}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    start_time = time.time()
    
    hf_token = os.getenv("HF_TOKEN")
    
    logger.info("-" * 40)
    logger.info("Auto Subtitle Pipeline - START")
    logger.info("-" * 40)

    # ---------------------------------------------------------------------
    # CLI
    # ---------------------------------------------------------------------

    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--recursive", action="store_true")

    ap.add_argument("--genre", choices=GENRE_PROFILES.keys(), default="sitcom")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--initial_prompt", default="")
    ap.add_argument("--translate", action="store_true")
    ap.add_argument("--debug", action="store_true")
    
    ap.add_argument("--dump_json", action="store_true")
    ap.add_argument("--from_json", action="store_true")
    ap.add_argument("--blacklist", default="")
    ap.add_argument("--beam", type=int)
    ap.add_argument("--no_speech", type=float)


    # Subtitle overrides
    ap.add_argument("--max_cpl", type=int)
    ap.add_argument("--max_lines", type=int)
    ap.add_argument("--min_duration", type=float)
    ap.add_argument("--max_duration", type=float)
    ap.add_argument("--min_cps", type=float)
    ap.add_argument("--max_cps", type=float)
    ap.add_argument("--max_gap", type=float)

    # VAD overrides
    ap.add_argument("--vad_onset", type=float)
    ap.add_argument("--vad_offset", type=float)
    ap.add_argument("--vad_threshold", type=float)
    ap.add_argument("--vad_min_silence_ms", type=int)
    ap.add_argument("--vad_speech_pad_ms", type=int)

    args = ap.parse_args()

    if args.dump_json and args.from_json:
        raise ValueError("--dump_json and --from_json cannot be combined")

    # ------------------------------------------------------------
    # GENRE / PROFILE RESOLUTION
    # ------------------------------------------------------------

    genre_cfg = GENRE_PROFILES[args.genre]
    profile = genre_cfg["profile"]()

    vad = genre_cfg["vad"].copy()
    asr = genre_cfg["asr"].copy()
    
    # Apply CLI overrides (ASR)
    if args.beam is not None:
        asr["beam_size"] = args.beam
    
    if args.no_speech is not None:
        asr["no_speech_threshold"] = args.no_speech

    # Apply CLI overrides (VAD)
    if args.vad_onset is not None:
       vad["vad_onset"] = args.vad_onset
    if args.vad_offset is not None:
       vad["vad_offset"] = args.vad_offset
    if args.vad_threshold is not None:
       vad["threshold"] = args.vad_threshold
    if args.vad_min_silence_ms is not None:
       vad["min_silence_duration_ms"] = args.vad_min_silence_ms
    if args.vad_speech_pad_ms is not None:
       vad["speech_pad_ms"] = args.vad_speech_pad_ms

    # Initiliaze subtitle paramaters
    subtitle_params = {
        "max_cpl": profile.max_cpl,
        "max_lines": profile.max_lines,
        "min_duration": profile.min_duration,
        "max_duration": profile.max_duration,
        "min_cps": profile.min_cps,
        "max_cps": profile.max_cps,
        "max_gap": profile.max_gap,
    }

    #Set defaults
    if "subtitle" in genre_cfg:
        for key, value in genre_cfg["subtitle"].items():
            subtitle_params[key] = value

    # Apply Subtitles overrides 
    if args.max_cpl is not None:
        subtitle_params["max_cpl"] = args.max_cpl
    if args.max_lines is not None:
        subtitle_params["max_lines"] = args.max_lines 
    if args.min_duration is not None:
        subtitle_params["min_duration"] = args.min_duration
    if args.max_duration is not None:
        subtitle_params["max_duration"] = args.max_duration
    if args.min_cps is not None:
        subtitle_params["min_cps"] = args.min_cps
    if args.max_cps is not None:
        subtitle_params["max_cps"] = args.max_cps
    if args.max_gap is not None:
        subtitle_params["max_gap"] = args.max_gap
    
    # ------------------------------------------------------------
    # BUILD TRANSCRIBE PARAMS
    # ------------------------------------------------------------

    transcribe_params = {
        "model": args.model,
        "language": args.lang,
        "initial_prompt": args.initial_prompt,
        **asr,
    }
    
    alignment_params = {
        **vad
    }

    # ------------------------------------------------------------
    # LOG EFFECTIVE CONFIG
    # ------------------------------------------------------------

    logger.info(f"Genre            : {args.genre}")
    logger.info(f"Subtitle profile : {profile.name}")
    
    log_cli_overrides(args)
    
    if not args.from_json:
        logger.info("Effective transcribe parameters:")
        for k, v in transcribe_params.items():
            logger.info(f"  - {k:25}: {v}")

        logger.info("Effective alignment parameters:")
        for k, v in alignment_params.items():
            logger.info(f"  - {k:25}: {v}")

        logger.info("Effective diarization parameters:")
        if hf_token:
            logger.info("  - enabled                 : yes")
            logger.info("  - model                   : pyannote/speaker-diarization")
            logger.info("  - device                  : cuda")
        else:
            logger.info("  - enabled                 : no (HF_TOKEN not set)")



    logger.info("Effective subtitle parameters:")
    for k, v in subtitle_params.items():
       logger.info(f"  - {k:25}: {v}")


    # ---------------------------------------------------------------------
    # FILES
    # ---------------------------------------------------------------------

    files = collect_files(args.input, args.recursive)
    logger.info(f"Files to process: {len(files)}")

    blacklist = [b.strip() for b in args.blacklist.split(",") if b.strip()]
    logger.info(f"Blacklist terms  : {blacklist if blacklist else '(none)'}")

    # ---------------------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------------------
    
    if not args.from_json:
        from faster_whisper.utils import available_models

        # Dynamisch beschikbare modellen ophalen
        VALID_MODELS = set(available_models())

        model = None

        # Modelnaam normaliseren
        model_name = args.model.strip()

        # Eerst valideren, zodat load_model nooit een fout kan gooien
        if model_name not in VALID_MODELS:
            logger.error(
                f"Invalid model '{model_name}'. Must be one of: {', '.join(sorted(VALID_MODELS))}"
            )
            sys.exit(1)

        model = None
        if not args.from_json:
            logger.info(f"Loading Whisper model: {args.model}")
            with suppress_output():
                model = whisperx.load_model(
                    args.model,
                    "cuda",
                    compute_type="float16",
                    language=args.lang,
                    vad_options=vad,
                    asr_options={
                        **asr,
                        "initial_prompt": args.initial_prompt,
                    },
                )

        
    # ---------------------------------------------------------------------
    # PROCESS LOOP
    # ---------------------------------------------------------------------

    processed = translated = skipped = errors = 0

    for idx, media in enumerate(files, 1):
        try:
            logger.info(f"[{idx}/{len(files)}] Start of processing {media.name}")

            if args.output_dir:
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                srt_path = out_dir / f"{media.stem}.{args.lang}.srt"
            else:
                srt_path = media.with_suffix(f".{args.lang}.srt")

            if srt_path.exists() and not args.overwrite:
                logger.info("  -> Skipped (exists)")
                skipped += 1
                continue

            # ---------------------------------------------------------
            # LOAD / TRANSCRIBE
            # ---------------------------------------------------------

            if args.from_json:
                json_path = srt_path.with_suffix(".segments.json")
                logger.info("  -> Loading segments from JSON")
                with open(json_path, "r", encoding="utf-8") as f:
                    result = json.load(f)

            else:
                logger.info(f"  -> Loading audio into memory: {media.name}")
                audio = whisperx.load_audio(str(media))
                logger.info("  -> Starting Transcribing")
                with suppress_output():
                    result = model.transcribe(audio, batch_size=16)
                logger.info("  -> Finished Transcribing")
                if args.debug:
                    save_debug_json(srt_path, result, "raw")

                logger.info("  -> Starting Aligning")
                with suppress_output():
                    model_a, meta = whisperx.load_align_model(
                        result.get("language", args.lang),
                        "cuda",
                    )
                    result = whisperx.align(
                        result["segments"],
                        model_a,
                        meta,
                        audio,
                        "cuda",
                        return_char_alignments=False
                    )
                logger.info("  -> Finished Aligning")
                if args.debug:
                    save_debug_json(srt_path, result, "aligned")

                if hf_token:
                    logger.info("  -> Starting Diarization")
                    diar = DiarizationPipeline(
                        use_auth_token=hf_token,
                        device="cuda",
                    )
                    result = whisperx.assign_word_speakers(
                        diar(audio),
                        result,
                    )
                    logger.info("  -> Finished Diarization")

                if args.debug:
                    save_debug_json(srt_path, result, "diarized")
                
                logger.info("  -> Starting energy-based segment clamping")
                audio, sr = librosa.load(str(media), sr=None)

                result["segments"] = clamp_segments_energy(
                    result["segments"],
                    audio=audio,
                    sr=sr,
                    energy_threshold=0.02,  
                )
                logger.info("  -> Finished energy-based segment clamping")
                
                if args.debug:
                    save_debug_json(srt_path, result, "clamped")

                if args.dump_json:
                    json_path = srt_path.with_suffix(".segments.json")
                    with open(
                       json_path,
                        "w",
                        encoding="utf-8",
                    ) as f:
                        json.dump(
                            {
                                "language": result.get("language", args.lang),
                                "segments": result["segments"],
                            },
                            f,
                            indent=2,
                            ensure_ascii=False,
                        )
                    logger.info(f"  -> Segments JSON saved: {json_path.name}")
                    
            # ---------------------------------------------------------
            # SUBTITLES
            # ---------------------------------------------------------

            logger.info("  -> Generating subtitles")
            segments_copy = copy.deepcopy(result["segments"])
            engine = SubtitleEngine(
                segments=segments_copy,
                max_cpl=subtitle_params["max_cpl"],
                max_lines=subtitle_params["max_lines"],
                min_duration=subtitle_params["min_duration"],
                max_duration=subtitle_params["max_duration"],
                min_cps=subtitle_params["min_cps"],
                max_cps=subtitle_params["max_cps"],
                max_gap=subtitle_params["max_gap"],
            )

            logger.info("  -> Starting Software Timing Alignment (CPS based)")
            aligned_list = engine.align_timings_heuristically()
            
            engine.segments = aligned_list
            result["segments"] = aligned_list

            if args.debug:
                save_debug_json(srt_path, {"segments": aligned_list}, "soft_aligned")

            engine.process_and_save(srt_path, blacklist)
            logger.info(f"  -> Subtitles saved: {srt_path.name}")
            if args.debug:
                save_debug_json(
                    srt_path,
                    {"segments": segments_copy},
                    "subtitles",
                )

            # ---------------------------------------------------------
            # TRANSLATION
            # ---------------------------------------------------------

            if args.translate:
                logger.info("  -> Translating subtitles")

                try:
                    translator = SubtitleTranslator()
                    translated_segments = translator.translate_segments(
                        copy.deepcopy(result["segments"])
                    )
                    
                    if translated_segments is None:
                        logger.warning("  -> Translation failed, skipping writing translated SRT")
                    else:
                        translated_path = srt_path.with_name(
                            f"{media.stem}.{translator.target_lang}.srt"
                        )

                        SubtitleEngine(
                            segments=translated_segments,
                            max_cpl=subtitle_params["max_cpl"],
                            max_lines=subtitle_params["max_lines"],
                            min_duration=subtitle_params["min_duration"],
                            max_duration=subtitle_params["max_duration"],
                            min_cps=subtitle_params["min_cps"],
                            max_cps=subtitle_params["max_cps"],
                            max_gap=subtitle_params["max_gap"],
                        ).process_and_save(translated_path, blacklist)
                        logger.info(f"  -> Translated subtitles saved: {translated_path.name}")
                        translated += 1

                except Exception as e:
                    logger.error(f"  -> Translation failed: {e}")

            processed += 1
            logger.info(f"  -> Finished processing {media.name}")

        except Exception as e:
            logger.exception(f"  -> ERROR: {e}")
            errors += 1

    # ---------------------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------------------

    elapsed = time.time() - start_time
    logger.info("-" * 40)
    logger.info("FINISH")
    logger.info(f"Processed : {processed}")
    logger.info(f"Skipped   : {skipped}")
    logger.info(f"Translated: {translated}")
    logger.info(f"Errors    : {errors}")
    logger.info(f"Time      : {int(elapsed // 60)}m {int(elapsed % 60)}s")
    logger.info("-" * 40)


if __name__ == "__main__":
    main()
