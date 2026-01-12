import os, sys, re, json, warnings, logging, argparse
from pathlib import Path
from pipeline.subtitle import Subtitle
from pipeline.translator import SubtitleTranslator
from contextlib import contextmanager

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# --- PYTORCH FIX ---
import torch, omegaconf
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

try:
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    torch.serialization.add_safe_globals([ListConfig, DictConfig, omegaconf.base.ContainerMetadata, omegaconf.base.Node])
except:
    pass

# --- LOGGING ---
import whisperx
from whisperx.diarize import DiarizationPipeline
from utils.logging import setup_logger

warnings.filterwarnings("ignore", message="Model was trained with")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=FutureWarning)

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR) # Vaak ook een bron van spam

logger = setup_logger()

import time

SUPPORTED_EXTS = {".mp4", ".mkv", ".avi", ".wav", ".mp3", ".mov"}

def collect_files(input_path, recursive=False):
    path = Path(input_path)
    if path.is_file():
        return [path]
    
    files = []
    pattern = "**/*" if recursive else "*"
    for p in path.glob(pattern):
        if p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    return sorted(files)

def main():
    start_time = time.time()
    target_lang = os.getenv("TRANSLATE_TO_LANG", "nl")
    
    # --- ARGUMENTS ---
    ap = argparse.ArgumentParser(description="Auto-Subtitle Generator with Batch Support")
    ap.add_argument("--input", required=True, help="Input file or directory")
    ap.add_argument("--output_dir", default=None, help="Optional output directory (default: same as input)")
    ap.add_argument("--recursive", action="store_true", help="Process directories recursively")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing subtitle files")
    
    ap.add_argument("--model", default="large-v3", help="Whisper model size")
    ap.add_argument("--lang", default="en", help="Language code")
    ap.add_argument("--initial_prompt", default="", help="Context prompt for Whisper")
    ap.add_argument("--blacklist", default="¶¶,thank you,be right back", help="Comma-separated phrases to remove")
    
    # Tuning params
    ap.add_argument("--beam", type=int, default=1, help="Beam size")
    ap.add_argument("--onset", type=float, default=0.02, help="VAD onset")
    ap.add_argument("--no_speech", type=float, default=0.1, help="No-speech threshold")
    
    # Subtitle params
    ap.add_argument("--max_cpl", type=int, default=42, help="Max characters per line")
    ap.add_argument("--min_duration", type=float, default=1.0, help="Min duration (sec)")
    ap.add_argument("--max_gap", type=float, default=0.5, help="Merge gap (sec)")
    
    # Translation params
    ap.add_argument("--translate", action="store_true", help="Enable translation to Dutch")
    ap.add_argument("--translator_url", default=os.getenv("LIBRETRANSLATE_URL", "http://translator:5000"), help="URL of LibreTranslate")

    ap.add_argument("--debug", action="store_true", help="Save debug JSON")
    
    args = ap.parse_args()

    # --- COLLECT FILES ---
    files_to_process = collect_files(args.input, args.recursive)
    if not files_to_process:
        logger.error(f"No supported media files found in: {args.input}")
        return

    logger.info(f"Found {len(files_to_process)} file(s) to process.")
    blacklist_list = [i.strip() for i in args.blacklist.split(",")]

    # --- LOAD WHISPER MODEL (ONCE) ---
    logger.info(f"Loading Whisper model '{args.model}'...")
    try:
        with suppress_output():
            model = whisperx.load_model(
                args.model, "cuda", compute_type="float16", language=args.lang,
                vad_options={"vad_onset": args.onset, "vad_offset": args.onset},
                asr_options={"beam_size": args.beam, "initial_prompt": args.initial_prompt, "no_speech_threshold": args.no_speech}
            )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    processed_count = 0
    translated_count = 0
    skipped_count = 0
    error_count = 0

    # --- PROCESS LOOP ---
    for idx, p in enumerate(files_to_process, 1):
        try:
            # Determine output path
            if args.output_dir:
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                srt_path = out_dir / f"{p.stem}.{args.lang}.srt"
            else:
                srt_path = p.with_name(f"{p.stem}.{args.lang}.srt")

            # Skip check
            if srt_path.exists() and not args.overwrite:
                logger.info(f"[{idx}/{len(files_to_process)}] Skipping {p.name} (SRT exists)")
                skipped_count += 1
                continue

            logger.info(f"[{idx}/{len(files_to_process)}] Processing: {p.name}")

            # 1. Transcribe
            audio = whisperx.load_audio(str(p))
            with suppress_output():
                result = model.transcribe(audio, batch_size=16)

            # 2. Align
            logger.info(f"   -> Aligning...")
            detected_lang = result.get("language", args.lang)
            with suppress_output():
                model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device="cuda")
                result = whisperx.align(result["segments"], model_a, metadata, audio, "cuda")

            # 3. Diarization
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                logger.info("   -> Diarization (Speaker ID)...")
                diarize_model = DiarizationPipeline(use_auth_token=hf_token, device="cuda")
                result = whisperx.assign_word_speakers(diarize_model(audio), result)
            
            # 4. Generate SRT
            logger.info("   -> Generating SRT...")
            processor = Subtitle(
                result['segments'], 
                max_cpl=args.max_cpl, 
                min_duration=args.min_duration,
                max_gap=args.max_gap
            )
            processor.process_and_save(srt_path, blacklist=blacklist_list)

            # 5. OPTIONAL: Translate and Generate SRT
            if args.translate:
                translator = SubtitleTranslator() 
                
                logger.info(f"   -> Translating to {translator.target_lang}...")
                
                import copy
                translated_segments = copy.deepcopy(result['segments'])
                
                translated_segments = translator.translate_segments(translated_segments)
                
                # Bepaal pad voor vertaalde SRT (gebruik translator.target_lang i.p.v. args.translate_lang)
                nl_srt_path = srt_path.with_name(f"{p.stem}.{translator.target_lang}.srt")
                
                logger.info(f"   -> Generating Translated SRT ({translator.target_lang})...")
                nl_processor = Subtitle(
                    translated_segments,
                    max_cpl=args.max_cpl,
                    min_duration=args.min_duration,
                    max_gap=args.max_gap
                )
                nl_processor.process_and_save(nl_srt_path, blacklist=blacklist_list)
                translated_count += 1
                logger.info(f"   -> Saved translated SRT ({translator.target_lang})")

            if args.debug:
                debug_path = srt_path.with_suffix(".debug.json")
                with open(debug_path, "w") as jf: json.dump(result, jf, indent=2)
                logger.info(f"   -> Debug JSON saved")

            logger.info(f"   -> DONE: {srt_path.name}")
            processed_count += 1
        except Exception as e:
            logger.error(f"Error processing {p.name}: {str(e)}")
            error_count += 1
            if args.debug:
                import traceback
                traceback.print_exc()

    if files_to_process:
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        logger.info("-" * 30)
        logger.info("FINISH REPORT")
        logger.info(f"Total files found: {len(files_to_process)}")
        if args.translate:
            logger.info(f"Translated  ({target_lang}): {translated_count}")
        logger.info(f"Successfully processed: {processed_count}")
        logger.info(f"Skipped (already exists): {skipped_count}")
        
        if error_count > 0:
            logger.error(f"Errors encountered: {error_count}")
        else:
            logger.info(f"Errors encountered: 0")
            
        logger.info(f"Total time: {minutes}m {seconds}s")
        logger.info("-" * 30)

if __name__ == "__main__":
    main()
