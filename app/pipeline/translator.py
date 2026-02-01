import os
import time
import logging
import requests
from typing import List, Optional

logger = logging.getLogger(__name__)

class SubtitleTranslator:
    def __init__(self, strategy: str = "libre", context: Optional[str] = None, url: Optional[str] = None, source_lang: Optional[str] = None, target_lang: Optional[str] = None):
        self.strategy = strategy
        self.context = context
        
        # -----------------------------------------------------------
        # LANGUAGE CONFIGURATION
        # Logic: Constructor Argument > Environment Variable > Default
        # -----------------------------------------------------------
        
        # 1. Source Language (Dynamic from Whisper detection)
        self.source_lang = source_lang or os.getenv("TRANSLATE_FROM_LANG", "en")
        
        # 2. Target Language (From Docker ENV, default NL)
        self.target_lang = target_lang or os.getenv("TRANSLATE_TO_LANG", "nl")

        # -----------------------------------------------------------
        # CONNECTION CONFIGURATION
        # -----------------------------------------------------------
        
        if self.strategy == "llm":
            self.url = url or os.getenv("LLM_URL", "http://ollama:11434")
            self.batch_size = int(os.getenv("LLM_BATCH_SIZE", 8))
        else:
            self.url = os.getenv("LIBRETRANSLATE_URL", "http://translator:5000")
            self.batch_size = int(os.getenv("TRANSLATE_BATCH_SIZE", 15))

        self.connect_timeout = 5
        self.read_timeout = 120
        self.max_retries = int(os.getenv("TRANSLATE_MAX_RETRIES", 2))

        # Logging
        mode_display = "LLM (Context-Aware)" if self.strategy == "llm" else "Standard (LibreTranslate)"
        logger.info(f"Translator initialized | Strategy: {mode_display} | {self.source_lang.upper()} -> {self.target_lang.upper()} | URL: {self.url}")

    def is_available(self) -> bool:
        """Check if the translation service is reachable."""
        try:
            endpoint = "/languages" if self.strategy == "libre" else "/api/tags"
            r = requests.get(f"{self.url}{endpoint}", timeout=3)
            return r.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed for {self.strategy}: {e}")
            return False

    def translate_segments(self, segments: List[dict]) -> Optional[List[dict]]:
        if not segments:
            return None
        if not self.is_available():
            logger.error(f"Translation service ({self.strategy}) not reachable at {self.url}")
            return None

        # Filter segments with actual text
        texts = [s.get("text", "").strip() for s in segments]
        non_empty_indices = [i for i, t in enumerate(texts) if t]
        
        if not non_empty_indices:
            logger.warning("No translatable text found in segments")
            return segments

        texts_to_translate = [texts[i] for i in non_empty_indices]
        logger.info(f"Starting {self.strategy} translation to {self.target_lang.upper()} ({len(texts_to_translate)} lines)")

        try:
            # Execute translation via batch processor
            translated_all = self._translate_in_batches(texts_to_translate)

            if len(translated_all) != len(texts_to_translate):
                raise RuntimeError(f"Count mismatch: {len(translated_all)} translated, {len(texts_to_translate)} expected")

            # Map translations back to original segment structure
            translated_segments = []
            translate_idx = 0
            for idx, seg in enumerate(segments):
                seg_copy = seg.copy()
                if idx in non_empty_indices:
                    seg_copy["text"] = translated_all[translate_idx]
                    translate_idx += 1
                translated_segments.append(seg_copy)

            logger.info("Translation process finished successfully")
            return translated_segments

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return None

    def _translate_in_batches(self, texts: List[str]) -> List[str]:
        #Splits the list into chunks and processes them through the retry handler.
        translated_all = []
        for batch in self._chunked(texts, self.batch_size):
            translated_all.extend(self._translate_batch_with_retry(batch))
        return translated_all

    def _translate_batch_with_retry(self, batch: List[str]) -> List[str]:
        #Retries a batch translation on failure or timeout.
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                if self.strategy == "llm":
                    return self._translate_batch_llm(batch)
                else:
                    return self._translate_batch_libre(batch)
            except Exception as e:
                last_error = e
                logger.warning(f"Batch failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt < self.max_retries:
                    time.sleep(2)
        
        raise RuntimeError(f"Batch translation failed after {self.max_retries} retries: {last_error}")

    def _translate_batch_libre(self, batch: List[str]) -> List[str]:
        response = requests.post(
            f"{self.url}/translate",
            json={
                "q": batch,
                "source": self.source_lang,
                "target": self.target_lang,
                "format": "text",
            },
            timeout=(self.connect_timeout, self.read_timeout),
        )
        response.raise_for_status()
        return response.json().get("translatedText", [])

    def _translate_batch_llm(self, batch: List[str]) -> List[str]:
        model_name = os.getenv("LLM_MODEL", "llama3.1")
        
        # Professional English System Prompt
        system_prompt = (
            f"You are a professional subtitle translator specializing in {self.source_lang} to {self.target_lang} translation. "
            f"Your goal is to provide natural, fluid, and contextually accurate subtitles in {self.target_lang}."
        )
        
        if self.context:
            system_prompt += f"\n\nCRITICAL CONTEXT AND INSTRUCTIONS:\n{self.context}"

        # Instructions in English to ensure the LLM follows the formatting rules
        user_prompt = (
            f"Translate the following {len(batch)} lines into {self.target_lang}. "
            "Maintain the original meaning and tone. If a line is a name or doesn't require translation, leave it as is. "
            "Output ONLY the translated text, one line per input line. Do not add numbering or explanations. "
            "Maintain the exact order:\n\n"
            + "\n".join(batch)
        )

        response = requests.post(
            f"{self.url}/api/generate",
            json={
                "model": model_name,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=self.read_timeout
        )
        response.raise_for_status()
        raw_text = response.json().get("response", "").strip()
        
        # Split and clean
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        
        # Safety check for missing lines
        while len(lines) < len(batch):
            lines.append("[translation missing]")
            
        return lines[:len(batch)]
        
    @staticmethod
    def _chunked(items: List[str], size: int):
        for i in range(0, len(items), size):
            yield items[i : i + size]
