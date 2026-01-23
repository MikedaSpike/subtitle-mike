import os
import time
import logging
import requests
from typing import List, Optional

logger = logging.getLogger(__name__)


class SubtitleTranslator:
    """
    Safe, batch-based subtitle translator using LibreTranslate.

    Guarantees:
    - Never returns partially translated segments
    - Never returns original text on failure
    - Fails fast if the service is unavailable
    """

    def __init__(self):
        self.url = os.getenv("LIBRETRANSLATE_URL", "http://translator:5000")
        self.source_lang = os.getenv("TRANSLATE_FROM_LANG", "en")
        self.target_lang = os.getenv("TRANSLATE_TO_LANG", "nl")

        # Tuning
        self.batch_size = int(os.getenv("TRANSLATE_BATCH_SIZE", 15))
        self.connect_timeout = int(os.getenv("TRANSLATE_CONNECT_TIMEOUT", 5))
        self.read_timeout = int(os.getenv("TRANSLATE_READ_TIMEOUT", 120))
        self.max_retries = 1

    # ------------------------------------------------------------------
    # HEALTH CHECK
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if LibreTranslate is reachable."""
        try:
            r = requests.get(f"{self.url}/languages", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def translate_segments(self, segments: List[dict]) -> Optional[List[dict]]:
        """
        Translate subtitle segments.

        Returns:
            - translated segments on success
            - None on ANY failure
        """

        if not segments:
            logger.warning("No segments provided for translation")
            return None

        if not self.is_available():
            logger.error("Translation service is not reachable")
            return None

        texts = [s.get("text", "").strip() for s in segments]
        non_empty_indices = [i for i, t in enumerate(texts) if t]

        if not non_empty_indices:
            logger.warning("No translatable text found in segments")
            return None

        texts_to_translate = [texts[i] for i in non_empty_indices]

        logger.info(
            f"Starting translation to {self.target_lang.upper()} "
            f"({len(texts_to_translate)} sentences, batch size {self.batch_size})"
        )

        try:
            translated_texts = self._translate_in_batches(texts_to_translate)

            if len(translated_texts) != len(texts_to_translate):
                raise RuntimeError("Translated sentence count mismatch")

            # Apply translations back to segments
            translated_segments = []
            translate_idx = 0

            for idx, seg in enumerate(segments):
                seg_copy = seg.copy()

                if idx in non_empty_indices:
                    seg_copy["text"] = translated_texts[translate_idx]
                    translate_idx += 1

                translated_segments.append(seg_copy)

            logger.info("Translation completed successfully")
            return translated_segments

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # INTERNALS
    # ------------------------------------------------------------------

    def _translate_in_batches(self, texts: List[str]) -> List[str]:
        translated_all = []

        for batch in self._chunked(texts, self.batch_size):
            translated_all.extend(self._translate_batch_with_retry(batch))

        return translated_all

    def _translate_batch_with_retry(self, batch: List[str]) -> List[str]:
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return self._translate_batch(batch)
            except requests.exceptions.ReadTimeout as e:
                last_error = e
                logger.warning(
                    f"Translation timeout (attempt {attempt + 1}/{self.max_retries + 1})"
                )
                time.sleep(2)

        raise RuntimeError(f"Translation batch failed after retries: {last_error}")

    def _translate_batch(self, batch: List[str]) -> List[str]:
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

        data = response.json()
        translated = data.get("translatedText")

        if not isinstance(translated, list):
            raise RuntimeError("Invalid translation response format")

        return translated

    @staticmethod
    def _chunked(items: List[str], size: int):
        for i in range(0, len(items), size):
            yield items[i : i + size]
