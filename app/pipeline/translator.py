import os
import logging
import requests

logger = logging.getLogger(__name__)

class SubtitleTranslator:
    def __init__(self):
        self.url = os.getenv("LIBRETRANSLATE_URL", "http://translator:5000")
        self.target_lang = os.getenv("TRANSLATE_TO_LANG", "nl")
        
    def translate_segments(self, segments):
        if not segments:
            return segments

        logger.info(f"Starting batch translation to {self.target_lang.upper()} via {self.url}...")
        
        try:
            texts_to_translate = [s['text'].strip() for s in segments if s['text'].strip()]
            
            if not texts_to_translate:
                return segments

            response = requests.post(
                f"{self.url}/translate",
                json={
                    "q": texts_to_translate, # LibreTranslate accepteert een lijst!
                    "source": "en",
                    "target": self.target_lang,
                    "format": "text"
                },
                timeout=60 # Geef de 20 cores de tijd om te rekenen
            )
            
            if response.status_code == 200:
                translated_texts = response.json().get('translatedText', [])
                
                translate_idx = 0
                for s in segments:
                    if s['text'].strip():
                        s['text'] = translated_texts[translate_idx]
                        translate_idx += 1
                
                logger.info("Translation completed successfully.")
            else:
                logger.warning(f"Translation API returned error: {response.text}")
                
            return segments
            
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            return segments