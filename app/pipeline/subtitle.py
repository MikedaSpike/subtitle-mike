import re
import math
import copy

def format_ts(seconds: float) -> str:
    ms = int((seconds - int(seconds)) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

class SubtitleEngine:
    def __init__(
        self,
        segments,
        max_cpl=42,
        max_lines=2,
        min_duration=1.4,
        max_duration=7.0,
        min_cps=12.0,
        max_cps=14.0,
        max_gap=0.4,
    ):
        self.segments = segments
        self.max_cpl = max_cpl
        self.max_lines = max_lines
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_cps = min_cps
        self.max_cps = max_cps
        self.max_gap = max_gap

    # ============================================================
    # PUBLIC
    # ============================================================

    def process_and_save(self, path, blacklist):
        sentences = self._extract_sentences()
        cues = self._sentences_to_cues(sentences)
        cues = self._enforce_timing(cues)
        self._write_srt(path, cues, blacklist)

    # ============================================================
    # SENTENCE EXTRACTION (NO LOSS POSSIBLE)
    # ============================================================

    def _extract_sentences(self):
        sentences = []
        for seg in self.segments:
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg["text"].strip()
            if not text:
                continue
            sentences.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
                "speaker": speaker
            })
        return sentences

    # ============================================================
    # WRAPPING & LAYOUT
    # ============================================================

    def _wrap_text(self, text):
        words = text.split()
        lines = []
        current = ""
        for w in words:
            if not current:
                current = w
            elif len(current) + len(w) + 1 <= self.max_cpl:
                current += " " + w
            else:
                lines.append(current)
                current = w
        if current:
            lines.append(current)
        return "\n".join(lines)

    def _exceeds_layout(self, wrapped_text):
        lines = wrapped_text.split("\n")
        if len(lines) > self.max_lines:
            return True
        for l in lines:
            if len(l) > self.max_cpl:
                return True
        return False

    # ============================================================
    # SEGMENT SPLITTING (FOR LONG SENTENCES)
    # ============================================================

    def _split_long_segment(self, s):
        """Breekt een segment op als het op zichzelf al te groot is voor 1 cue."""
        words = s["text"].split()
        chunks = []
        current_words = []
        start_t = s["start"]
        total_len = len(s["text"])

        # Eerst verzamelen we alle brokken (chunks)
        temp_chunks = []
        for w in words:
            trial = " ".join(current_words + [w])
            wrapped = self._wrap_text(trial)
            
            if self._exceeds_layout(wrapped) and current_words:
                chunk_text = " ".join(current_words)
                ratio = len(chunk_text) / total_len
                end_t = start_t + (s["end"] - s["start"]) * ratio
                
                temp_chunks.append({
                    "start": start_t,
                    "end": end_t,
                    "text": chunk_text,
                    "speaker": s["speaker"]
                })
                start_t = end_t
                current_words = [w]
            else:
                current_words.append(w)

        if current_words:
            temp_chunks.append({
                "start": start_t,
                "end": s["end"],
                "text": " ".join(current_words),
                "speaker": s["speaker"]
            })

        # Nu verwerken we de brokken en voegen we de puntjes toe
        num_chunks = len(temp_chunks)
        for i, chunk in enumerate(temp_chunks):
            text = chunk["text"]
            # Als dit NIET de laatste chunk is, voeg ... toe
            if i < num_chunks - 1:
                text = text.strip() + "..."
            
            chunks.append({
                "start": chunk["start"],
                "end": chunk["end"],
                "lines": self._wrap_text(text).split("\n"),
                "speaker": chunk["speaker"]
            })
            
        return chunks

    # ============================================================
    # LOGIC
    # ============================================================

    def _sentences_to_cues(self, sentences):
        cues = []
        buffer = []

        for s in sentences:
            # Als een enkel segment al te lang is -> splitsen
            if self._exceeds_layout(self._wrap_text(s["text"])):
                if buffer:
                    cues.append(self._finalize_cue(buffer))
                    buffer = []
                cues.extend(self._split_long_segment(s))
                continue

            if not buffer:
                buffer.append(s)
                continue

            gap = s["start"] - buffer[-1]["end"]
            combined_text = " ".join([b["text"] for b in buffer] + [s["text"]])
            duration = s["end"] - buffer[0]["start"]
            same_speaker = all(b["speaker"] == s["speaker"] for b in buffer)

            if (gap > self.max_gap or 
                duration > self.max_duration or 
                not same_speaker or 
                self._exceeds_layout(self._wrap_text(combined_text))):
                
                cues.append(self._finalize_cue(buffer))
                buffer = [s]
            else:
                buffer.append(s)

        if buffer:
            cues.append(self._finalize_cue(buffer))
        return cues

    # ============================================================
    # CUE FINALIZATION
    # ============================================================

    def _finalize_cue(self, buffer_list):
        full_text = " ".join([b["text"] for b in buffer_list])
        return {
            "start": buffer_list[0]["start"],
            "end": buffer_list[-1]["end"],
            "lines": self._wrap_text(full_text).split("\n"),
            "speaker": buffer_list[0]["speaker"]
        }

    # ============================================================
    # TIMING ENFORCEMENT
    # ============================================================

    def _enforce_timing(self, cues):
        for i, c in enumerate(cues):
            text_len = sum(len(l) for l in c["lines"])
            min_dur = max(self.min_duration, text_len / self.max_cps)
            if (c["end"] - c["start"]) < min_dur:
                c["end"] = c["start"] + min_dur
            if i > 0 and cues[i - 1]["end"] > c["start"]:
                cues[i - 1]["end"] = c["start"]
        return cues

    # ============================================================
    # Software Alligment
    # ============================================================

    def align_timings_heuristically(self):
        """
        Corrigeert de start- en eindtijd van segmenten op basis van CPS uitschieters.
        Minder agressief door gebruik van buffers en gemiddelde CPS targets.
        """
        aligned_segments = []
        for seg in self.segments:
            s = copy.deepcopy(seg)
            words = s.get("words", [])
            if not words:
                aligned_segments.append(s)
                continue

            # Gebruik een veilige target CPS (midden van je range) voor correcties
            target_cps = (self.min_cps + self.max_cps) / 2
            
            # --- 1. START TIMING (EERSTE WOORD) ---
            first_w = words[0]
            dur_first = first_w["end"] - first_w["start"]
            
            # Alleen corrigeren als het woord langer duurt dan 0.4s (minder agressief)
            # en de CPS echt veel te laag is (< min_cps)
            if dur_first > 0.4: 
                word_cps = len(first_w["word"]) / dur_first
                if word_cps < self.min_cps:
                    # Bereken ideale duur, maar laat altijd minstens 0.3s staan
                    ideal_dur = max(0.3, len(first_w["word"]) / target_cps)
                    # Verschuif starttijd naar achteren
                    s["start"] = max(s["start"], first_w["end"] - ideal_dur)

            # --- 2. END TIMING (LAATSTE WOORD) ---
            last_w = words[-1]
            dur_last = last_w["end"] - last_w["start"]
            
            if dur_last > 0.4:
                word_cps = len(last_w["word"]) / dur_last
                if word_cps < self.min_cps:
                    ideal_dur = max(0.3, len(last_w["word"]) / target_cps)
                    # Verschuif eindtijd naar voren
                    s["end"] = min(s["end"], last_w["start"] + ideal_dur)

            aligned_segments.append(s)
        
        self.segments = aligned_segments
        return self.segments
        
    # ============================================================
    # OUTPUT
    # ============================================================

    def _write_srt(self, path, cues, blacklist):
        with open(path, "w", encoding="utf-8") as f:
            idx = 1
            for c in cues:
                text = "\n".join(c["lines"])
                if any(b.lower() in text.lower() for b in blacklist):
                    continue
                f.write(f"{idx}\n")
                f.write(f"{format_ts(c['start'])} --> {format_ts(c['end'])}\n")
                f.write(text + "\n\n")
                idx += 1
