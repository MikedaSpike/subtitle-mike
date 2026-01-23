import re
import math


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
            speaker = seg.get("speaker")

            # 1. translated text has priority
            text = (
                seg.get("translated_text")
                or seg.get("text")
                or ""
            ).strip()

            if not text:
                continue

            start = float(seg["start"])
            end = float(seg["end"])

            sentences.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": self._clean_text(text),
                }
            )

        return sentences

    # ============================================================
    # CUE BUILDING (SENTENCE-FIRST)
    # ============================================================

    def _sentences_to_cues(self, sentences):
        cues = []
        current = None

        for s in sentences:
            if current is None:
                current = self._new_cue_from_sentence(s)
                continue

            gap = s["start"] - current["end"]

            trial_sentences = current["sentences"] + [s]
            trial_text = self._format_text(trial_sentences)
            trial_duration = s["end"] - current["start"]

            if (
                gap > self.max_gap
                or len(self._collect_speakers(trial_sentences)) > 2
                or trial_duration > self.max_duration
                or self._exceeds_layout(trial_text)
            ):
                cues.append(self._finalize_cue(current))
                current = self._new_cue_from_sentence(s)
            else:
                current["sentences"].append(s)
                current["end"] = s["end"]

        if current:
            cues.append(self._finalize_cue(current))

        return cues

    def _new_cue_from_sentence(self, s):
        return {
            "start": s["start"],
            "end": s["end"],
            "sentences": [s],
        }

    def _finalize_cue(self, cue):
        text = self._format_text(cue["sentences"])
        return {
            "start": cue["start"],
            "end": cue["end"],
            "lines": text.split("\n"),
        }

    # ============================================================
    # TEXT FORMATTING (SITCOM RULES)
    # ============================================================

    def _format_text(self, sentences):
        if len(sentences) == 1:
            return self._wrap_text(sentences[0]["text"])

        s1, s2 = sentences[0], sentences[1]
        l1 = self._wrap_text(s1["text"])
        l2 = self._wrap_text(s2["text"])

        if s1["speaker"] and s2["speaker"] and s1["speaker"] != s2["speaker"]:
            return f"{l1}\n- {l2}"

        return f"{l1}\n{l2}"

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

        return "\n".join(lines[: self.max_lines])

    def _exceeds_layout(self, text):
        lines = text.split("\n")
        if len(lines) > self.max_lines:
            return True
        return any(len(l) > self.max_cpl for l in lines)

    # ============================================================
    # TIMING ENFORCEMENT (NO FLASHING)
    # ============================================================

    def _enforce_timing(self, cues):
        for i, c in enumerate(cues):
            duration = c["end"] - c["start"]

            # minimum duration
            if duration < self.min_duration:
                c["end"] = c["start"] + self.min_duration

            # CPS clamp (extend duration if needed)
            text_len = sum(len(l) for l in c["lines"])
            min_dur = text_len / self.max_cps
            if (c["end"] - c["start"]) < min_dur:
                c["end"] = c["start"] + min_dur

            # prevent overlap
            if i > 0 and cues[i - 1]["end"] > c["start"]:
                cues[i - 1]["end"] = c["start"]

        return cues

    # ============================================================
    # OUTPUT
    # ============================================================

    def _write_srt(self, path, cues, blacklist):
        def ts(t):
            ms = int((t % 1) * 1000)
            s = int(t)
            return f"{s//3600:02}:{(s%3600)//60:02}:{s%60:02},{ms:03}"

        with open(path, "w", encoding="utf-8") as f:
            idx = 1
            for c in cues:
                text = "\n".join(c["lines"])

                if any(b.lower() in text.lower() for b in blacklist):
                    continue

                f.write(f"{idx}\n")
                f.write(f"{ts(c['start'])} --> {ts(c['end'])}\n")
                f.write(text + "\n\n")
                idx += 1

    # ============================================================
    # UTILS
    # ============================================================

    def _collect_speakers(self, sentences):
        return [s["speaker"] for s in sentences if s.get("speaker")]

    def _clean_text(self, text):
        text = re.sub(r"\s+([,.!?])", r"\1", text)
        return re.sub(r"\s+", " ", text).strip()
