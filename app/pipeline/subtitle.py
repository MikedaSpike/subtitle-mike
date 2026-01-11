import re
from utils.timing import format_srt_time

class Subtitle:
    def __init__(self, segments, max_cpl=42, min_duration=1.0, max_gap=0.5):
        self.segments = segments
        self.max_cpl = max_cpl
        self.min_duration = min_duration
        self.max_gap = max_gap

    def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\[SPEAKER_\d+\]', '', text)
        text = text.lstrip("- ")
        return text

    def wrap_text(self, text):
        """Splits text into balanced lines."""
        text = text.strip()
        if len(text) <= self.max_cpl or "\n" in text:
            return text

        words = text.split()
        if len(words) <= 1:
            return text

        mid_point = len(text) / 2
        best_split_point = -1
        min_diff = float('inf')

        # Find the split point closest to the middle
        current_len = 0
        for i in range(len(words) - 1):
            line1 = " ".join(words[:i+1])
            line2 = " ".join(words[i+1:])
            
            if len(line1) <= self.max_cpl and len(line2) <= self.max_cpl:
                diff = abs(len(line1) - mid_point)
                if diff < min_diff:
                    min_diff = diff
                    best_split_point = i + 1

        if best_split_point != -1:
            line1 = " ".join(words[:best_split_point])
            line2 = " ".join(words[best_split_point:])
            return f"{line1}\n{line2}"
        
        # Fallback if balancing fails
        return text

    def process_and_save(self, output_path, blacklist):
        # 1. Clean and filter
        raw_segments = []
        for seg in self.segments:
            txt = self.clean_text(seg['text'])
            if txt and not any(b.lower() in txt.lower() for b in blacklist):
                raw_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': txt,
                    'speaker': seg.get('speaker')
                })

        if not raw_segments:
            return

        # 2. Smart Merging
        merged = []
        curr = raw_segments[0]
        
        for i in range(1, len(raw_segments)):
            nxt = raw_segments[i]
            gap = nxt['start'] - curr['end']
            
            # Merge conditions: small gap AND combined length fits reasonable limit
            can_merge = gap <= self.max_gap and (len(curr['text']) + len(nxt['text'])) < (self.max_cpl * 1.8)
            
            if can_merge:
                if curr['speaker'] != nxt['speaker']:
                    # Dialogue: different speakers -> dashes
                    curr['text'] = f"- {curr['text']}\n- {nxt['text']}"
                else:
                    # Same speaker -> join text
                    curr['text'] = f"{curr['text']} {nxt['text']}"
                curr['end'] = nxt['end']
            else:
                merged.append(curr)
                curr = nxt
        merged.append(curr)

        # 3. Write to SRT
        with open(output_path, "w", encoding="utf-8") as f:
            for i, s in enumerate(merged, 1):
                start = s['start']
                end = s['end']
                
                # Minimum duration enforcement
                if (end - start) < self.min_duration:
                    end = start + self.min_duration

                formatted_text = self.wrap_text(s['text'])
                
                f.write(f"{i}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{formatted_text}\n\n")