# pipeline/profiles.py

class SubtitleProfile:
    name = "base"

    # Layout
    max_cpl = 42
    max_lines = 2

    # Timing
    min_duration = 1.0
    max_duration = 7.0
    max_gap = 0.4

    # Reading speed
    min_cps = 12.0
    max_cps = 17.0

    # Structural behavior (NOT yet used, but valid)
    allow_sentence_merge = True
    aggressive_merge = False
    prefer_semantic_breaks = True
    merge_strategy = "semantic"
    split_strategy = "timing"

class BroadcastProfile(SubtitleProfile):
    name = "broadcast"

class NetflixProfile(SubtitleProfile):
    name = "netflix"

    max_cpl = 37
    min_duration = 0.83
    max_duration = 5.5
    max_cps = 15.0

    allow_sentence_merge = False
    aggressive_merge = True
