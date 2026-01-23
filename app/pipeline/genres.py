# pipeline/genres.py

from pipeline.profiles import BroadcastProfile, NetflixProfile


GENRE_PROFILES = {
    "sitcom": {
        "profile": BroadcastProfile,
        "vad": {
            "threshold": 0.6,
            "min_silence_duration_ms": 150,
            "speech_pad_ms": 50,
            "vad_onset": 0.2,
            "vad_offset": 0.3,
        },
        "asr": {
            "beam_size": 1,
            "no_speech_threshold": 0.1,
        },
        "subtitle": {
            "max_gap": 0.3,
            "min_duration": 0.8,
        },
    },

    "drama": {
        "profile": BroadcastProfile,
        "vad": {
            "threshold": 0.5,
            "min_silence_duration_ms": 250,
            "speech_pad_ms": 100,
            "vad_onset": 0.2,
            "vad_offset": 0.35,
        },
        "asr": {
            "beam_size": 3,
            "no_speech_threshold": 0.05,
        },
        "subtitle": {
            "max_duration": 6.0,
            "prefer_semantic_breaks": True,
        },
    },

    "netflix": {
        "profile": NetflixProfile,
        "vad": {
            "threshold": 0.6,
            "min_silence_duration_ms": 200,
            "speech_pad_ms": 75,
            "vad_onset": 0.2,
            "vad_offset": 0.3,
        },
        "asr": {
            "beam_size": 2,
            "no_speech_threshold": 0.08,
        },
        "subtitle": {
            "max_cps": 15.0,
            "allow_sentence_merge": False,
        },
    },

    "film": {
        "profile": BroadcastProfile,
        "vad": {
            "threshold": 0.55,
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 80,
            "vad_onset": 0.12,
            "vad_offset": 0.28,
        },
        "asr": {
            "beam_size": 4,
            "no_speech_threshold": 0.07,
        },
        "subtitle": {
            "max_duration": 6.5,
            "prefer_semantic_breaks": True,
        },
    },

    "talkshow": {
        "profile": BroadcastProfile,
        "vad": {
            "threshold": 0.65,
            "min_silence_duration_ms": 120,
            "speech_pad_ms": 40,
            "vad_onset": 0.18,
            "vad_offset": 0.25,
        },
        "asr": {
            "beam_size": 2,
            "no_speech_threshold": 0.12,
        },
        "subtitle": {
            "max_gap": 0.25,
            "min_duration": 0.7,
        },
    },

    "podcast": {
        "profile": BroadcastProfile,
        "vad": {
            "threshold": 0.45,
            "min_silence_duration_ms": 350,
            "speech_pad_ms": 60,
            "vad_onset": 0.1,
            "vad_offset": 0.25,
        },
        "asr": {
            "beam_size": 5,
            "no_speech_threshold": 0.03,
        },
        "subtitle": {
            "max_duration": 7.0,
            "prefer_semantic_breaks": True,
        },
    },

    "music": {
        "profile": BroadcastProfile,
        "vad": {
            "threshold": 0.7,
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 50,
            "vad_onset": 0.05,
            "vad_offset": 0.15,
        },
        "asr": {
            "beam_size": 3,
            "no_speech_threshold": 0.15,
        },
        "subtitle": {
            "max_cps": 14.0,
            "min_duration": 1.0,
        },
    },

    "documentary": {
        "profile": BroadcastProfile,
        "vad": {
            "threshold": 0.5,
            "min_silence_duration_ms": 400,
            "speech_pad_ms": 90,
            "vad_onset": 0.1,
            "vad_offset": 0.3,
        },
        "asr": {
            "beam_size": 4,
            "no_speech_threshold": 0.05,
        },
        "subtitle": {
            "max_duration": 6.5,
            "prefer_semantic_breaks": True,
        },
    },

    "youtube": {
        "profile": BroadcastProfile,
        "vad": {
            "threshold": 0.6,
            "min_silence_duration_ms": 180,
            "speech_pad_ms": 50,
            "vad_onset": 0.15,
            "vad_offset": 0.25,
        },
        "asr": {
            "beam_size": 2,
            "no_speech_threshold": 0.1,
        },
        "subtitle": {
            "max_gap": 0.25,
            "min_duration": 0.7,
        },
    },

    "asmr": {
        "profile": BroadcastProfile,
        "vad": {
            "threshold": 0.35,
            "min_silence_duration_ms": 200,
            "speech_pad_ms": 80,
            "vad_onset": 0.02,
            "vad_offset": 0.1,
        },
        "asr": {
            "beam_size": 5,
            "no_speech_threshold": 0.02,
        },
        "subtitle": {
            "min_duration": 1.0,
            "max_duration": 6.0,
        },
    },

    "action": {
        "profile": BroadcastProfile,
        "vad": {
            "threshold": 0.7,
            "min_silence_duration_ms": 250,
            "speech_pad_ms": 60,
            "vad_onset": 0.2,
            "vad_offset": 0.35,
        },
        "asr": {
            "beam_size": 3,
            "no_speech_threshold": 0.1,
        },
        "subtitle": {
            "max_gap": 0.35,
            "min_duration": 0.9,
        },
    },
}