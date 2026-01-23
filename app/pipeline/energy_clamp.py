import numpy as np
import librosa

def clamp_segments_energy(
    segments,
    audio,
    sr,
    energy_threshold=0.02,
    frame_length=2048,
    hop_length=256,
    min_duration=0.3,
    pad_start=0.04,
    pad_end=0.06,
):
    """
    Clamp segment start/end based on RMS energy.
    """

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=hop_length,
    )

    for seg in segments:
        start = seg["start"]
        end = seg["end"]

        mask = (times >= start) & (times <= end)
        idx = np.where(mask)[0]

        if len(idx) == 0:
            continue

        energies = rms[idx]
        voiced = energies > energy_threshold

        if not voiced.any():
            continue

        # Simpler and safer
        voiced_idx = idx[voiced]
        first = voiced_idx[0]
        last = voiced_idx[-1]

        new_start = max(times[first] - pad_start, start)
        new_end = min(times[last] + pad_end, end)

        if new_end - new_start >= min_duration:
            seg["start"] = round(new_start, 3)
            seg["end"] = round(new_end, 3)

    return segments