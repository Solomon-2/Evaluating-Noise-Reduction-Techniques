import numpy as np
import soundfile as sf

def detect_apnea(audio, sr, frame_sec=1, silence_thresh=1e-4, min_apnea_sec=10):
    frame_len = int(frame_sec * sr)
    energies = [
        np.mean(np.abs(audio[i:i+frame_len]))
        for i in range(0, len(audio), frame_len)
    ]

    apnea_frames = [i for i, e in enumerate(energies) if e < silence_thresh]

    # Group consecutive frames
    apneas = []
    start = None
    prev = None

    for idx in apnea_frames:
        if start is None:
            start = idx
        elif prev is not None and idx != prev + 1:
            # End of a group
            duration = (prev - start + 1) * frame_sec
            if duration >= min_apnea_sec:
                apneas.append((start * frame_sec, (prev + 1) * frame_sec))
            start = idx
        prev = idx

    # Check last group
    if start is not None and prev is not None:
        duration = (prev - start + 1) * frame_sec
        if duration >= min_apnea_sec:
            apneas.append((start * frame_sec, (prev + 1) * frame_sec))

    return apneas

audio, sr = sf.read("output_with_apnea.wav")
apnea_events = detect_apnea(audio, sr)
print("Detected apnea events (start, end in seconds):", apnea_events)