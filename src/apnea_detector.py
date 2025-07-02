import numpy as np
import soundfile as sf
import argparse

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


def main():
    parser = argparse.ArgumentParser(description="Detect apnea events in an audio file.")
    parser.add_argument('--input', required=True, help='Input audio file path')
    parser.add_argument('--frame_sec', type=float, default=1, help='Frame size in seconds (default: 1)')
    parser.add_argument('--silence_thresh', type=float, default=1e-4, help='Silence threshold (default: 1e-4)')
    parser.add_argument('--min_apnea_sec', type=float, default=10, help='Minimum apnea duration in seconds (default: 10)')
    args = parser.parse_args()

    audio, sr = sf.read(args.input)
    apnea_events = detect_apnea(audio, sr, frame_sec=args.frame_sec, silence_thresh=args.silence_thresh, min_apnea_sec=args.min_apnea_sec)
    print("Detected apnea events (start, end in seconds):", apnea_events)

if __name__ == "__main__":
    main()