import numpy as np
import soundfile as sf
import argparse

def inject_apnea(input_path, output_path, apnea_duration_sec, event_starts_sec):
    """
    Injects apnea (silence) events into an audio file.
    input_path: path to input audio
    output_path: path to save output audio
    apnea_duration_sec: duration of each apnea event in seconds
    event_starts_sec: list of start times (in seconds) for apnea events
    """
    audio, sr = sf.read(input_path)
    apnea_samples = int(apnea_duration_sec * sr)
    for start_sec in event_starts_sec:
        start = int(start_sec * sr)
        end = min(start + apnea_samples, len(audio))
        audio[start:end] = np.random.normal(0, 0.001, size=end-start)
    sf.write(output_path, audio, sr)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Inject apnea (silence) events into an audio file.")
    parser.add_argument('--input', required=True, help='Input audio file path')
    parser.add_argument('--output', required=True, help='Output audio file path')
    parser.add_argument('--apnea_duration', type=float, required=True, help='Apnea event duration in seconds')
    parser.add_argument('--event_starts', required=True, help='Comma-separated list of apnea event start times (in seconds)')
    args = parser.parse_args()

    # Parse event_starts as a list of floats
    event_starts_sec = [float(x) for x in args.event_starts.split(',')]
    inject_apnea(args.input, args.output, args.apnea_duration, event_starts_sec)
    print(f"Output with apnea events saved to {args.output}")

if __name__ == "__main__":
    main()