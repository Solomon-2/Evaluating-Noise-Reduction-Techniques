import numpy as np
import soundfile as sf
import argparse
import csv
import os
import glob

def detect_apnea(audio, sr, frame_sec=1, silence_thresh=1e-2, min_apnea_sec=9):
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
    parser = argparse.ArgumentParser(description="Detect apnea events in audio files.")
    parser.add_argument('--input', nargs='+', help='Input audio file path(s)')
    parser.add_argument('--dir', type=str, default=None, help='Directory containing audio files to process (all .wav files)')
    parser.add_argument('--frame_sec', type=float, default=1, help='Frame size in seconds (default: 1)')
    parser.add_argument('--silence_thresh', type=float, default=0.5, help='Silence threshold (default: 1e-4)')
    parser.add_argument('--min_apnea_sec', type=float, default=10, help='Minimum apnea duration in seconds (default: 10)')
    parser.add_argument('--output_csv', type=str, default=None, help='Optional: Output CSV file for detected events')
    args = parser.parse_args()

    # Collect files from --input and/or --dir
    files_to_process = []
    if args.input:
        files_to_process.extend(args.input)
    if args.dir:
        wav_files = glob.glob(os.path.join(args.dir, '*.wav'))
        files_to_process.extend(wav_files)

    if not files_to_process:
        print("No input files provided. Use --input or --dir.")
        return

    all_results = []
    for input_file in files_to_process:
        try:
            audio, sr = sf.read(input_file)
            apnea_events = detect_apnea(audio, sr, frame_sec=args.frame_sec, silence_thresh=args.silence_thresh, min_apnea_sec=args.min_apnea_sec)
            print(f"{input_file}: Detected apnea events (start, end in seconds): {apnea_events}")
            for start, end in apnea_events:
                all_results.append([input_file, start, end])
        except Exception as e:
            print(f"Error processing {input_file}: {e}")

    if args.output_csv and all_results:
        with open(args.output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['filename', 'start_sec', 'end_sec'])
            writer.writerows(all_results)
        print(f"Apnea events saved to {args.output_csv}")

if __name__ == "__main__":
    main()