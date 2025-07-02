import argparse
import librosa
import numpy as np
import soundfile as sf


def replicate_audio(input_path, output_path, factor):
    """
    Replicates the input audio by the given factor (float or int).
    If factor is not an integer, the fractional part is filled by appending the corresponding fraction of the audio.
    """
    y, sr = librosa.load(input_path, sr=None)
    int_part = int(np.floor(factor))
    frac_part = factor - int_part

    # Repeat the audio int_part times
    y_replicated = np.tile(y, int_part)

    # Add the fractional part if needed
    if frac_part > 0:
        extra_len = int(len(y) * frac_part)
        y_replicated = np.concatenate([y_replicated, y[:extra_len]])

    sf.write(output_path, y_replicated, sr)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Replicate audio by a given factor.")
    parser.add_argument("--input", required=True, help="Input audio file path")
    parser.add_argument("--output", required=True, help="Output audio file path")
    parser.add_argument("--factor", type=float, required=True, help="Replication factor (e.g., 1.5)")
    args = parser.parse_args()

    replicate_audio(args.input, args.output, args.factor)
    print(f"Replicated audio saved to {args.output}")

if __name__ == "__main__":
    main()
