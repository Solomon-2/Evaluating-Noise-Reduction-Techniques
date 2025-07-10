
from logmmse import logmmse_from_file
import argparse
import soundfile as sf

def run_logmmse(input_path, output_path=None, initial_noise=6, window_size=0, noise_threshold=0.15):
    """Run logmmse denoising on a file. Optionally save to output_path."""
    # The correct API returns only y_denoised (audio array), not (audio, sr)
    y_denoised = logmmse_from_file(
        input_path,
        initial_noise=initial_noise,
        window_size=window_size,
        noise_threshold=noise_threshold
    )
    # Set your default sample rate (change if needed)
    sr = 16000
    if output_path:
        sf.write(output_path, y_denoised, sr)
        print(f"Denoised audio saved to {output_path}")
    return y_denoised, sr


def main():
    parser = argparse.ArgumentParser(description="Apply logMMSE denoising to an audio file.")
    parser.add_argument('--input', required=True, help='Input noisy audio file path')
    parser.add_argument('--output', required=False, help='Output denoised audio file path (optional)')
    parser.add_argument('--initial_noise', type=int, default=6, help='Number of initial frames for noise estimation (default: 6)')
    parser.add_argument('--window_size', type=int, default=0, help='Window size for processing (default: 0, auto)')
    parser.add_argument('--noise_threshold', type=float, default=0.15, help='Noise threshold (default: 0.15)')
    args = parser.parse_args()


    import os
    def is_wav_file(path):
        return os.path.isfile(path) and path.lower().endswith('.wav')

    # If input is a directory, process all .wav files in it
    if os.path.isdir(args.input):
        input_dir = args.input
        output_dir = args.output
        if not output_dir:
            print("Please provide --output as output directory when using directory input.")
            return
        os.makedirs(output_dir, exist_ok=True)
        wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.wav')]
        if not wav_files:
            print(f"No .wav files found in directory {input_dir}")
            return
        for fname in wav_files:
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            print(f"Processing {in_path} -> {out_path}")
            run_logmmse(
                in_path,
                out_path,
                initial_noise=args.initial_noise,
                window_size=args.window_size,
                noise_threshold=args.noise_threshold
            )
    else:
        run_logmmse(
            args.input,
            args.output,
            initial_noise=args.initial_noise,
            window_size=args.window_size,
            noise_threshold=args.noise_threshold
        )

if __name__ == "__main__":
    main()

