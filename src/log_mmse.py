from logmmse import logmmse_from_file
import argparse

def run_logmmse(input_path, output_path=None, initial_noise=6, window_size=0, noise_threshold=0.15):
    """Run logmmse denoising on a file. Optionally save to output_path."""
    result = logmmse_from_file(
        input_path,
        output_filename=output_path,
        initial_noise=initial_noise,
        window_size=window_size,
        noise_threshold=noise_threshold
    )
    if output_path:
        print(f"Denoised audio saved to {output_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Apply logMMSE denoising to an audio file.")
    parser.add_argument('--input', required=True, help='Input noisy audio file path')
    parser.add_argument('--output', required=False, help='Output denoised audio file path (optional)')
    parser.add_argument('--initial_noise', type=int, default=6, help='Number of initial frames for noise estimation (default: 6)')
    parser.add_argument('--window_size', type=int, default=0, help='Window size for processing (default: 0, auto)')
    parser.add_argument('--noise_threshold', type=float, default=0.15, help='Noise threshold (default: 0.15)')
    args = parser.parse_args()

    run_logmmse(
        args.input,
        args.output,
        initial_noise=args.initial_noise,
        window_size=args.window_size,
        noise_threshold=args.noise_threshold
    )

if __name__ == "__main__":
    main()

