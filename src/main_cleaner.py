import argparse
from combining_audio import combine_audio
from spec_subtraction_same_file import spectral_subtraction_auto_noise
import librosa
import soundfile as sf



def main():
    parser = argparse.ArgumentParser(description="Combine clean and noise audio, then denoise with spectral subtraction.")
    parser.add_argument("--clean", required=True, help="Path to clean audio file")
    parser.add_argument("--noise", help="Path to noise audio file")
    parser.add_argument("--noise_category", help="Noise category (optional)")
    parser.add_argument("--noise_level", type=float, default=0.2, help="Relative noise level (default: 0.2)")
    parser.add_argument("--mixed_out", default="mixed.wav", help="Output path for mixed audio")
    parser.add_argument("--denoised_out", default="denoised.wav", help="Output path for denoised audio")
    args = parser.parse_args()

    # Determine noise file: use --noise if provided, else use --noise_category
    noise_file = args.noise
    if args.noise_category and not noise_file:
        from find_data_categories import return_file_path
        file_paths = return_file_path([args.noise_category])
        if file_paths:
            noise_file = file_paths[0]
            print(f"Using noise file from category '{args.noise_category}': {noise_file}")
        else:
            print(f"No files found for category '{args.noise_category}'.")
            return

    if not noise_file:
        print("Please provide either --noise or --noise_category.")
        return

    # Step 1: Combine clean and noise
    mixed_file = combine_audio(args.clean, noise_file, args.mixed_out, args.noise_level)

    # Step 2: Denoise with spectral subtraction
    y_mixed, sr = librosa.load(mixed_file, sr=16000)
    y_denoised = spectral_subtraction_auto_noise(y_mixed, sr)
    sf.write(args.denoised_out, y_denoised, sr)
    print(f"Denoised audio saved to {args.denoised_out}")

if __name__ == "__main__":
    main()