
import os
import librosa
import numpy as np
import soundfile as sf
import argparse

def spectral_subtraction_auto_noise(
    y_mixed,          # The noisy audio signal (numpy array)
    sr,               # Sampling rate
    noise_duration_sec=0.5, # How many seconds from the start to use for noise profile
    n_fft=2048,       # FFT window size
    hop_length=512,   # Hop length for STFT
    subtraction_factor=3.0,
    floor_factor=0.01,
    verbose=False
):
    """
    Performs spectral subtraction, estimating noise from the initial part
    of the input signal.
    """

    if verbose:
        print("Starting Spectral Subtraction (Auto Noise Estimation)...")
        print(f"  Noise Profile Estimation from first {noise_duration_sec} seconds.")
        print(f"  N_FFT: {n_fft}, Hop Length: {hop_length}")
        print(f"  Subtraction Factor: {subtraction_factor}, Floor Factor: {floor_factor}")

    # --- Step 1: Estimate the Noise Profile FROM y_mixed ITSELF ---

    # 1A. Transform the ENTIRE Noisy Signal into the Frequency Domain (STFT)
    S_mixed_full = librosa.stft(y_mixed, n_fft=n_fft, hop_length=hop_length)
    if verbose:
        print(f"  Shape of y_mixed: {y_mixed.shape}")
        print(f"  Shape of S_mixed_full (freq_bins, total_time_frames): {S_mixed_full.shape}")

    # 1B. Determine how many STFT frames correspond to noise_duration_sec
    # Each frame covers hop_length samples.
    # Time per frame = hop_length / sr
    # Num frames = noise_duration_sec / (hop_length / sr)
    #            = (noise_duration_sec * sr) / hop_length
    noise_frames_count = int(np.ceil((noise_duration_sec * sr) / hop_length))
    # Ensure we don't try to take more frames than available
    noise_frames_count = min(noise_frames_count, S_mixed_full.shape[1])

    if verbose:
        print(f"  Using first {noise_frames_count} frames for noise profile.")

    if noise_frames_count == 0:
        print("Warning: noise_duration_sec is too short or audio is too short, resulting in 0 frames for noise profile. Denoising might be ineffective.")
        # Handle this case: maybe return original signal or use a tiny default noise profile
        # For now, let's make a dummy noise profile (mostly silent)
        avg_power_noise_profile = np.zeros((S_mixed_full.shape[0], 1)) * 1e-10 # very small power
    else:
        # 1C. Extract the initial frames assumed to be noise
        S_initial_noise_segment = S_mixed_full[:, :noise_frames_count]

        # 1D. Calculate power spectrum of this initial segment
        power_initial_noise_segment = np.abs(S_initial_noise_segment)**2

        # 1E. Average the power across these initial frames for each frequency bin
        avg_power_noise_profile = np.mean(power_initial_noise_segment, axis=1, keepdims=True)

    if verbose:
        print(f"  Shape of avg_power_noise_profile (freq_bins, 1): {avg_power_noise_profile.shape}")


    # --- Step 2: Process the Noisy Signal Frame by Frame (using S_mixed_full) ---

    # 2A. Get the Power and Phase of Each Frame of the ENTIRE Noisy Signal
    magnitude_mixed_full, phase_mixed_full = librosa.magphase(S_mixed_full)
    power_mixed_full = magnitude_mixed_full**2

    # 2B. Perform Spectral Subtraction (vectorized)
    power_cleaned_est = power_mixed_full - subtraction_factor * avg_power_noise_profile

    # 2C. Handle Negative Power Values (Flooring)
    spectral_floor = floor_factor * avg_power_noise_profile
    power_cleaned = np.maximum(power_cleaned_est, spectral_floor)

    # 2D. Convert Cleaned Power Back to Magnitude
    magnitude_cleaned = np.sqrt(power_cleaned)

    # 2E. Recombine with Original Phase
    S_cleaned = magnitude_cleaned * phase_mixed_full


    # --- Step 3: Transform Back to Time Domain (Inverse STFT - ISTFT) ---
    y_cleaned = librosa.istft(S_cleaned, hop_length=hop_length, length=len(y_mixed))

    if verbose:
        print("Spectral Subtraction (Auto Noise Estimation) Complete.")
        print(f"  Shape of y_cleaned: {y_cleaned.shape}")

    return y_cleaned


def main():
    parser = argparse.ArgumentParser(description="Apply spectral subtraction to a noisy audio file or all files in a directory.")
    parser.add_argument('--input', required=True, help='Input noisy audio file path or directory')
    parser.add_argument('--output', required=True, help='Output denoised audio file path or directory')
    parser.add_argument('--noise_duration_sec', type=float, default=0.5, help='Seconds from start to use for noise profile (default: 0.5)')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT window size (default: 2048)')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length for STFT (default: 512)')
    parser.add_argument('--subtraction_factor', type=float, default=3.0, help='Spectral subtraction factor (default: 3.0)')
    parser.add_argument('--floor_factor', type=float, default=0.01, help='Spectral floor factor (default: 0.01)')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    args = parser.parse_args()

    # If input is a directory, process all .wav files in it
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        input_files = [f for f in os.listdir(args.input) if f.lower().endswith('.wav')]
        if not input_files:
            print(f"No .wav files found in directory {args.input}")
            return
        for fname in input_files:
            in_path = os.path.join(args.input, fname)
            out_path = os.path.join(args.output, f"denoised_{fname}")
            y_mixed, sr = librosa.load(in_path, sr=None)
            y_denoised = spectral_subtraction_auto_noise(
                y_mixed, sr,
                noise_duration_sec=args.noise_duration_sec,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                subtraction_factor=args.subtraction_factor,
                floor_factor=args.floor_factor,
                verbose=args.verbose
            )
            sf.write(out_path, y_denoised, sr)
            print(f"Denoised audio saved to {out_path}")
    else:
        y_mixed, sr = librosa.load(args.input, sr=None)
        y_denoised = spectral_subtraction_auto_noise(
            y_mixed, sr,
            noise_duration_sec=args.noise_duration_sec,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            subtraction_factor=args.subtraction_factor,
            floor_factor=args.floor_factor,
            verbose=args.verbose
        )
        sf.write(args.output, y_denoised, sr)
        print(f"Denoised audio saved to {args.output}")

if __name__ == "__main__":
    main()