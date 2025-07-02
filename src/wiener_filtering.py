import librosa
import numpy as np
import soundfile as sf

def wiener_filter_basic_auto_noise(
    y_mixed,
    sr,
    noise_duration_sec=0.5,
    n_fft=2048,
    hop_length=512,
    epsilon=1e-10,
    verbose=False
):
    """
    Performs basic frequency-domain Wiener filtering, estimating noise
    from the initial part of the input signal.
    """
    if verbose:
        print("Starting Wiener Filter (Auto Noise Estimation)...")
        print(f"  Noise Profile Estimation from first {noise_duration_sec} seconds.")
        print(f"  N_FFT: {n_fft}, Hop Length: {hop_length}, Epsilon: {epsilon}")

    # Estimate the Noise Power Spectral Density (PSD)
    S_mixed_full = librosa.stft(y_mixed, n_fft=n_fft, hop_length=hop_length)
    if verbose:
        print(f"  Shape of y_mixed: {y_mixed.shape}")
        print(f"  Shape of S_mixed_full (freq_bins, total_time_frames): {S_mixed_full.shape}")

    noise_frames_count = int(np.ceil((noise_duration_sec * sr) / hop_length))
    noise_frames_count = min(noise_frames_count, S_mixed_full.shape[1])

    if verbose:
        print(f"  Using first {noise_frames_count} STFT frames for noise PSD estimation.")

    if noise_frames_count <= 0:
        print("Warning: noise_duration_sec/audio length too short for noise estimation. Using a tiny default noise PSD.")
        avg_power_noise_profile = np.full((S_mixed_full.shape[0], 1), epsilon)
    else:
        S_initial_noise_segment = S_mixed_full[:, :noise_frames_count]
        power_initial_noise_segment = np.abs(S_initial_noise_segment)**2
        avg_power_noise_profile = np.mean(power_initial_noise_segment, axis=1, keepdims=True)
        avg_power_noise_profile = np.maximum(avg_power_noise_profile, epsilon)

    if verbose:
        print(f"  Shape of avg_power_noise_profile (Noise PSD P_nn) (freq_bins, 1): {avg_power_noise_profile.shape}")

    # Apply Wiener Filter to the Full Signal
    magnitude_mixed_full, phase_mixed_full = librosa.magphase(S_mixed_full)
    power_mixed_full = magnitude_mixed_full**2

    # Estimate the Power Spectral Density of the Clean Signal
    P_ss_estimated = power_mixed_full - avg_power_noise_profile
    P_ss_estimated = np.maximum(P_ss_estimated, 0)

    # Calculate the Wiener Filter Gain
    wiener_gain = P_ss_estimated / (P_ss_estimated + avg_power_noise_profile + epsilon)

    # Apply the Wiener Gain to the Magnitude
    magnitude_cleaned = wiener_gain * magnitude_mixed_full

    # Recombine with the Original Phase
    S_cleaned = magnitude_cleaned * phase_mixed_full

    # Transform Back to Time Domain
    y_cleaned = librosa.istft(S_cleaned, hop_length=hop_length, length=len(y_mixed))

    if verbose:
        print("Wiener Filter (Auto Noise Estimation) Complete.")
        print(f"  Shape of y_cleaned: {y_cleaned.shape}")

    return y_cleaned


def main(input_noisy_filename, output_cleaned_filename,
         target_sr=16000, initial_noise_duration=0.75, n_fft_val=2048, hop_length_val=512, epsilon_val=1e-10, verbose=True):
    try:
        y_combined, sr_comb = librosa.load(input_noisy_filename, sr=target_sr)
        if verbose:
            print(f"Successfully loaded '{input_noisy_filename}'")
    except Exception as e:
        print(f"Error loading '{input_noisy_filename}': {e}")
        print("Please ensure the file exists and is a valid audio file.")
        print("If it's an MP3 or other format, you might need ffmpeg installed and on your PATH.")
        return

    if verbose:
        print(f"Running Wiener filter with initial noise duration: {initial_noise_duration}s")

    y_cleaned_wiener = wiener_filter_basic_auto_noise(
        y_mixed=y_combined,
        sr=target_sr,
        noise_duration_sec=initial_noise_duration,
        n_fft=n_fft_val,
        hop_length=hop_length_val,
        epsilon=epsilon_val,
        verbose=verbose
    )

    try:
        sf.write(output_cleaned_filename, y_cleaned_wiener, target_sr)
        if verbose:
            print(f"Saved Wiener-filtered audio to: '{output_cleaned_filename}'")
    except Exception as e:
        print(f"Error saving audio file: {e}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Apply Wiener filtering to denoise an audio file.")
    parser.add_argument('--input', required=True, help='Input noisy audio file path')
    parser.add_argument('--output', required=True, help='Output denoised audio file path')
    parser.add_argument('--noise_duration', type=float, default=0.75, help='Initial noise duration in seconds (default: 0.75)')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT window size (default: 2048)')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length for STFT (default: 512)')
    parser.add_argument('--epsilon', type=float, default=1e-10, help='Small value to avoid division by zero (default: 1e-10)')
    parser.add_argument('--sr', type=int, default=16000, help='Target sample rate (default: 16000)')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')
    args = parser.parse_args()
    main(
        input_noisy_filename=args.input,
        output_cleaned_filename=args.output,
        target_sr=args.sr,
        initial_noise_duration=args.noise_duration,
        n_fft_val=args.n_fft,
        hop_length_val=args.hop_length,
        epsilon_val=args.epsilon,
        verbose=args.verbose
    )