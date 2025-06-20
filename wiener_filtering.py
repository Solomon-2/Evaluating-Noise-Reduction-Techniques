import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    target_sr = 16000
    input_noisy_filename = input("Enter the combined audio file's path: ")

    initial_noise_duration = 0.75
    n_fft_val = 2048
    hop_length_val = 512
    epsilon_val = 1e-10

    try:
        y_combined, sr_comb = librosa.load(input_noisy_filename, sr=target_sr)
        print(f"Successfully loaded '{input_noisy_filename}'")
    except Exception as e:
        print(f"Error loading '{input_noisy_filename}': {e}")
        print("Please ensure the file exists and is a valid audio file.")
        print("If it's an MP3 or other format, you might need ffmpeg installed and on your PATH.")
        exit()

    print(f"Running Wiener filter with initial noise duration: {initial_noise_duration}s")

    y_cleaned_wiener = wiener_filter_basic_auto_noise(
        y_mixed=y_combined,
        sr=target_sr,
        noise_duration_sec=initial_noise_duration,
        n_fft=n_fft_val,
        hop_length=hop_length_val,
        epsilon=epsilon_val,
        verbose=True
    )

    output_cleaned_filename = (
        f"cleaned_wiener_noise_dur{initial_noise_duration}_"
        f"fft{n_fft_val}_hop{hop_length_val}.wav"
    )
    try:
        sf.write(output_cleaned_filename, y_cleaned_wiener, target_sr)
        print(f"Saved Wiener-filtered audio to: '{output_cleaned_filename}'")
    except Exception as e:
        print(f"Error saving audio file: {e}")

    # Visualize Spectrograms
    print("Generating spectrograms...")
    try:
        fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(12, 8))
        S_noisy_db = librosa.amplitude_to_db(np.abs(librosa.stft(y_combined, n_fft=n_fft_val, hop_length=hop_length_val)), ref=np.max)
        librosa.display.specshow(S_noisy_db, y_axis='log', x_axis='time', sr=target_sr, ax=ax[0], hop_length=hop_length_val)
        ax[0].set(title=f'Noisy Input: {input_noisy_filename}')
        ax[0].label_outer()

        S_cleaned_db = librosa.amplitude_to_db(np.abs(librosa.stft(y_cleaned_wiener, n_fft=n_fft_val, hop_length=hop_length_val)), ref=np.max)
        librosa.display.specshow(S_cleaned_db, y_axis='log', x_axis='time', sr=target_sr, ax=ax[1], hop_length=hop_length_val)
        ax[1].set(title=f'Wiener Filtered Output (Noise Est. from first {initial_noise_duration}s)')

        fig.colorbar(ax[0].collections[0], ax=ax[0], format='%+2.0f dB')
        fig.colorbar(ax[1].collections[0], ax=ax[1], format='%+2.0f dB')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error generating spectrograms: {e}. Ensure matplotlib is installed.")