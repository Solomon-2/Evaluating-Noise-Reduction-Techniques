import librosa
import numpy as np
import soundfile as sf 

def spectral_subtraction_auto_noise( # Renamed for clarity
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

    sf.write("sleep_demo/mono_cleaned_audio.wav", y_cleaned, 16000)


  

y_combined, sr_voice = librosa.load('sleep_demo/sleep_test_1.wav', sr=16000)

spectral_subtraction_auto_noise(y_combined,16000)