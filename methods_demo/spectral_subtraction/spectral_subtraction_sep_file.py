import librosa
import numpy as np
import soundfile as sf

def spectral_subtraction_basic(
    y_mixed,          # The noisy audio signal (numpy array)
    y_noise_clip,     # A clip of just noise (numpy array, same sr as y_mixed)
    sr,               # Sampling rate
    n_fft=2048,       # FFT window size
    hop_length=512,   # Hop length for STFT
    subtraction_factor=10.0, # How much of the noise profile to subtract
    floor_factor=0.01, # For spectral flooring to reduce musical noise
    verbose=False      # If True, prints some shapes and info
):
    """
    Performs basic spectral subtraction.

    Args:
        y_mixed (np.array): The input noisy audio signal.
        y_noise_clip (np.array): A short clip of audio containing only noise.
        sr (int): Sampling rate of the audio.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        subtraction_factor (float): Factor by which to subtract the noise estimate.
        floor_factor (float): Factor for the spectral floor. Residual power won't go
                              below floor_factor * avg_noise_power.
        verbose (bool): If True, prints status messages.

    Returns:
        np.array: The denoised audio signal.
    """

    if verbose:
        print("Starting Spectral Subtraction...")
        print(f"  N_FFT: {n_fft}, Hop Length: {hop_length}")
        print(f"  Subtraction Factor: {subtraction_factor}, Floor Factor: {floor_factor}")

    # --- Step 1: Estimate the Noise Profile ---
    # STFT of the noise clip
    S_noise_clip = librosa.stft(y_noise_clip, n_fft=n_fft, hop_length=hop_length)
    # Calculate power spectrum of the noise clip
    power_noise_clip = np.abs(S_noise_clip)**2
    # Average the power across all frames of the noise clip for each frequency bin
    avg_power_noise_profile = np.mean(power_noise_clip, axis=1, keepdims=True) # keepdims for broadcasting

    if verbose:
        print(f"  Shape of y_mixed: {y_mixed.shape}")
        print(f"  Shape of y_noise_clip: {y_noise_clip.shape}")
        print(f"  Shape of S_noise_clip (freq_bins, time_frames): {S_noise_clip.shape}")
        print(f"  Shape of avg_power_noise_profile (freq_bins, 1): {avg_power_noise_profile.shape}")


    # --- Step 2: Process the Noisy Signal Frame by Frame ---

    # 1. Transform the Noisy Signal into the Frequency Domain (STFT)
    S_mixed = librosa.stft(y_mixed, n_fft=n_fft, hop_length=hop_length)

    if verbose:
        print(f"  Shape of S_mixed (freq_bins, time_frames): {S_mixed.shape}")

    # 2. Get the Power and Phase of Each Frame of the Noisy Signal
    magnitude_mixed, phase_mixed = librosa.magphase(S_mixed)
    power_mixed = magnitude_mixed**2

    # Initialize matrix for the cleaned power spectrum
    power_cleaned_est = np.zeros_like(power_mixed)

    # 3. Perform Spectral Subtraction for Each Frame (can be vectorized)
    # The noise profile (avg_power_noise_profile) will be broadcast across the time frames of power_mixed
    power_cleaned_est = power_mixed - subtraction_factor * avg_power_noise_profile

    # 4. Handle Negative Power Values (Flooring)
    # Define the spectral floor. Here, it's a fraction of the average noise power profile.
    # Other flooring strategies exist (e.g., fraction of current frame's power).
    spectral_floor = floor_factor * avg_power_noise_profile

    # Ensure power_cleaned_est doesn't go below the spectral_floor (and thus not below zero)
    power_cleaned = np.maximum(power_cleaned_est, spectral_floor)
    # A simpler alternative (half-wave rectification, more prone to musical noise):
    # power_cleaned = np.maximum(power_cleaned_est, 0)


    # 5. Convert Cleaned Power Back to Magnitude
    magnitude_cleaned = np.sqrt(power_cleaned)

    # 6. Recombine with Original Phase
    # We use the phase from the original noisy signal
    S_cleaned = magnitude_cleaned * phase_mixed


    # --- Step 3: Transform Back to Time Domain (Inverse STFT - ISTFT) ---
    y_cleaned = librosa.istft(S_cleaned, hop_length=hop_length, length=len(y_mixed))
    # 'length' argument helps ensure output matches input length, handling potential edge effects of STFT/ISTFT

    if verbose:
        print("Spectral Subtraction Complete.")
        print(f"  Shape of y_cleaned: {y_cleaned.shape}")

    sf.write("cleaned_audio.wav", y_cleaned, 16000)


  

y_combined, sr_voice = librosa.load('combined_audio.wav', sr=16000)
y_noise, sr_noise = librosa.load('noise.wav', sr=16000)

# Change the length of noise (since it's longer) to match voice
y_noise_truncated = y_noise[:len(y_combined)]
spectral_subtraction_basic(y_combined,y_noise_truncated,16000)