import librosa
import numpy as np
import soundfile as sf # For saving audio
import matplotlib.pyplot as plt

def spectral_subtraction(noisy_signal, noise_clip, sr, n_fft=2048, hop_length=512,
                         subtraction_factor=1.0, spectral_floor_factor=0.02,
                         verbose=False):
    """
    Performs spectral subtraction to reduce noise from an audio signal.

    Args:
        noisy_signal (np.array): The input audio signal with noise.
        noise_clip (np.array): A short clip of audio containing only noise.
        sr (int): Sampling rate of the audio.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        subtraction_factor (float): Factor by which to subtract the noise estimate.
                                    Values > 1 can lead to more aggressive subtraction.
        spectral_floor_factor (float): Factor for the spectral floor. The denoised power
                                       in a bin won't go below this factor times the
                                       original noisy power in that bin. Helps reduce musical noise.
                                       Should be small (e.g., 0.01 to 0.1).
        verbose (bool): If True, plots spectrograms.

    Returns:
        np.array: The denoised audio signal.
    """
    win_length = n_fft # Window length often same as n_fft

    # 1. STFT of the noise clip
    S_noise = librosa.stft(noise_clip, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    # Calculate magnitude and phase (though phase of noise isn't directly used for profile)
    mag_noise, _ = librosa.magphase(S_noise)
    # Calculate average noise power spectrum (noise profile)
    # Averaging the power (magnitude squared) across the frames of the noise clip
    avg_power_noise = np.mean(mag_noise**2, axis=1, keepdims=True)

    # 2. STFT of the noisy signal
    S_noisy = librosa.stft(noisy_signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag_noisy, phase_noisy = librosa.magphase(S_noisy)
    power_noisy = mag_noisy**2

    # 3. Perform subtraction
    # The core subtraction: noisy power - scaled average noise power
    power_denoised_est = power_noisy - subtraction_factor * avg_power_noise

    # 4. Apply spectral floor
    # The floor is based on the original noisy signal's power in each bin
    spectral_floor = (spectral_floor_factor**2) * power_noisy # Floor is in power domain
    # Ensure the estimated denoised power doesn't go below the floor (and also not below zero)
    power_denoised = np.maximum(power_denoised_est, spectral_floor)
    # An alternative simpler floor would be: power_denoised = np.maximum(power_denoised_est, 0)
    # but the spectral_floor_factor method is generally better for reducing musical noise.

    # 5. Convert back to magnitude and combine with original phase
    mag_denoised = np.sqrt(power_denoised)
    S_denoised = mag_denoised * phase_noisy

    # 6. Inverse STFT to get the time-domain signal
    y_denoised = librosa.istft(S_denoised, hop_length=hop_length, win_length=win_length, length=len(noisy_signal))

    if verbose:
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        librosa.display.specshow(librosa.amplitude_to_db(mag_noisy, ref=np.max),
                                 sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
        plt.title('Noisy Signal Spectrogram')
        plt.colorbar(format='%+2.0f dB')

        plt.subplot(3, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(np.sqrt(avg_power_noise), ref=np.max),
                                 sr=sr, hop_length=hop_length, y_axis='log') # No x-axis, it's an average
        plt.title('Average Noise Magnitude Profile')
        plt.colorbar(format='%+2.0f dB')


        plt.subplot(3, 1, 3)
        librosa.display.specshow(librosa.amplitude_to_db(mag_denoised, ref=np.max),
                                 sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
        plt.title('Denoised Signal Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.show()

    return y_denoised

# --- Example Usage ---
if __name__ == '__main__':
    # Create a dummy noisy signal and a noise clip for demonstration
    sr = 22050
    duration_signal = 5  # seconds
    duration_noise_clip = 1 # seconds

    # Generate a simple clean signal (e.g., a sine wave)
    freq_clean = 440  # A4 note
    t_signal = np.linspace(0, duration_signal, int(sr * duration_signal), endpoint=False)
    clean_signal = 0.5 * np.sin(2 * np.pi * freq_clean * t_signal)

    # Generate some noise
    noise_amp = 0.3
    # Noise for the main signal
    random_noise_main = noise_amp * np.random.randn(len(clean_signal))
    # A separate clip assumed to be only noise
    t_noise_clip = np.linspace(0, duration_noise_clip, int(sr*duration_noise_clip), endpoint=False)
    noise_only_clip = noise_amp * np.random.randn(len(t_noise_clip)) # Should ideally be similar characteristic to main noise

    # Create noisy signal
    noisy_signal_demo = clean_signal + random_noise_main
    # Ensure noise_only_clip is representative (here it's just more random noise)

    print("Processing... (this might take a moment)")
    # Apply spectral subtraction
    denoised_signal = spectral_subtraction(
        noisy_signal_demo,
        noise_only_clip,
        sr,
        n_fft=2048,
        hop_length=512,
        subtraction_factor=1.5,      # Try tuning this
        spectral_floor_factor=0.05, # Try tuning this
        verbose=True
    )

    # Save the results (optional, requires soundfile library: pip install soundfile)
    try:
        sf.write('original_noisy.wav', noisy_signal_demo, sr)
        sf.write('noise_profile_clip.wav', noise_only_clip, sr)
        sf.write('denoised_spectral_subtraction.wav', denoised_signal, sr)
        print("Saved audio files: original_noisy.wav, noise_profile_clip.wav, denoised_spectral_subtraction.wav")
        print("You can listen to these to hear the effect.")
    except Exception as e:
        print(f"Could not save audio files (is soundfile installed?): {e}")

    print("Done.")