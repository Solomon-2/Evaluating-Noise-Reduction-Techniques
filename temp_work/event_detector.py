import librosa
import numpy as np
import soundfile as sf

# ======================================================================
# YOUR SPECTRAL SUBTRACTION FUNCTION (with the fix)
# ======================================================================
def spectral_subtraction_auto_noise(
    y_mixed,
    sr,
    noise_duration_sec=0.5,
    n_fft=2048,
    hop_length=512,
    subtraction_factor=3.0,
    floor_factor=0.01
):
    """
    Performs spectral subtraction, estimating noise from the initial part
    of the input signal.
    """
    S_mixed_full = librosa.stft(y_mixed, n_fft=n_fft, hop_length=hop_length)
    
    noise_frames_count = int(np.ceil((noise_duration_sec * sr) / hop_length))
    noise_frames_count = min(noise_frames_count, S_mixed_full.shape[1])

    if noise_frames_count == 0:
        avg_power_noise_profile = np.zeros((S_mixed_full.shape[0], 1)) * 1e-10
    else:
        S_initial_noise_segment = S_mixed_full[:, :noise_frames_count]
        power_initial_noise_segment = np.abs(S_initial_noise_segment)**2
        avg_power_noise_profile = np.mean(power_initial_noise_segment, axis=1, keepdims=True)

    magnitude_mixed_full, phase_mixed_full = librosa.magphase(S_mixed_full)
    power_mixed_full = magnitude_mixed_full**2

    power_cleaned_est = power_mixed_full - subtraction_factor * avg_power_noise_profile
    spectral_floor = floor_factor * avg_power_noise_profile
    power_cleaned = np.maximum(power_cleaned_est, spectral_floor)

    magnitude_cleaned = np.sqrt(power_cleaned)
    S_cleaned = magnitude_cleaned * phase_mixed_full
    y_cleaned = librosa.istft(S_cleaned, hop_length=hop_length, length=len(y_mixed))

    # ---- THE FIX IS HERE ----
    return y_cleaned
    # -------------------------


# ======================================================================
# APNEA DETECTOR FUNCTION (as provided before)
# ======================================================================
def detect_apneas_from_array(
    y,
    sr,
    frame_length_ms=250,
    hop_length_ms=125,
    energy_thresh_ratio=0.25,
    smoothing_window_s=1.0,
    min_duration_s=10.0
):
    """
    Detects apneic events from an audio signal's energy contour.
    """
    if y is None:
        print("Error inside detector: received empty audio signal (y is None).")
        return []

    frame_length = int(frame_length_ms / 1000 * sr)
    hop_length = int(hop_length_ms / 1000 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    smoothing_window_frames = int(smoothing_window_s / (hop_length / sr))
    if smoothing_window_frames > 0:
        rms_smoothed = np.convolve(rms, np.ones(smoothing_window_frames) / smoothing_window_frames, mode='same')
    else:
        rms_smoothed = rms
        
    median_energy = np.median(rms_smoothed)
    low_energy_threshold = max(1e-8, energy_thresh_ratio * median_energy)
    is_below_threshold = rms_smoothed < low_energy_threshold
    
    detected_events = []
    in_event = False
    start_frame = 0
    frame_duration_s = hop_length / sr
    min_event_frames = int(min_duration_s / frame_duration_s)

    for i, is_low in enumerate(is_below_threshold):
        if is_low and not in_event:
            in_event = True
            start_frame = i
        elif not is_low and in_event:
            in_event = False
            end_frame = i
            if (end_frame - start_frame) >= min_event_frames:
                detected_events.append((start_frame * frame_duration_s, end_frame * frame_duration_s))

    if in_event and (len(is_below_threshold) - start_frame) >= min_event_frames:
        detected_events.append((start_frame * frame_duration_s, len(is_below_threshold) * frame_duration_s))
        
    return detected_events

# ======================================================================
# MAIN EXECUTION BLOCK (what runs when you execute the file)
# ======================================================================
if __name__ == '__main__':
    input_filename = 'event_test_with_apnea.wav'  # Make sure this file exists!
    sr_target = 16000

    try:
        y_noisy, sr = librosa.load(input_filename, sr=sr_target)
        print(f"Loaded '{input_filename}'")
        
        # --- Run Detection on Raw (Noisy) Audio ---
        print("\nAnalyzing RAW audio...")
        raw_events = detect_apneas_from_array(y_noisy, sr)
        if raw_events:
            print(f"Found {len(raw_events)} potential apnea events in raw audio:")
            for start, end in raw_events:
                print(f"  - Start: {start:.2f}s, End: {end:.2f}s, Duration: {end-start:.2f}s")
        else:
            print("No apnea events detected in raw audio.")

        # --- Denoise Audio using your function ---
        print("\nDenoising with spectral subtraction...")
        # This will now correctly receive the audio array
        y_cleaned = spectral_subtraction_auto_noise(y_noisy, sr, subtraction_factor=2.0)
        
        # --- Run Detection on Cleaned Audio ---
        print("\nAnalyzing CLEANED audio...")
        cleaned_events = detect_apneas_from_array(y_cleaned, sr)
        if cleaned_events:
            print(f"Found {len(cleaned_events)} potential apnea events in cleaned audio:")
            for start, end in cleaned_events:
                print(f"  - Start: {start:.2f}s, End: {end:.2f}s, Duration: {end-start:.2f}s")
        else:
            print("No apnea events detected in cleaned audio.")
            
    except FileNotFoundError:
        print(f"\nERROR: File not found at '{input_filename}'")
        print("Please make sure you have a file named 'event_test.wav' in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")