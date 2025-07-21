import argparse
import numpy as np
import scipy.io.wavfile as wavfile
import warnings

# Suppress warnings from reading potentially corrupt WAV files
warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)

def calculate_rms_energy(audio_data):
    """Calculates the Root Mean Square (RMS) energy of an audio segment."""
    return np.sqrt(np.mean(audio_data**2))

def detect_apnea_events(
    filepath,
    window_size_s=0.5,
    min_apnea_duration_s=10.0,
    energy_threshold_factor=0.2,
    silence_threshold_factor=0.05
):
    """
    Detects apnea events in a WAV file using a signal processing approach.

    Args:
        filepath (str): Path to the input WAV file.
        window_size_s (float): The size of the analysis window in seconds.
        min_apnea_duration_s (float): The minimum duration for an event to be
                                     classified as apnea.
        energy_threshold_factor (float): Factor to multiply with the reference
                                         energy to get the apnea threshold.
        silence_threshold_factor (float): Factor to identify absolute silence
                                          for calculating the dynamic threshold.

    Returns:
        list: A list of tuples, where each tuple contains the start and end
              time (in seconds) of a detected apnea event.
    """
    print(f"Loading audio file: {filepath}...")
    try:
        sample_rate, data = wavfile.read(filepath)
    except Exception as e:
        print(f"Error: Could not read WAV file. Reason: {e}")
        return []

    # --- 1. Preprocessing ---
    # Ensure data is a float array in the range [-1, 1]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max

    # Convert to mono if stereo
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    print(f"Audio loaded successfully. Sample Rate: {sample_rate}, Duration: {len(data)/sample_rate:.2f}s")

    # --- 2. Calculate Energy in Sliding Windows ---
    window_size_samples = int(window_size_s * sample_rate)
    num_windows = len(data) // window_size_samples

    window_energies = np.array([
        calculate_rms_energy(data[i*window_size_samples:(i+1)*window_size_samples])
        for i in range(num_windows)
    ])

    # --- 3. Dynamic Threshold Calculation ---
    # Use a fraction of the median energy of non-silent parts as a reference
    median_energy = np.median(window_energies)
    silence_threshold = median_energy * silence_threshold_factor
    
    # Filter out silent parts to get a better reference for breathing energy
    active_energies = window_energies[window_energies > silence_threshold]
    
    if len(active_energies) == 0:
        print("Warning: The audio file appears to be completely silent.")
        return []
        
    reference_energy = np.median(active_energies)
    apnea_energy_threshold = reference_energy * energy_threshold_factor

    print(f"Analyzing {num_windows} windows of {window_size_s}s each...")
    print(f"Dynamic Apnea Energy Threshold: {apnea_energy_threshold:.4f}")

    # --- 4. Identify Potential Apnea Windows ---
    is_below_threshold = window_energies < apnea_energy_threshold

    # --- 5. Consolidate Events ---
    apnea_events = []
    in_event = False
    event_start_index = 0
    min_windows_for_apnea = int(min_apnea_duration_s / window_size_s)

    for i, is_low_energy in enumerate(is_below_threshold):
        if is_low_energy and not in_event:
            in_event = True
            event_start_index = i
        elif not is_low_energy and in_event:
            in_event = False
            event_duration_windows = i - event_start_index
            if event_duration_windows >= min_windows_for_apnea:
                start_time = event_start_index * window_size_s
                end_time = i * window_size_s
                apnea_events.append((round(start_time, 2), round(end_time, 2)))

    # Check if the file ends during an apnea event
    if in_event:
        event_duration_windows = num_windows - event_start_index
        if event_duration_windows >= min_windows_for_apnea:
            start_time = event_start_index * window_size_s
            end_time = num_windows * window_size_s
            apnea_events.append((round(start_time, 2), round(end_time, 2)))

    return apnea_events

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect apnea events in a WAV file using signal processing.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the input WAV file."
    )
    args = parser.parse_args()

    detected_events = detect_apnea_events(args.filepath)

    if detected_events:
        print(f"\n--- Detected {len(detected_events)} Apnea Events ---")
        for start, end in detected_events:
            duration = end - start
            print(f"  - Start: {start:>5.2f}s, End: {end:>5.2f}s, Duration: {duration:>4.2f}s")
    else:
        print("\n--- No Apnea Events Detected ---")