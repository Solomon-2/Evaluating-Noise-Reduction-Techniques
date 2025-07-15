import pyedflib
import numpy as np
from scipy.io.wavfile import write as write_wav

def extract_audio_from_edf(edf_path, output_wav_path, mic_channel_label='Mic'):
    # Open the EDF file
    edf = pyedflib.EdfReader(edf_path)
    
    # Find the index of the channel labeled 'Mic'
    channel_labels = edf.getSignalLabels()
    if mic_channel_label not in channel_labels:
        raise ValueError(f"Channel '{mic_channel_label}' not found in EDF file. Available channels: {channel_labels}")
    
    mic_channel_index = channel_labels.index(mic_channel_label)
    
    # Get the sample rate of the Mic channel
    sample_rate = edf.getSampleFrequency(mic_channel_index)
    
    # Read the entire signal data from the Mic channel
    mic_data = edf.readSignal(mic_channel_index)
    
    # Normalize data to int16 range if needed
    if mic_data.dtype != np.int16:
        mic_data = mic_data / np.max(np.abs(mic_data))  # Normalize to -1.0 to 1.0
        mic_data = (mic_data * 32767).astype(np.int16)  # Scale to int16
    
    # Write to WAV
    write_wav(output_wav_path, int(sample_rate), mic_data)
    
    print(f"Audio extracted and saved to '{output_wav_path}'")
    
    # Close the EDF file
    edf.close()

# Example usage
extract_audio_from_edf('../sleep_audio_db/patient_2_1.edf', '../useful_sleep_sounds/output_audio.wav')
