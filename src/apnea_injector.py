import numpy as np
import soundfile as sf

# Load audio
audio, sr = sf.read('../data/raw/kaggle_sleep/2/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files/101_1b1_Pr_sc_Meditron.wav')

# Parameters
apnea_duration_sec = 10
apnea_samples = int(apnea_duration_sec * sr)
event_starts = [sr*8]  # Example: apnea at 30s and 90s

# Inject silence
for start in event_starts:
    end = min(start + apnea_samples, len(audio))
    audio[start:end] = 0

# Save output
sf.write('output_with_apnea.wav', audio, sr)