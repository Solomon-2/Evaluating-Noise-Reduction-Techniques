import librosa
import numpy as np

def detect_apnea(audio, sr, threshold=0.01, min_silence=10):
    rms = librosa.feature.rms(r=audio)[0]
    silence = rms<threshold

    silence_duration = librosa.frames_to_time(np.diff(np.where(np.diff(silence.astype(int)) !=0)[0]), sr=sr)

    return list_of_events