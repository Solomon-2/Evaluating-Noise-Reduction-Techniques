import soundfile as sf
import numpy as np
import joblib
import os
import pandas as pd

# Parameters
wav_path = '../useful_sleep_sounds/full_mic.wav'  # Path to your input WAV file
model_path = '../models/apnea_rf_model.joblib'  # Path to trained model
frame_sec = 1  # Frame size in seconds (must match training)
chunk_sec = 300  # Chunk size in seconds (5 minutes)

# Load model
clf = joblib.load(model_path)

def extract_features(frame, sr):
    energy = np.mean(np.abs(frame))
    zcr = ((frame[:-1] * frame[1:]) < 0).sum() / len(frame)
    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(len(frame), 1/sr)
    centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
    return [energy, zcr, centroid]

# Process WAV in chunks
def detect_apnea_events(wav_path, clf, frame_sec=1, chunk_sec=300):
    events = []
    ongoing_event_start = None  # Track event across chunks
    with sf.SoundFile(wav_path, 'r') as f:
        sr = f.samplerate
        total_frames = len(f)
        frame_len = int(sr * frame_sec)
        chunk_len = int(sr * chunk_sec)
        n_chunks = int(np.ceil(total_frames / chunk_len))
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_len
            chunk_end = min((chunk_idx + 1) * chunk_len, total_frames)
            f.seek(chunk_start)
            chunk_data = f.read(frames=chunk_end - chunk_start, dtype='float32')
            n_frames = int(np.floor(len(chunk_data) / frame_len))
            features = []
            frame_times = []
            for i in range(n_frames):
                frame = chunk_data[i*frame_len:(i+1)*frame_len]
                if len(frame) < frame_len:
                    break
                frame_start = chunk_start / sr + i * frame_sec
                frame_end = frame_start + frame_sec
                features.append(extract_features(frame, sr))
                frame_times.append((frame_start, frame_end))
            if features:
                X = pd.DataFrame(features, columns=['energy', 'zcr', 'centroid'])
                y_pred = clf.predict(X)
                # Group consecutive apnea frames, tracking across chunks
                for j, label in enumerate(y_pred):
                    if label == 1 and ongoing_event_start is None:
                        ongoing_event_start = frame_times[j][0]
                    if label == 0 and ongoing_event_start is not None:
                        events.append((ongoing_event_start, frame_times[j-1][1]))
                        ongoing_event_start = None
                # If event continues at end of chunk, keep ongoing_event_start
        # After all chunks, if event is still ongoing, close it
        if ongoing_event_start is not None:
            events.append((ongoing_event_start, frame_times[-1][1]))
    return events

apnea_events = detect_apnea_events(wav_path, clf, frame_sec, chunk_sec)

print("Detected apnea events (start, end in seconds):")
for start, end in apnea_events:
    print(f"{start:.2f} - {end:.2f}")
