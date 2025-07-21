
import soundfile as sf
import numpy as np
import csv
import os

# Parameters
wav_path = '../useful_sleep_sounds/full_mic.wav'  # Path to your input WAV file
output_csv = 'output_features.csv'  # Path to output CSV file
frame_sec = 1  # Frame size in seconds
chunk_sec = 300  # Chunk size in seconds (5 minutes)

header = ['filename', 'frame_start', 'frame_end', 'energy', 'zcr', 'centroid']
write_header = not os.path.exists(output_csv) or os.stat(output_csv).st_size == 0

# Feature extraction function
def extract_features(frame, sr):
    energy = np.mean(np.abs(frame))
    zcr = ((frame[:-1] * frame[1:]) < 0).sum() / len(frame)
    spectrum = np.abs(np.fft.rfft(frame))
    freqs = np.fft.rfftfreq(len(frame), 1/sr)
    centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
    return [energy, zcr, centroid]

with sf.SoundFile(wav_path, 'r') as f:
    sr = f.samplerate
    frame_len = int(sr * frame_sec)
    chunk_len = int(sr * chunk_sec)
    total_frames = len(f)
    n_chunks = int(np.ceil(total_frames / chunk_len))
    frame_counter = 0
    with open(output_csv, 'a', newline='') as out_f:
        writer = csv.writer(out_f)
        if write_header:
            writer.writerow(header)
        for chunk_idx in range(n_chunks):
            chunk_start = chunk_idx * chunk_len
            chunk_end = min((chunk_idx + 1) * chunk_len, total_frames)
            f.seek(chunk_start)
            chunk_data = f.read(frames=chunk_end - chunk_start, dtype='float32')
            n_frames = int(np.floor(len(chunk_data) / frame_len))
            for i in range(n_frames):
                frame = chunk_data[i*frame_len:(i+1)*frame_len]
                if len(frame) < frame_len:
                    break
                frame_start_sec = frame_counter * frame_sec
                frame_end_sec = frame_start_sec + frame_sec
                features = extract_features(frame, sr)
                writer.writerow([os.path.basename(wav_path), frame_start_sec, frame_end_sec] + features)
                frame_counter += 1
print(f"Appended {frame_counter} frames to {output_csv}")
