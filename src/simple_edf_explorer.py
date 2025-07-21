import mne
import soundfile as sf
import os
import numpy as np

edf_path = "../sleep_audio_db/no_1.edf"
output_dir = "../useful_sleep_sounds/"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "full_mic.wav")

print("loading only the 'Mic' channel header...")
raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

if "Mic" not in raw.ch_names:
    print("\n❌ 'Mic' channel not found!")
    exit()

mic_idx = raw.ch_names.index("Mic")
sfreq = int(raw.info["sfreq"])
n_samples = raw.n_times

print(f"\n=== EDF FILE INFO ===")
print(f"num channels: {len(raw.ch_names)}")
print(f"duration: {raw.times[-1]/60:.1f} min")
print(f"sampling rate: {sfreq}")
print(f"mic channel index: {mic_idx}")

chunk_samples = 10 * sfreq  # e.g., 10 seconds at a time (~ so ~10MB/float)
start = 0
print(f"\nwriting 'Mic' channel to {output_file} in {chunk_samples} sample chunks...")

with sf.SoundFile(output_file, mode='w', samplerate=sfreq, channels=1, subtype='FLOAT') as f:
    while start < n_samples:
        stop = min(start + chunk_samples, n_samples)
        # "start:stop" gives us only this slice for the mic channel
        chunk = raw.get_data(picks=[mic_idx], start=start, stop=stop)[0]
        f.write(chunk)
        print(f"  wrote samples {start:,}–{stop-1:,}")
        start = stop

print(f"\n=== SUMMARY ===")
print(f"output file: {output_file}")
print(f"done!")