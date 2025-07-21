import mne
import numpy as np

# Load EDF file WITHOUT preloading all data into memory
print("Loading EDF file header (without preloading data)...")
raw = mne.io.read_raw_edf("../data/sleep_data.edf", preload=False)

# Display basic information
print(f"\nFile info:")
print(f"Channels: {len(raw.ch_names)}")
print(f"Channel names: {raw.ch_names}")
print(f"Sampling frequency: {raw.info['sfreq']} Hz")
print(f"Duration: {raw.times[-1]:.1f} seconds ({raw.times[-1]/3600:.1f} hours)")
print(f"Total samples per channel: {len(raw.times)}")
print(f"Estimated memory if fully loaded: {len(raw.ch_names) * len(raw.times) * 8 / (1024**3):.1f} GB")

# Show channel types
print(f"\nChannel types:")
for ch_name, ch_type in zip(raw.ch_names, raw.get_channel_types()):
    print(f"  {ch_name}: {ch_type}")

# Load only a small segment to test (first 30 seconds)
print(f"\nLoading first 30 seconds of data...")
start_time = 0  # seconds
end_time = 30   # seconds
raw_segment = raw.copy().crop(tmin=start_time, tmax=end_time)
raw_segment.load_data()  # Now safe to load this small segment

print(f"Successfully loaded {end_time-start_time} seconds of data")
print(f"Data shape: {raw_segment.get_data().shape}")  # (channels, samples)