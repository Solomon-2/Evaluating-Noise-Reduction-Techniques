import mne
import numpy as np
import soundfile as sf
import os

print("Loading EDF file (header only, no data)...")
raw = mne.io.read_raw_edf("../data/sleep_data.edf", preload=False)

print(f"\n=== EDF FILE INFO ===")
print(f"Number of channels: {len(raw.ch_names)}")
print(f"Duration: {raw.times[-1]:.1f} seconds ({raw.times[-1]/60:.1f} minutes)")
print(f"Sampling rate: {raw.info['sfreq']} Hz")

print(f"\n=== ALL CHANNELS ===")
for i, channel_name in enumerate(raw.ch_names, 1):
    print(f"{i:2d}. {channel_name}")

# Find the Mic channel
if "Mic" in raw.ch_names:
    print(f"\n=== EXTRACTING FROM 'Mic' CHANNEL ===")
    
    # Configuration
    segment_duration = 30  # seconds per file
    total_duration = raw.times[-1]
    output_dir = "../tests/raw/"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate number of segments
    num_segments = int(total_duration // segment_duration)
    print(f"Total duration: {total_duration:.1f} seconds")
    print(f"Segment duration: {segment_duration} seconds")
    print(f"Number of complete {segment_duration}-second segments: {num_segments}")
    
    # Process each segment
    for segment_num in range(num_segments):
        start_time = segment_num * segment_duration
        end_time = start_time + segment_duration
        
        print(f"\nProcessing segment {segment_num + 1}/{num_segments} (time: {start_time}-{end_time}s)...")
        
        try:
            # Extract the segment
            mic_segment = raw.copy().pick_channels(['Mic']).crop(tmin=start_time, tmax=end_time)
            mic_segment.load_data()
            
            # Get the data array
            data = mic_segment.get_data()[0]
            
            # Save as WAV file
            output_file = os.path.join(output_dir, f"file_{segment_num + 1:03d}.wav")
            sf.write(output_file, data, int(mic_segment.info['sfreq']))
            
            print(f"  ✓ Saved: {output_file} ({len(data)} samples)")
            
        except Exception as e:
            print(f"  ❌ Error processing segment {segment_num + 1}: {e}")
    
    # Handle remaining data (if any)
    remaining_time = total_duration % segment_duration
    if remaining_time > 5:  # Only save if more than 5 seconds remaining
        start_time = num_segments * segment_duration
        end_time = total_duration
        
        print(f"\nProcessing final segment ({remaining_time:.1f}s): {start_time}-{end_time}s...")
        
        try:
            mic_segment = raw.copy().pick_channels(['Mic']).crop(tmin=start_time, tmax=end_time)
            mic_segment.load_data()
            
            data = mic_segment.get_data()[0]
            
            output_file = os.path.join(output_dir, f"file_{num_segments + 1:03d}.wav")
            sf.write(output_file, data, int(mic_segment.info['sfreq']))
            
            print(f"  ✓ Saved final segment: {output_file} ({len(data)} samples)")
            
        except Exception as e:
            print(f"  ❌ Error processing final segment: {e}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total segments created: {len(os.listdir(output_dir)) if os.path.exists(output_dir) else 0}")
    print(f"Output directory: {output_dir}")
    
else:
    print("\n❌ 'Mic' channel not found!")

print(f"\nDone!")