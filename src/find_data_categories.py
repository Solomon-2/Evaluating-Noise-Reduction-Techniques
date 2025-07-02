import pandas as pd
import librosa
import os
import numpy as np

meta = pd.read_csv("../data/raw/ESC-50-MASTER/meta/esc50.csv")

# desired_categories = ['dog','cat','cow','vacuum_cleaner']
# filtered = meta[meta['category'].isin(desired_categories)].copy()

# audio_dir = 'ESC-50-master/audio/'

# durations = []
# for fname in filtered['filename']:
#     path = os.path.join(audio_dir, fname)
#     y, sr = librosa.load(path, sr=16000)
#     file_len = len(y) / sr
#     file_len = round(file_len)
#     durations.append(file_len)


# filtered['duration_sec'] = durations
# print(filtered)

# # Write to a new CSV file
# output_csv = 'filtered_audio_data.csv'
# filtered.to_csv(output_csv, index=False)
# print(f"Filtered data saved to {output_csv}")

def return_file_path(category):
    # Returns a list of file paths for the given category
    filtered = meta[meta['category'].isin(category)].copy()
    
    audio_dir = '../data/raw/ESC-50-master/audio/'
    file_paths = [os.path.join(audio_dir, fname) for fname in filtered['filename']]
    return file_paths
 

