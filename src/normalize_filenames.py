"""
Utility functions for normalizing filenames between ground truth and detected event CSVs.
"""
import re

def normalize_filename(filename):
    """
    Normalize filename to extract the core identifier for matching.
    
    Examples:
    - 'raw_1_20s.wav' -> '1_20s'
    - 'denoised_mixed_raw_2_62s.wav' -> '2_62s' 
    - 'mixed_raw_3_30s.wav' -> '3_30s'
    """
    # Extract base filename without path
    base = re.search(r'[^/\\]*\.wav$', filename)
    if base:
        base_name = base.group(0).replace('.wav', '')
    else:
        base_name = filename.replace('.wav', '')
    
    # Extract the core pattern: number_duration (e.g., "1_20s", "2_62s")
    # This handles various prefixes like "raw_", "denoised_mixed_raw_", "mixed_raw_"
    pattern = re.search(r'(\d+_\d+s)', base_name)
    if pattern:
        return pattern.group(1)
    
    # Fallback: return the base filename
    return base_name

def create_filename_mapping(csv_path):
    """
    Create a mapping from normalized filenames to original filenames.
    Useful for debugging filename matching issues.
    """
    import csv
    from collections import defaultdict
    
    mapping = defaultdict(list)
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            original = row['filename']
            normalized = normalize_filename(original)
            mapping[normalized].append(original)
    
    return dict(mapping)

if __name__ == "__main__":
    # Test the normalization function
    test_files = [
        'raw_1_20s.wav',
        'raw_2_62s.wav', 
        'raw_3_30s.wav',
        'denoised_mixed_raw_1_20s.wav',
        'denoised_mixed_raw_2_62s.wav',
        'denoised_mixed_raw_3_30s.wav',
        '../tests/raw/raw_1_20s.wav',
        '../tests/denoised/deepfilternet/denoised_mixed_raw_2_62s.wav'
    ]
    
    print("Filename normalization test:")
    for filename in test_files:
        normalized = normalize_filename(filename)
        print(f"'{filename}' -> '{normalized}'")
