#!/usr/bin/env python3
"""
EDF to WAV Audio Extraction Script

Extracts audio from EDF files in patient folders and saves as WAV files.
Uses the same logic as parallel_feature_extraction.ipynb for consistency.

Usage:
    python edf_to_wav_converter.py --input_dir data/sleep_data --output_dir audio_data
    
Output Structure:
    audio_data/
    ├── patient_01_wav/
    │   ├── patient_01_edf01.wav
    │   ├── patient_01_edf02.wav
    │   └── ...
    ├── patient_02_wav/
    └── ...
"""

import os
import sys
import argparse
import numpy as np
import librosa
import mne
import soundfile as sf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration matching parallel_feature_extraction.ipynb
AUDIO_CHANNEL = 'Mic'
TARGET_SAMPLE_RATE = 16000  # 16kHz for consistency with training data
CHUNK_DURATION = 300.0      # Process in 5-minute chunks to avoid memory issues

def extract_audio_from_edf(edf_path, output_path, target_sr=16000, chunk_duration=300.0):
    """
    Extract audio from a single EDF file and save as WAV.
    
    Args:
        edf_path (str): Path to input EDF file
        output_path (str): Path to output WAV file
        target_sr (int): Target sample rate for output
        chunk_duration (float): Process in chunks of this duration (seconds)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"    Processing: {os.path.basename(edf_path)}")
        
        # Load EDF file
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        
        # Check if audio channel exists
        if AUDIO_CHANNEL not in raw.ch_names:
            print(f"      WARNING: No {AUDIO_CHANNEL} channel found, skipping")
            return False
        
        # Select only the audio channel
        raw.pick_channels([AUDIO_CHANNEL])
        original_sample_rate = int(raw.info['sfreq'])
        duration_minutes = raw.n_times / original_sample_rate / 60
        
        print(f"      Duration: {duration_minutes:.1f} min, SR: {original_sample_rate} Hz -> {target_sr} Hz")
        
        # Process in chunks to avoid memory issues
        chunk_samples = int(chunk_duration * original_sample_rate)
        total_samples = raw.n_times
        
        all_audio_chunks = []
        
        for chunk_start in range(0, total_samples, chunk_samples):
            chunk_end = min(chunk_start + chunk_samples, total_samples)
            
            # Load audio chunk
            audio_chunk, _ = raw[:, chunk_start:chunk_end]
            audio_chunk = audio_chunk.flatten()
            
            # Downsample to target sample rate
            if original_sample_rate != target_sr:
                audio_chunk = librosa.resample(
                    audio_chunk, 
                    orig_sr=original_sample_rate, 
                    target_sr=target_sr
                )
            
            all_audio_chunks.append(audio_chunk)
        
        # Concatenate all chunks
        final_audio = np.concatenate(all_audio_chunks)
        
        # Save as WAV file
        sf.write(output_path, final_audio, target_sr)
        
        # Verify output
        output_duration = len(final_audio) / target_sr / 60
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"      SUCCESS: Saved: {output_duration:.1f} min, {file_size_mb:.1f} MB")
        
        # Clean up memory
        del raw, all_audio_chunks, final_audio
        
        return True
        
    except Exception as e:
        print(f"      ERROR: Failed: {e}")
        return False

def process_patient_folder(patient_dir, output_base_dir, patient_id):
    """
    Process all EDF files in a patient folder.
    
    Args:
        patient_dir (str): Path to patient directory
        output_base_dir (str): Base output directory
        patient_id (str): Patient ID (e.g., 'patient_01')
    
    Returns:
        dict: Processing results
    """
    print(f"\\nProcessing {patient_id}")
    
    # Create output directory for this patient
    patient_output_dir = os.path.join(output_base_dir, f"{patient_id}_wav")
    os.makedirs(patient_output_dir, exist_ok=True)
    
    # Find all EDF files in patient directory
    try:
        edf_files = sorted([f for f in os.listdir(patient_dir) if f.endswith('.edf')])
    except Exception as e:
        print(f"  ERROR: Cannot list directory: {e}")
        return {'success': False, 'files_processed': 0, 'error': str(e)}
    
    if not edf_files:
        print(f"  WARNING: No EDF files found")
        return {'success': False, 'files_processed': 0, 'error': 'No EDF files found'}
    
    print(f"  Found {len(edf_files)} EDF files")
    
    # Process each EDF file
    results = {
        'success': True,
        'files_processed': 0,
        'files_failed': 0,
        'total_files': len(edf_files),
        'output_dir': patient_output_dir
    }
    
    for edf_idx, edf_file in enumerate(edf_files, 1):
        # Create output filename
        base_name = os.path.splitext(edf_file)[0]
        output_filename = f"{patient_id}_edf{edf_idx:02d}_{base_name}.wav"
        output_path = os.path.join(patient_output_dir, output_filename)
        
        # Skip if already exists
        if os.path.exists(output_path):
            print(f"    Already exists: {output_filename}")
            results['files_processed'] += 1
            continue
        
        # Extract audio from this EDF
        edf_path = os.path.join(patient_dir, edf_file)
        success = extract_audio_from_edf(edf_path, output_path, TARGET_SAMPLE_RATE, CHUNK_DURATION)
        
        if success:
            results['files_processed'] += 1
        else:
            results['files_failed'] += 1
    
    print(f"  SUCCESS: {patient_id}: {results['files_processed']}/{results['total_files']} files processed")
    
    if results['files_failed'] > 0:
        print(f"  WARNING: {results['files_failed']} files failed")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Extract audio from EDF files in patient folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all patients in data/sleep_data
  python edf_to_wav_converter.py --input_dir data/sleep_data --output_dir audio_data
  
  # Process specific patients only
  python edf_to_wav_converter.py --input_dir data/sleep_data --output_dir audio_data --patients patient_01 patient_02
  
  # Test with single patient
  python edf_to_wav_converter.py --input_dir data/sleep_data --output_dir test_audio --patients patient_01
        """
    )
    
    parser.add_argument(
        '--input_dir', 
        required=True,
        help='Input directory containing patient folders (e.g., data/sleep_data)'
    )
    
    parser.add_argument(
        '--output_dir', 
        required=True,
        help='Output directory for WAV files (e.g., audio_data)'
    )
    
    parser.add_argument(
        '--patients',
        nargs='*',
        help='Specific patients to process (e.g., patient_01 patient_02). If not specified, processes all patients.'
    )
    
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=TARGET_SAMPLE_RATE,
        help=f'Target sample rate (default: {TARGET_SAMPLE_RATE} Hz)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"ERROR: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("EDF TO WAV CONVERTER")
    print("=" * 50)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target sample rate: {args.sample_rate} Hz")
    print(f"Audio channel: {AUDIO_CHANNEL}")
    
    # Find patient directories
    if args.patients:
        # Process specified patients only
        patient_dirs = []
        for patient_id in args.patients:
            patient_path = os.path.join(args.input_dir, patient_id)
            if os.path.exists(patient_path):
                patient_dirs.append((patient_path, patient_id))
            else:
                print(f"WARNING: Patient directory not found: {patient_path}")
    else:
        # Process all patient directories
        try:
            all_items = os.listdir(args.input_dir)
            patient_dirs = []
            for item in all_items:
                item_path = os.path.join(args.input_dir, item)
                if os.path.isdir(item_path) and item.startswith('patient_'):
                    patient_dirs.append((item_path, item))
            patient_dirs.sort(key=lambda x: x[1])  # Sort by patient ID
        except Exception as e:
            print(f"ERROR: Cannot list input directory: {e}")
            sys.exit(1)
    
    if not patient_dirs:
        print("ERROR: No patient directories found")
        sys.exit(1)
    
    print(f"\\nFound {len(patient_dirs)} patient directories")
    for _, patient_id in patient_dirs:
        print(f"  - {patient_id}")
    
    # Process each patient
    overall_results = {
        'patients_processed': 0,
        'patients_failed': 0,
        'total_files_processed': 0,
        'total_files_failed': 0
    }
    
    for patient_dir, patient_id in patient_dirs:
        try:
            result = process_patient_folder(patient_dir, args.output_dir, patient_id)
            
            if result['success']:
                overall_results['patients_processed'] += 1
            else:
                overall_results['patients_failed'] += 1
            
            overall_results['total_files_processed'] += result['files_processed']
            overall_results['total_files_failed'] += result.get('files_failed', 0)
            
        except Exception as e:
            print(f"ERROR: Critical error processing {patient_id}: {e}")
            overall_results['patients_failed'] += 1
    
    # Final summary
    print("\\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print(f"SUCCESS: Patients processed: {overall_results['patients_processed']}")
    print(f"ERROR: Patients failed: {overall_results['patients_failed']}")
    print(f"Total files processed: {overall_results['total_files_processed']}")
    print(f"Total files failed: {overall_results['total_files_failed']}")
    
    if overall_results['patients_processed'] > 0:
        print(f"\\nOutput directory: {args.output_dir}")
        print("Ready for noise injection and denoising evaluation!")
    
    return 0 if overall_results['patients_failed'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())