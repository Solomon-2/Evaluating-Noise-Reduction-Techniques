# Modular Notebook Structure for Audio Processing Pipeline

Here are the cells to add to your notebook to make it fully modular:

## Cell 7: Test Module - Spectral Subtraction Denoiser
```python
# Cell 7: Test Module - Spectral Subtraction Denoiser
print("--- Testing Module: Spectral Subtraction Denoiser ---")

def test_spectral_denoiser(noisy_test_data):
    if not noisy_test_data:
        print("❌ No noisy test data available. Run noise injection test first.")
        return None
        
    print("Testing spectral subtraction denoiser...")
    input_path = noisy_test_data['output_path']
    output_path = input_path.replace('test_noisy_', 'test_denoised_spectral_')
    
    denoiser_command = [
        VENV_PYTHON, os.path.join(DENOISER_SCRIPTS_DIR, "spec_subtraction_same_file.py"),
        "--input", input_path, "--output", output_path
    ]
    
    print(f"Command: {' '.join(denoiser_command)}")
    
    try:
        result = subprocess.run(denoiser_command, capture_output=True, text=True, check=True, timeout=120)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            denoised_audio, sr = librosa.load(output_path, sr=None)
            print(f"✅ Spectral subtraction successful!")
            print(f"   Output file: {output_path}")
            print(f"   Duration: {len(denoised_audio)/sr:.1f} seconds")
            return {'output_path': output_path, 'denoised_audio': denoised_audio, 'sample_rate': sr}
        else:
            print(f"❌ Output file not created: {output_path}")
            return None
            
    except Exception as e:
        print(f"❌ Spectral denoiser failed: {e}")
        return None

# Test spectral denoiser
first_successful_noisy = next((result for result in noise_test_results.values() if result is not None), None)
spectral_result = test_spectral_denoiser(first_successful_noisy)
```

## Cell 8: Test Module - Wiener Filter Denoiser
```python
# Cell 8: Test Module - Wiener Filter Denoiser
print("--- Testing Module: Wiener Filter Denoiser ---")

def test_wiener_denoiser(noisy_test_data):
    if not noisy_test_data:
        print("❌ No noisy test data available.")
        return None
        
    print("Testing Wiener filter denoiser...")
    input_path = noisy_test_data['output_path']
    output_path = input_path.replace('test_noisy_', 'test_denoised_wiener_')
    
    denoiser_command = [
        VENV_PYTHON, os.path.join(DENOISER_SCRIPTS_DIR, "wiener_filtering.py"),
        "--input", input_path, "--output", output_path
    ]
    
    print(f"Command: {' '.join(denoiser_command)}")
    
    try:
        result = subprocess.run(denoiser_command, capture_output=True, text=True, check=True, timeout=120)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            denoised_audio, sr = librosa.load(output_path, sr=None)
            print(f"✅ Wiener filter successful!")
            print(f"   Output file: {output_path}")
            return {'output_path': output_path, 'denoised_audio': denoised_audio, 'sample_rate': sr}
        else:
            print(f"❌ Output file not created: {output_path}")
            return None
            
    except Exception as e:
        print(f"❌ Wiener denoiser failed: {e}")
        return None

# Test Wiener denoiser
wiener_result = test_wiener_denoiser(first_successful_noisy)
```

## Cell 9: Test Module - LogMMSE Denoiser
```python
# Cell 9: Test Module - LogMMSE Denoiser
print("--- Testing Module: LogMMSE Denoiser ---")

def test_logmmse_denoiser(noisy_test_data):
    if not noisy_test_data:
        print("❌ No noisy test data available.")
        return None
        
    print("Testing LogMMSE denoiser...")
    input_path = noisy_test_data['output_path']
    output_path = input_path.replace('test_noisy_', 'test_denoised_logmmse_')
    
    denoiser_command = [
        VENV_PYTHON, os.path.join(DENOISER_SCRIPTS_DIR, "log_mmse.py"),
        "--input", input_path, "--output", output_path
    ]
    
    print(f"Command: {' '.join(denoiser_command)}")
    
    try:
        result = subprocess.run(denoiser_command, capture_output=True, text=True, check=True, timeout=120)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            denoised_audio, sr = librosa.load(output_path, sr=None)
            print(f"✅ LogMMSE successful!")
            print(f"   Output file: {output_path}")
            return {'output_path': output_path, 'denoised_audio': denoised_audio, 'sample_rate': sr}
        else:
            print(f"❌ Output file not created: {output_path}")
            return None
            
    except Exception as e:
        print(f"❌ LogMMSE denoiser failed: {e}")
        return None

# Test LogMMSE denoiser
logmmse_result = test_logmmse_denoiser(first_successful_noisy)
```

## Cell 10: Test Module - Feature Extraction
```python
# Cell 10: Test Module - Feature Extraction
print("--- Testing Module: Feature Extraction ---")

def test_feature_extraction(audio_file_path, label=0):
    """Test feature extraction on a single audio file"""
    try:
        audio_data, sr = librosa.load(audio_file_path, sr=None)
        frame_size = int(FRAME_DURATION_SEC * sr)
        
        features_list = []
        num_frames = len(audio_data) // frame_size
        
        print(f"Extracting features from {num_frames} frames...")
        
        for i in range(num_frames):
            frame_start = i * frame_size
            frame_end = frame_start + frame_size
            frame = audio_data[frame_start:frame_end]
            
            # Extract features using the helper function
            features = extract_features(frame, sr)
            features['frame_index'] = i
            features['label'] = label
            features_list.append(features)
        
        print(f"✅ Feature extraction successful!")
        print(f"   Extracted {len(features_list)} feature vectors")
        print(f"   Features per frame: {len(features_list[0])-2}")  # -2 for frame_index and label
        
        return features_list
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return None

# Test feature extraction on different audio types
print("\n--- Testing on Raw Audio ---")
if test_data:
    raw_features = test_feature_extraction(test_data['chunk_path'], label=0)

print("\n--- Testing on Noisy Audio ---")
if first_successful_noisy:
    noisy_features = test_feature_extraction(first_successful_noisy['output_path'], label=0)

print("\n--- Testing on Denoised Audio ---")
if spectral_result:
    denoised_features = test_feature_extraction(spectral_result['output_path'], label=0)
```

## Cell 11: Test Module - Complete Pipeline Test
```python
# Cell 11: Test Module - Complete Pipeline Test
print("--- Testing Module: Complete Pipeline Test ---")

def test_complete_pipeline(test_chunk_path, temp_dir):
    """Test the complete pipeline: raw -> noisy -> denoised -> features"""
    
    pipeline_results = {}
    
    print("🔄 Running complete pipeline test...")
    
    # Step 1: Process raw audio
    print("\n1️⃣ Processing raw audio...")
    raw_features = test_feature_extraction(test_chunk_path, label=0)
    if raw_features:
        pipeline_results['raw'] = {'features': raw_features, 'count': len(raw_features)}
        print(f"   ✅ Raw: {len(raw_features)} feature vectors")
    
    # Step 2: Add noise and process
    for noise_cat in NOISE_CATEGORIES[:1]:  # Test with just first noise category
        print(f"\n2️⃣ Processing {noise_cat} noise...")
        
        # Generate noisy audio
        noisy_result = test_noise_injection(test_data, noise_cat)
        if not noisy_result:
            continue
            
        # Extract features from noisy audio
        noisy_features = test_feature_extraction(noisy_result['output_path'], label=0)
        if noisy_features:
            pipeline_results[f'{noise_cat}_noisy'] = {'features': noisy_features, 'count': len(noisy_features)}
            print(f"   ✅ {noise_cat} noisy: {len(noisy_features)} feature vectors")
        
        # Step 3: Denoise and process
        for denoiser_name in DENOISER_SCRIPT_MAP.keys():
            print(f"\n3️⃣ Processing {noise_cat} + {denoiser_name} denoising...")
            
            # Create denoised audio
            input_path = noisy_result['output_path']
            output_path = os.path.join(temp_dir, f"pipeline_test_{noise_cat}_{denoiser_name}.wav")
            
            denoiser_command = [
                VENV_PYTHON, os.path.join(DENOISER_SCRIPTS_DIR, DENOISER_SCRIPT_MAP[denoiser_name]),
                "--input", input_path, "--output", output_path
            ]
            
            try:
                subprocess.run(denoiser_command, capture_output=True, text=True, check=True, timeout=120)
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    # Extract features from denoised audio
                    denoised_features = test_feature_extraction(output_path, label=0)
                    if denoised_features:
                        pipeline_results[f'{noise_cat}_{denoiser_name}'] = {'features': denoised_features, 'count': len(denoised_features)}
                        print(f"   ✅ {noise_cat} + {denoiser_name}: {len(denoised_features)} feature vectors")
                
            except Exception as e:
                print(f"   ❌ {denoiser_name} failed: {e}")
    
    return pipeline_results

# Run complete pipeline test
if test_data:
    pipeline_results = test_complete_pipeline(test_data['chunk_path'], test_data['temp_dir'])
    
    print(f"\n🎯 Pipeline Test Summary:")
    for condition, result in pipeline_results.items():
        print(f"   {condition}: {result['count']} feature vectors")
    
    total_conditions = len(pipeline_results)
    expected_conditions = 1 + len(NOISE_CATEGORIES) + (len(NOISE_CATEGORIES) * len(DENOISER_SCRIPT_MAP))
    print(f"\n📊 Conditions generated: {total_conditions}")
    print(f"📊 Expected conditions: {expected_conditions}")
```

## Cell 12: Main Processing Loop (Full Dataset Generation)
```python
# Cell 12: Main Processing Loop (Full Dataset Generation)
print("--- Main Processing Loop: Full Dataset Generation ---")

def run_full_pipeline():
    """Run the complete dataset generation pipeline"""
    
    print("🚀 Starting full dataset generation...")
    
    all_features_list = []
    
    # Get patient folders to process
    if not PATIENT_FOLDERS_TO_PROCESS:
        all_local_patient_folders = sorted([f.name for f in os.scandir(RAW_PATIENT_DATA_BASE_DIR) 
                                          if f.is_dir() and f.name.startswith('patient-')])
        patient_folders_to_use = all_local_patient_folders
    else:
        patient_folders_to_use = PATIENT_FOLDERS_TO_PROCESS

    if DEBUG_MODE:
        patient_folders_to_use = patient_folders_to_use[:DEBUG_PATIENT_COUNT]
        print(f"🔧 DEBUG_MODE: Processing only {DEBUG_PATIENT_COUNT} patient(s)")

    # Process each patient
    for i, patient_folder_name in enumerate(tqdm(patient_folders_to_use, desc="Processing Patients")):
        
        patient_id_formatted = f"patient_{str(i+1).zfill(2)}"
        print(f"\n👤 Processing {patient_id_formatted} (Folder: {patient_folder_name})")
        
        # Create temporary processing directory
        current_temp_process_dir = os.path.join(RAW_PATIENT_DATA_BASE_DIR, f"temp_{patient_id_formatted}")
        os.makedirs(current_temp_process_dir, exist_ok=True)
        
        try:
            # Load RML annotations
            patient_local_dir = os.path.join(RAW_PATIENT_DATA_BASE_DIR, patient_folder_name)
            rml_files = [f for f in os.listdir(patient_local_dir) if f.endswith('.rml')]
            
            if not rml_files:
                print(f"   ❌ No RML file found. Skipping patient.")
                continue
                
            rml_path = os.path.join(patient_local_dir, rml_files[0])
            apnea_events = parse_respironics_rml(rml_path)
            
            # Process EDF files
            edf_files = sorted([f for f in os.listdir(patient_local_dir) if f.endswith('.edf')])
            if not edf_files:
                print(f"   ❌ No EDF files found. Skipping patient.")
                continue

            for edf_filename in edf_files:
                print(f"   📁 Processing {edf_filename}")
                edf_path = os.path.join(patient_local_dir, edf_filename)
                
                try:
                    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                    fs = int(raw.info['sfreq'])
                    raw.pick_channels([AUDIO_CHANNEL_NAME])
                except Exception as e:
                    print(f"   ❌ Could not read {edf_filename}: {e}")
                    continue

                # Process in chunks
                chunk_size_samples = CHUNK_DURATION_MIN * 60 * fs
                num_chunks = int(np.ceil(raw.n_times / chunk_size_samples))

                for chunk_idx in range(num_chunks):
                    start_sample = chunk_idx * chunk_size_samples
                    stop_sample = min(start_sample + chunk_size_samples, raw.n_times)
                    audio_chunk_raw, _ = raw[:, start_sample:stop_sample]
                    audio_chunk_raw = audio_chunk_raw.flatten()

                    # Save raw chunk
                    temp_raw_chunk_path = os.path.join(current_temp_process_dir, f"temp_chunk_raw_{chunk_idx}.wav")
                    sf.write(temp_raw_chunk_path, audio_chunk_raw, fs)

                    processed_chunk_wav_paths = {"raw": temp_raw_chunk_path}

                    # Add noise
                    for noise_category in NOISE_CATEGORIES:
                        noisy_output_path = os.path.join(current_temp_process_dir, f"temp_chunk_{noise_category}_noisy_{chunk_idx}.wav")
                        
                        combine_command = [
                            VENV_PYTHON, os.path.join(DENOISER_SCRIPTS_DIR, "combining_audio.py"),
                            "-cl", temp_raw_chunk_path,
                            "-out", noisy_output_path,
                            "-cat", noise_category,
                            "-nl", str(NOISE_LEVEL_RMS_RATIO)
                        ]
                        
                        try:
                            subprocess.run(combine_command, capture_output=True, text=True, check=True, timeout=300)
                            if os.path.exists(noisy_output_path) and os.path.getsize(noisy_output_path) > 0:
                                processed_chunk_wav_paths[f"{noise_category}_noisy"] = noisy_output_path
                        except Exception as e:
                            print(f"   ⚠️  Noise injection failed for {noise_category}: {e}")

                    # Apply denoising
                    for condition_key_prefix, noisy_chunk_path in list(processed_chunk_wav_paths.items()):
                        if "_noisy" in condition_key_prefix:
                            for denoiser_name, script_filename in DENOISER_SCRIPT_MAP.items():
                                denoised_output_path = os.path.join(current_temp_process_dir, f"temp_chunk_{condition_key_prefix}_{denoiser_name}_{chunk_idx}.wav")
                                
                                denoised_audio_result = run_denoiser_script(
                                    denoiser_name, noisy_chunk_path, denoised_output_path, 
                                    DENOISER_SCRIPT_MAP, fs, current_temp_process_dir
                                )
                                
                                if denoised_audio_result is not None:
                                    sf.write(denoised_output_path, denoised_audio_result, fs)
                                    processed_chunk_wav_paths[f"{condition_key_prefix.replace('_noisy', '')}_{denoiser_name}"] = denoised_output_path

                    # Extract features from all audio versions
                    for condition_suffix, audio_wav_path in processed_chunk_wav_paths.items():
                        try:
                            audio_data_for_features, current_sr = librosa.load(audio_wav_path, sr=None)
                            frame_size = int(FRAME_DURATION_SEC * current_sr)
                            num_frames_in_data = len(audio_data_for_features) // frame_size
                            
                            for i in range(num_frames_in_data):
                                frame_start = i * frame_size
                                frame_end = frame_start + frame_size
                                frame = audio_data_for_features[frame_start:frame_end]
                                
                                global_frame_start = (start_sample + frame_start) / fs
                                global_frame_end = global_frame_start + FRAME_DURATION_SEC
                                
                                is_apnea = 0
                                for event_start, event_end in apnea_events:
                                    if global_frame_start < event_end and global_frame_end > event_start:
                                        is_apnea = 1
                                        break
                                
                                features = extract_features(frame, sr=current_sr)
                                features['patient_id'] = f"{patient_id_formatted}_{condition_suffix}"
                                features['label'] = is_apnea
                                all_features_list.append(features)
                                
                        except Exception as e:
                            print(f"   ⚠️  Feature extraction failed for {condition_suffix}: {e}")
        
        finally:
            # Clean up temp directory
            if os.path.exists(current_temp_process_dir):
                shutil.rmtree(current_temp_process_dir)
                print(f"   🧹 Cleaned up temp files for {patient_id_formatted}")

    return all_features_list

# Ask user if they want to run the full pipeline
print("\n⚠️  This will run the complete dataset generation pipeline.")
print("   Make sure all individual modules are working first!")
print("\n   To run the full pipeline, uncomment the line below:")
print("# all_features_list = run_full_pipeline()")
```

## Cell 13: Final Dataset Creation and Cleanup
```python
# Cell 13: Final Dataset Creation and Cleanup
print("--- Final Dataset Creation and Cleanup ---")

def create_final_dataset(all_features_list):
    """Create final CSV dataset from features list"""
    
    if not all_features_list:
        print("❌ No features to save. Run the full pipeline first.")
        return
    
    print("💾 Creating final dataset...")
    
    df_final = pd.DataFrame(all_features_list)
    
    if not df_final.empty:
        # Generate filename
        num_patients_processed = len(df_final['patient_id'].apply(lambda x: x.split('_')[1]).unique())
        num_conditions_per_patient = 1 + len(NOISE_CATEGORIES) + (len(NOISE_CATEGORIES) * len(DENOISER_SCRIPT_MAP))
        
        final_csv_name = f"augmented_apnea_dataset_{num_patients_processed}patients_{num_conditions_per_patient}conditions.csv"
        final_save_path = os.path.join('..', 'data', 'sleep_data', final_csv_name)
        
        # Save dataset
        df_final.to_csv(final_save_path, index=False)
        
        print(f"✅ Dataset saved: {final_save_path}")
        print(f"📊 Total frames: {len(df_final)}")
        print(f"📊 Total patients: {num_patients_processed}")
        print(f"📊 Conditions per patient: {num_conditions_per_patient}")
        
        # Show unique patient_id variants (first 10)
        print("\n🔍 Sample patient_id variants:")
        for pid_variant in sorted(df_final['patient_id'].unique())[:10]:
            print(f"   - {pid_variant}")
        
        # Show label distribution
        print(f"\n📈 Label distribution:")
        label_dist = df_final['label'].value_counts(normalize=True)
        for label, proportion in label_dist.items():
            print(f"   Label {label}: {proportion:.2%}")
        
        return final_save_path
    else:
        print("❌ No data to save")
        return None

def cleanup_test_files():
    """Clean up any remaining test files"""
    print("\n🧹 Cleaning up test files...")
    
    test_dirs = []
    for item in os.listdir(RAW_PATIENT_DATA_BASE_DIR):
        item_path = os.path.join(RAW_PATIENT_DATA_BASE_DIR, item)
        if os.path.isdir(item_path) and item.startswith('temp_'):
            test_dirs.append(item_path)
    
    for test_dir in test_dirs:
        try:
            shutil.rmtree(test_dir)
            print(f"   🗑️  Removed: {test_dir}")
        except Exception as e:
            print(f"   ⚠️  Could not remove {test_dir}: {e}")
    
    print("✅ Cleanup complete!")

# Uncomment these lines after running the full pipeline:
# final_dataset_path = create_final_dataset(all_features_list)
# cleanup_test_files()

print("\n🎉 Modular notebook setup complete!")
print("   You can now test each module individually before running the full pipeline.")
```

To use this modular structure:

1. **Add these cells to your notebook** in order after your current cells
2. **Test each module individually** by running cells 5-11 one by one
3. **Debug any issues** in individual modules before running the full pipeline
4. **Run the full pipeline** (Cell 12) only after all modules are working
5. **Create final dataset** (Cell 13) after the full pipeline completes

Each cell is self-contained and can be run independently to test specific functionality!