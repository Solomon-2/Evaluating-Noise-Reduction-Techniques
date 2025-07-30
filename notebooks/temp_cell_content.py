# Cell 7: Test Module - Spectral Subtraction Denoiser
print("--- Testing Module: Spectral Subtraction Denoiser ---")

def test_spectral_denoiser(noisy_test_data):
    """
    Test the spectral subtraction denoiser
    """
    if not noisy_test_data:
        print("❌ No noisy test data available. Run noise injection test first.")
        return None
        
    print("Testing spectral subtraction denoiser...")
    
    # Use first successful noisy audio
    input_path = noisy_test_data['output_path']
    output_path = input_path.replace('test_noisy_', 'test_denoised_spectral_')
    
    # Build command
    denoiser_command = [
        VENV_PYTHON, os.path.join(DENOISER_SCRIPTS_DIR, "spec_subtraction_same_file.py"),
        "--input", input_path,
        "--output", output_path
    ]
    
    print(f"Command: {' '.join(denoiser_command)}")
    
    try:
        # Run the denoiser
        result = subprocess.run(denoiser_command, capture_output=True, text=True, check=True, timeout=120)
        
        # Check if output file was created
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            # Load and analyze the result
            denoised_audio, sr = librosa.load(output_path, sr=None)
            print(f"✅ Spectral subtraction successful!")
            print(f"   Output file: {output_path}")
            print(f"   Output duration: {len(denoised_audio)/sr:.1f} seconds")
            print(f"   Output sample rate: {sr} Hz")
            
            # Calculate RMS
            denoised_rms = np.sqrt(np.mean(denoised_audio**2))
            noisy_rms = np.sqrt(np.mean(noisy_test_data['noisy_audio']**2))
            print(f"   Noisy RMS: {noisy_rms:.6f}")
            print(f"   Denoised RMS: {denoised_rms:.6f}")
            
            return {'output_path': output_path, 'denoised_audio': denoised_audio, 'sample_rate': sr}
        else:
            print(f"❌ Output file not created or is empty: {output_path}")
            print(f"   STDOUT: {result.stdout}")
            print(f"   STDERR: {result.stderr}")
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Spectral denoiser failed with exit code {e.returncode}")
        print(f"   STDOUT: {e.stdout}")
        print(f"   STDERR: {e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        print("❌ Spectral denoiser timed out")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

# Test spectral denoiser on first successful noisy audio
first_successful_noisy = next((result for result in noise_test_results.values() if result is not None), None)
spectral_result = test_spectral_denoiser(first_successful_noisy)