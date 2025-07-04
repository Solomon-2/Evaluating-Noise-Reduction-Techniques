import os
import sys
import soundfile as sf
import librosa
import subprocess
import shutil
import numpy as np

def resample_audio(input_path, output_path, target_sr=44100):
    audio, sr = sf.read(input_path)
    # Handle mono and stereo
    if audio.ndim == 2:
        # Resample each channel separately
        channels = []
        for ch in audio.T:
            ch_resampled = librosa.resample(ch, orig_sr=sr, target_sr=target_sr)
            channels.append(ch_resampled)
        audio_resampled = np.stack(channels, axis=1)  # shape: (n_samples, 2)
        audio_resampled = np.ascontiguousarray(audio_resampled.copy())  # Ensure no memory aliasing
    else:
        audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        audio_resampled = np.ascontiguousarray(audio_resampled)
    if sr != target_sr:
        sf.write(output_path, audio_resampled, target_sr)
        print(f"Resampled {input_path} to {target_sr} Hz -> {output_path}")
    else:
        shutil.copy(input_path, output_path)
        print(f"Copied {input_path} to {output_path} (already {target_sr} Hz)")

def run_demucs(input_path, output_dir="demucs_output"):
    cmd = [
        "demucs",
        "--two-stems", "vocals",
        "-o", output_dir,
        input_path
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # Find the output file
    base = os.path.splitext(os.path.basename(input_path))[0]
    vocals_path = os.path.join(output_dir, "htdemucs", base, "vocals.wav")
    if os.path.exists(vocals_path):
        print(f"Denoised file saved at: {vocals_path}")
    else:
        print("Denoised file not found. Check Demucs output.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demucs_denoise.py <input_audio>")
        sys.exit(1)
    input_audio = sys.argv[1]
    resampled_audio = "resampled_input.wav"
    resample_audio(input_audio, resampled_audio)
    run_demucs(resampled_audio)
