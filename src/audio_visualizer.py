import matplotlib.pyplot as plt
import sys
import soundfile as sf
import librosa.display
import numpy as np

def visualize_audio(file_path, output_path="output.png"):
    audio, sr = sf.read(file_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Convert to mono if stereo

    plt.figure(figsize=(12, 6))

    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot spectrogram
    plt.subplot(2, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    plt.title('Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_visualizer.py <audio_file> [output_image]")
        sys.exit(1)
    audio_file = sys.argv[1]
    output_image = sys.argv[2] if len(sys.argv) > 2 else "output.png"
    visualize_audio(audio_file, output_image)