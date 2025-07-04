import os
import numpy as np
import soundfile as sf
import argparse
from scipy.signal import find_peaks

# Feature extraction for regular breathing: energy periodicity, mean, std, and spectral centroid
def extract_features(audio, sr, frame_sec=1.0):
    frame_len = int(frame_sec * sr)
    energies = np.array([np.mean(np.abs(audio[i:i+frame_len])) for i in range(0, len(audio), frame_len)])
    # Autocorrelation of energy envelope
    ac = np.correlate(energies - np.mean(energies), energies - np.mean(energies), mode='full')
    ac = ac[len(ac)//2:]
    # Find peaks in autocorrelation (ignore lag 0)
    peaks, _ = find_peaks(ac[1:])
    if len(peaks) > 0:
        dominant_period = peaks[0] + 1  # in frames
        dom_period_sec = dominant_period * frame_sec
    else:
        dom_period_sec = 0
    # Spectral centroid (low for breathing)
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1/sr)
    spectral_centroid = np.sum(freqs * spectrum) / np.sum(spectrum) if np.sum(spectrum) > 0 else 0
    return np.array([
        np.mean(energies),
        np.std(energies),
        dom_period_sec,
        spectral_centroid
    ])

def main():

    parser = argparse.ArgumentParser(description="Find top 5 sleep audio files with regular breathing features.")
    parser.add_argument('--search_dir', required=True, help='Directory to search for candidate wav files')
    parser.add_argument('--output_dir', required=True, help='Directory to copy top 5 regular breathing files to')
    args = parser.parse_args()

    candidates = [os.path.join(args.search_dir, f) for f in os.listdir(args.search_dir)
                  if f.lower().endswith('.wav')]

    feature_list = []
    for f in candidates:
        try:
            audio, sr = sf.read(f)
            feat = extract_features(audio, sr)
            # Heuristic: prefer files with moderate mean energy, low std, dominant period in 1-8s (7-60 bpm), low spectral centroid
            mean_energy, std_energy, dom_period_sec, spectral_centroid = feat
            # Score: penalize high std, out-of-range period, high centroid
            period_score = 0 if 1 <= dom_period_sec <= 8 else abs(dom_period_sec - 4)
            score = std_energy + period_score + (spectral_centroid / 1000)
            feature_list.append((score, f, mean_energy, std_energy, dom_period_sec, spectral_centroid))
        except Exception as e:
            print(f"Error processing {f}: {e}")

    feature_list.sort()
    top5 = feature_list[:5]

    os.makedirs(args.output_dir, exist_ok=True)
    for _, f, mean_energy, std_energy, dom_period_sec, spectral_centroid in top5:
        out_path = os.path.join(args.output_dir, os.path.basename(f))
        with open(f, 'rb') as src, open(out_path, 'wb') as dst:
            dst.write(src.read())
        print(f"Copied {f} to {out_path}")

    print("Top 5 regular breathing candidates:")
    for score, f, mean_energy, std_energy, dom_period_sec, spectral_centroid in top5:
        print(f"{f} | score={score:.3f} | mean_energy={mean_energy:.3f} | std={std_energy:.3f} | period={dom_period_sec:.2f}s | centroid={spectral_centroid:.1f}Hz")

if __name__ == "__main__":
    main()
