

import torch
import torchaudio
import os
import argparse
from speechbrain.inference.enhancement import SpectralMaskEnhancement

def enhance_file(input_path, output_path, enhance_model):
    noisy = enhance_model.load_audio(input_path).unsqueeze(0)
    enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))
    torchaudio.save(output_path, enhanced.cpu(), 16000)

def batch_enhance(input_dir, output_dir, enhance_model, ext='.wav'):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(ext):
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            print(f"Processing {in_path} -> {out_path}")
            enhance_file(in_path, out_path, enhance_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoise audio using SpeechBrain MetricGAN+ (single file or batch directory)")
    parser.add_argument('--input', required=True, help='Input audio file or directory')
    parser.add_argument('--output', required=True, help='Output audio file or directory')
    parser.add_argument('--ext', default='.wav', help='Audio file extension to process in batch mode (default: .wav)')
    args = parser.parse_args()

    enhance_model = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank",
    )

    if os.path.isdir(args.input):
        batch_enhance(args.input, args.output, enhance_model, ext=args.ext)
    else:
        enhance_file(args.input, args.output, enhance_model)
