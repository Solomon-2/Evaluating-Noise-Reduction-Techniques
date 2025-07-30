
import os
import librosa
import numpy as np
import soundfile as sf
import argparse

from find_data_categories import return_file_path


def combine_audio(clean_sound, noise, output_name, noise_level):
  """Legacy function - kept for backward compatibility"""
  y_voice, sr_voice = librosa.load(clean_sound, sr=16000)
  y_noise, sr_noise = librosa.load(noise, sr=16000)

  if len(y_noise) < len(y_voice):
      # Repeat noise to cover the entire clean file, then truncate
      repeats = int(np.ceil(len(y_voice) / len(y_noise)))
      y_noise_matched = np.tile(y_noise, repeats)[:len(y_voice)] 
  else:
      # Truncate noise if it's too long
      y_noise_matched = y_noise[:len(y_voice)]

  # Match noise energy to clean file (original method)
  rms_voice = np.sqrt(np.mean(y_voice**2))
  rms_noise = np.sqrt(np.mean(y_noise_matched**2))
  if rms_noise > 0:
    y_noise_matched = y_noise_matched * (rms_voice / rms_noise) * noise_level
  
  y_mixed=y_voice+y_noise_matched

  #Normalizing the audio to prevent severe suffering of listener
  max_ampl_val = np.max(np.abs(y_mixed))
  if max_ampl_val > 0:
    y_mixed_normalized = y_mixed / max_ampl_val
  else:
    y_mixed_normalized=y_mixed

  if not output_name.lower().endswith(".wav"):
    output_name +=".wav"
    #Save the combined audio

  sf.write(output_name, y_mixed_normalized, 16000)
  return output_name


def combine_audio_snr(clean_sound, noise, output_name, snr_db):
  """Combine clean audio with noise at specified SNR level"""
  y_voice, sr_voice = librosa.load(clean_sound, sr=16000)
  y_noise, sr_noise = librosa.load(noise, sr=16000)

  if len(y_noise) < len(y_voice):
      # Repeat noise to cover the entire clean file, then truncate
      repeats = int(np.ceil(len(y_voice) / len(y_noise)))
      y_noise_matched = np.tile(y_noise, repeats)[:len(y_voice)] 
  else:
      # Truncate noise if it's too long
      y_noise_matched = y_noise[:len(y_voice)]

  # Calculate signal and noise power
  signal_power = np.mean(y_voice**2)
  noise_power = np.mean(y_noise_matched**2)
  
  # Calculate required noise scaling for target SNR
  # SNR_dB = 10 * log10(signal_power / noise_power)
  # Therefore: noise_power_target = signal_power / (10^(SNR_dB/10))
  if noise_power > 0:
    target_noise_power = signal_power / (10**(snr_db/10))
    noise_scale = np.sqrt(target_noise_power / noise_power)
    y_noise_matched = y_noise_matched * noise_scale
  
  y_mixed = y_voice + y_noise_matched

  # Normalizing the audio to prevent severe suffering of listener😂
  max_ampl_val = np.max(np.abs(y_mixed))
  if max_ampl_val > 0:
    y_mixed_normalized = y_mixed / max_ampl_val
  else:
    y_mixed_normalized = y_mixed

  if not output_name.lower().endswith(".wav"):
    output_name += ".wav"
    # Save the combined audio

  sf.write(output_name, y_mixed_normalized, 16000)
  
  # Calculate and return actual achieved SNR for verification
  final_signal_power = np.mean(y_voice**2)
  final_noise_power = np.mean((y_noise_matched)**2)
  if final_noise_power > 0:
    actual_snr = 10 * np.log10(final_signal_power / final_noise_power)
    print(f"Target SNR: {snr_db:.1f} dB, Actual SNR: {actual_snr:.1f} dB")
  
  return output_name
  

  
def main():
    parser = argparse.ArgumentParser(
        description = "Combine audio files"
    )

    parser.add_argument("-cl", "--clean", help="Clean Audio file or directory")
    parser.add_argument("-n", "--noise", help="Noise")
    parser.add_argument("-out", "--output_name", help="Output file name (or output directory if --clean is a directory)")
    parser.add_argument("-cat", "--noise_category", help="Noise category (optional)")
    parser.add_argument("-nl", "--noise_level", help="Noise Level (legacy - use --snr instead)")
    parser.add_argument("-snr", "--snr_db", help="Target SNR in dB (recommended)", type=float)
    # Optionally, you can add a flag to specify the category of noise, e.g. thunderstorm, rain

    args = parser.parse_args()

    try:
        noise_file = args.noise
        # If noise_category is provided, use the first file from that category
        if args.noise_category and not args.noise:
            file_paths = return_file_path([args.noise_category])
            if file_paths:
                noise_file = file_paths[0]
                print(f"Using noise file from category '{args.noise_category}': {noise_file}")
            else:
                print(f"No files found for category '{args.noise_category}'.")
                return
        # If --clean is a directory, process all wav files in it
        if args.clean and os.path.isdir(args.clean):
            if not args.output_name:
                print("Please provide --output_name as an output directory when using a clean directory.")
                return
            os.makedirs(args.output_name, exist_ok=True)
            clean_files = [f for f in os.listdir(args.clean) if f.lower().endswith('.wav')]
            if not clean_files:
                print(f"No .wav files found in directory {args.clean}")
                return
            for clean_file in clean_files:
                clean_path = os.path.join(args.clean, clean_file)
                output_path = os.path.join(args.output_name, f"mixed_{clean_file}")
                if noise_file and (args.snr_db is not None or args.noise_level):
                    if args.snr_db is not None:
                        # Use new SNR-based method
                        combine_audio_snr(clean_path, noise_file, output_path, args.snr_db)
                        print(f"Combined {clean_path} with {noise_file} at {args.snr_db} dB SNR -> {output_path}")
                    else:
                        # Use legacy noise level method
                        noise_level = float(args.noise_level)
                        combine_audio(clean_path, noise_file, output_path, noise_level)
                        print(f"Combined {clean_path} with {noise_file} (legacy method) -> {output_path}")
                else:
                    print("Please provide --noise or --noise_category and either --snr or --noise_level.")
                    return
        # If --clean is a single file
        elif args.clean and noise_file and args.output_name and (args.snr_db is not None or args.noise_level):
            if args.snr_db is not None:
                # Use new SNR-based method
                combine_audio_snr(args.clean, noise_file, args.output_name, args.snr_db)
                print(f"Combined {args.clean} with {noise_file} at {args.snr_db} dB SNR -> {args.output_name}")
            else:
                # Use legacy noise level method
                noise_level = float(args.noise_level)
                combine_audio(args.clean, noise_file, args.output_name, noise_level)
                print(f"Combined {args.clean} with {noise_file} (legacy method) -> {args.output_name}")
        else:
            print("Please provide --clean, --output_name, and either --noise or --noise_category.")
            print("Also provide either --snr (recommended) or --noise_level (legacy).")
    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
   main()
