import librosa
import numpy as np
import soundfile as sf
import argparse

from find_data_categories import return_file_path


def combine_audio(clean_sound, noise, output_name):
  y_voice, sr_voice = librosa.load(clean_sound, sr=16000)
  y_noise, sr_noise = librosa.load(noise, sr=16000)


  if len(y_noise) < len(y_voice):
      # Repeat noise to cover the entire clean file, then truncate
      repeats = int(np.ceil(len(y_voice) / len(y_noise)))
      y_noise_matched = np.tile(y_noise, repeats)[:len(y_voice)]
  else:
      # Truncate noise if it's too long
      y_noise_matched = y_noise[:len(y_voice)]

# Match noise energy to clean file
  rms_voice = np.sqrt(np.mean(y_voice**2))
  rms_noise = np.sqrt(np.mean(y_noise_matched**2))
  if rms_noise > 0:
    y_noise_matched = y_noise_matched * (rms_voice / rms_noise)
  
  y_mixed=y_voice+y_noise_matched

  #Normalizing the audio to prevent severe suffering of listener😂
  max_ampl_val = np.max(np.abs(y_mixed))
  if max_ampl_val > 0:
    y_mixed_normalized = y_mixed / max_ampl_val

  else:
    y_mixed_normalized=y_mixed

  
  if not output_name.lower().endswith(".wav"):
    output_name +=".wav"
    #Save the combined audio

  sf.write(output_name, y_mixed_normalized, 16000)

  return
  

  
def main():
    parser = argparse.ArgumentParser(
        description = "Combine audio files"
    )

    parser.add_argument("-cl", "--clean", help="Clean Audio")
    parser.add_argument("-n", "--noise", help="Noise")
    parser.add_argument("-out", "--output_name", help="Output file name")
    parser.add_argument("-cat", "--noise_category", help="Noise category (optional)")
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
        if args.clean and noise_file and args.output_name:
            combine_audio(args.clean, noise_file, args.output_name)
        else:
            print("Please provide --clean, --output_name, and either --noise or --noise_category.")
    except Exception as e:
        print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
   main()
