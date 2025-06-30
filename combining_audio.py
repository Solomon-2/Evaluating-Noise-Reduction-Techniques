import librosa
import numpy as np
import soundfile as sf
import argparse


def combine_audio(clean_sound, noise, output_name):
  y_voice, sr_voice = librosa.load(clean_sound, sr=16000)
  y_noise, sr_noise = librosa.load(noise, sr=16000)


  if len(y_noise) < len(y_voice):
      # Pad noise with zeros if it's too short
      y_noise_matched = np.pad(y_noise, (0, len(y_voice) - len(y_noise)), mode='constant')
  else:
      # Truncate noise if it's too long
      y_noise_matched = y_noise[:len(y_voice)]

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

  args = parser.parse_args()

  try:
    if args.clean and args.noise and args.output_name:
      combine_audio(args.clean, args.noise, args.output_name)

  except Exception as e:
            print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
   main()
