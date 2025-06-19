import librosa

def get_audio_sample_rate(audio_file_path):
  """
  Returns the sampling rate of the given audio file.
  """
  try:
    y, sr = librosa.load(audio_file_path, sr=None)
    print(f"The sampling rate of {audio_file_path} is {sr} Hz")
    return sr
  except Exception as e:
    print(f"Error loading or processing audio file: {e}")
    return None

# Example usage:
# get_audio_sample_rate('ESC-50-master/audio/1-7057-A-12.wav')
