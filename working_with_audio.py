# Checking the asmple rate of an audio file

import librosa

audio_file_path='ESC-50-master/audio/1-7057-A-12.wav'

try:
  
  # Setting sr = None to ensure librosa uses original sampling rate

  y, sr = librosa.load(audio_file_path, sr=None)

  print(f"The sampling rate of {audio_file_path} is {sr} Hz")

except Exception as e:
    print(f"Error loading or processing audio file: {e}")
