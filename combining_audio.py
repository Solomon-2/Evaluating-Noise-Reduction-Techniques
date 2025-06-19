import librosa
import numpy as np
import soundfile as sf

y_voice, sr_voice = librosa.load('clean_voice.wav', sr=16000)
y_noise, sr_noise = librosa.load('noise.wav', sr=16000)

# Change the length of noise (since it's longer) to match voice
y_noise_truncated = y_noise[:len(y_voice)]

y_mixed=y_voice+y_noise_truncated

#Normalizing the audio to prevent severe suffering of listener😂
max_ampl_val = np.max(np.abs(y_mixed))
if max_ampl_val > 0:
  y_mixed_normalized = y_mixed / max_ampl_val

else:
  y_mixed_normalized=y_mixed

#Save the combined audio
sf.write("combined_audio.wav", y_mixed_normalized, 16000)
