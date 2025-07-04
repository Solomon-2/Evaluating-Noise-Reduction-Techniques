To Do:
    1. Modularize every part of the pipeline
    2. Create a main.py file to run everything

Attempt at Running the full pipeline
    1. Lengthening an audio file by replicating it on itself.
    2. Adding apnea events. --Gaussian noise
    3. Running the detector on non-denoised audio to establish ground truth events.
    4. Applying spectral subtraction to the replication with the apnea detector.
    5. Applying Wiener filtering to the detector.
    6. 


7/3/2025
## Activity Log: Apnea Detection Evaluation Pipeline

- Generated synthetic data by injecting apnea (silence as Gaussian noise) into clean sleep audio.
- Ran `apnea_detector.py` on raw (non-denoised) audio to establish ground truth apnea events and saved results to CSV.
- Mixed in real-world noise (e.g., vacuum cleaner, cat) from ESC-50 to simulate challenging conditions.

- Processed the noisy audio with denoising algorithms and saved the denoised files.
- Ran `apnea_detector.py` on denoised audio files and saved detected events to another CSV.
- Used `compare_sensitivity.py` to compare ground truth and detected events, computing sensitivity for each file and overall.
- Confirmed pipeline is robust and reproducible; next steps include adding more noise/event variation and additional evaluation metrics.

8/4/2025
## Activity Log:
- 
    