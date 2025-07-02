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