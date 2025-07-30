# Research Proposal: A Multi-Dimensional Evaluation of Denoising Methods for Smartphone-Based Sleep Apnea Detection

  

# Abstract

_To be written after results are finalized. Placeholder for summary of research question, methodology, findings, and key insight._

## 1. Introduction

  

Sleep apnea is a prevalent sleep disorder affecting an estimated 1 billion people globally, with as many as 80% of cases remaining undiagnosed. The gold standard for diagnosis, polysomnography (PSG), is expensive, inaccessible, and requires an overnight stay in a clinical facility. This creates a significant barrier to diagnosis for a large portion of the population. The widespread availability of smartphones presents a unique opportunity for accessible, low-cost, and non-invasive pre-screening for sleep apnea. However, audio recordings made in uncontrolled home environments are inevitably corrupted by environmental noise, which can severely degrade the performance of automated detection algorithms.

  

While many studies have focused on developing sophisticated deep learning models for apnea detection, they often rely on clean, clinical-grade audio and require computational resources that are infeasible for real-time, on-device processing on a smartphone. There is a critical research gap in systematically evaluating the trade-offs of various practical denoising methods in the context of a computationally efficient, classifier-based detection pipeline suitable for mobile health (mHealth) applications.

  

This research aims to fill that gap by conducting a rigorous, multi-dimensional evaluation of traditional and deep-learning-based denoising techniques. The primary research question is: **"Which audio preprocessing denoising method provides the optimal balance of detection accuracy, signal quality improvement, computational efficiency, and biomarker preservation for smartphone-based sleep apnea detection using Random Forest classifiers?"**

  

By moving beyond simple accuracy metrics to include computational cost, signal quality, and the preservation of critical breathing-related acoustic features, this study will provide evidence-based, practical recommendations for developing robust and effective sleep apnea screening tools on mobile devices.

  ## Related Work

Several studies have addressed the feasibility of detecting sleep apnea through acoustic analysis. Nakano et al. (2004) demonstrated that snoring sounds could be analyzed to distinguish obstructive sleep apnea events from simple snoring, showing the early viability of audio as a diagnostic tool. More recent research by Hassan et al. (2019) applied convolutional neural networks (CNNs) on spectrogram images of breathing audio to detect apneic episodes with high accuracy in controlled environments.

Other approaches have incorporated time-domain features like zero-crossing rate (ZCR) and root mean square (RMS) energy, alongside frequency-domain features such as Mel-frequency cepstral coefficients (MFCCs), to represent breathing dynamics (Mesquita et al., 2017). These features have been widely adopted in machine learning pipelines for apnea detection, particularly when using traditional classifiers such as support vector machines (SVMs), k-nearest neighbors (KNN), or ensemble models.

Noise interference, however, remains a major concern. Studies such as Chen et al. (2018) have shown that real-world environmental noise significantly reduces classifier accuracy. Attempts to denoise signals prior to classification include applying spectral subtraction, Wiener filtering, and, more recently, deep learning approaches like SEGAN (Speech Enhancement GAN) and Demucs. While these methods have been evaluated primarily for speech enhancement, their impact on bio-acoustic biomarkers specific to apnea detection remains underexplored.

To the best of our knowledge, no prior work has systematically compared the downstream effects of different denoising strategies on apnea-relevant audio features. This study is motivated by this gap, aiming to determine whether denoising enhances or impairs the discriminatory value of features used for apnea classification in smartphone-based settings.
## 2. Methodology

  

This study employs a multi-phase methodology designed to systematically evaluate the impact of environmental noise on sleep apnea detection and the efficacy of various denoising techniques. The process begins with data acquisition and preprocessing, followed by robust model training and validation, and culminates in a controlled noise simulation experiment.

  

### 2.1. Dataset and Ground Truth Acquisition

  

The data for this research was sourced from the PhysioNet Apnea-ECG database, a public repository of polysomnography (PSG) recordings. For this study, we utilized the full overnight recordings of 23 patients, each containing multi-channel physiological signals stored in the European Data Format (EDF). The crucial ground truth for apnea events was provided by corresponding annotation files in Respiratory Markup Language (RML), which contain physician-annotated timestamps for specific respiratory events, including Obstructive Apnea, Central Apnea, Mixed Apnea, and Hypopnea.

  

Audio data was extracted exclusively from the 'Mic' (microphone) channel of the EDF files, simulating a single-channel audio recording environment akin to that of a smartphone placed near a patient.

  

### 2.2. Audio Preprocessing and Feature Engineering

  

To prepare the raw audio for machine learning analysis, a comprehensive preprocessing and feature engineering pipeline was developed.

  

#### 2.2.1. Signal Preprocessing

  

The raw audio signal from the EDF files, originally sampled at 48kHz, was downsampled to 16kHz. This reduces computational complexity while retaining the necessary frequency information for analyzing breathing sounds. The continuous audio stream was then segmented into 30-second windows with a 50% (15-second) overlap. This window size was empirically chosen to be long enough to capture the physiological characteristics of apnea events, which typically last 10 seconds or more, while the overlap ensures that events occurring at the boundary of a window are not missed.

  

#### 2.2.2. Feature Extraction

  

For each 30-second audio window, a set of 27 acoustic and temporal features was extracted. These features were specifically engineered to quantify the characteristics of breathing patterns rather than speech. They include:

  

*   **Basic Acoustic Features:** Root Mean Square (RMS) energy, Zero-Crossing Rate (ZCR), Spectral Centroid, Spectral Bandwidth, and Spectral Rolloff. The RMS energy, for instance, is a measure of the signal's amplitude and is calculated as:

  

    $$

    \text{RMS} = \sqrt{\frac{1}{N} \sum_{n=1}^{N} x(n)^2}

    $$

  

    where $N$ is the number of samples in the window and $x(n)$ is the value of the $n$-th sample.

  

*   **Mel-Frequency Cepstral Coefficients (MFCCs):** The mean and standard deviation of the first 8 MFCCs were extracted to represent the spectral shape of the audio.

  

*   **Temporal Breathing Pattern Features:** To capture the rhythm and consistency of breathing, variability metrics (standard deviation) of RMS and ZCR were calculated over smaller 5-second sub-windows. This allowed for the quantification of breathing regularity.

  

*   **Silence and Pause Metrics:** Features such as the silence ratio, average pause duration, and maximum pause duration were engineered to detect interruptions in breathing, which are hallmarks of apneic events.

  

#### 2.2.3. Ground Truth Labeling

  

Each 30-second window was assigned a binary label (1 for apnea, 0 for normal breathing) based on the physician-annotated RML files. A proportion-based labeling strategy was employed: if the total duration of annotated apnea or hypopnea events within a window exceeded a 10% threshold (i.e., 3 seconds), the window was labeled as 'apnea' (1). Otherwise, it was labeled as 'normal' (0).

  

### 2.3. Apnea Detection Model and Validation

  

A Random Forest classifier was selected as the primary model for this study due to its robustness to class imbalance, high interpretability, and strong performance on tabular data.

  

To ensure the clinical validity and generalizability of the model, a rigorous **patient-based cross-validation** strategy was implemented using `GroupKFold`. The dataset was split into training and testing sets based on patient IDs, guaranteeing that no audio from a single patient appeared in both the training and testing sets within a fold. This prevents data leakage and simulates the real-world scenario of testing a model on a new, unseen patient.

  

The model trained on the clean, preprocessed audio data established our baseline performance, achieving an **F1-score of 0.758**, a **sensitivity of 76.9%**, and a **specificity of 76.6%**. This performance is competitive with existing literature and serves as the benchmark against which all noisy and denoised audio will be compared.

  

### 2.4. Environmental Noise Simulation

  

To evaluate the model's robustness and create a testbed for denoising methods, a systematic noise injection protocol was designed.

  

#### 2.4.1. Noise Source Selection

  

Environmental noise samples were selected from the ESC-50 dataset. Five distinct noise categories were chosen to represent a variety of potential real-world recording conditions:

1.  **Vacuum Cleaner:** High-frequency, stationary mechanical noise.

2.  **Cat:** Non-stationary, organic, and unpredictable animal sounds.

3.  **Door Creaks:** Low-frequency, transient structural noise.

4.  **Crying Baby:** Non-stationary human vocalizations, which can spectrally overlap with breathing.

5.  **Coughing:** Respiratory sounds that directly interfere with and mimic breathing patterns, representing the most challenging condition.

  

#### 2.4.2. Controlled Noise Injection

  

The selected noises were mixed with the clean audio samples at three specific Signal-to-Noise Ratios (SNRs): **5 dB** (poor quality), **10 dB** (moderate quality), and **15 dB** (good quality). The SNR is a measure of the ratio of the power of the signal to the power of the noise, calculated in decibels (dB) as:

  

$$

\text{SNR}_{\text{dB}} = 10 \log_{10} \left( \frac{P_{\text{signal}}}{P_{\text{noise}}} \right)

$$

  

where $P_{\text{signal}}$ is the power of the clean audio and $P_{\text{noise}}$ is the power of the noise. This process resulted in a comprehensive test matrix of 45 unique noise conditions (3 patients × 5 noise types × 3 SNR levels), forming the basis for the subsequent degradation and denoising evaluation.