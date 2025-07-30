# Research Proposal: Sleep Apnea Detection with Denoising Methods

## Title
Comparative evaluation of denoising methods for preserving sleep apnea detection accuracy under controlled noise conditions simulating smartphone recording environments

## Research Question
Which audio preprocessing denoising method (spectral subtraction, Wiener filtering, or deep learning-based denoisers) best maintains clinical-grade sleep apnea detection performance when applied to artificially degraded audio signals that simulate smartphone recording conditions?

## What We Want to Find Out
1. **Primary Objective:** Determine which denoising method best preserves sleep apnea detection accuracy when clinical-quality audio is degraded with controlled noise
2. **Signal Quality vs Detection Performance:** Establish whether improved signal quality metrics (SNR, PESQ) correlate with better detection accuracy
3. **Robustness Analysis:** Identify which denoising methods are most robust across different noise types and intensity levels
4. **Feature Stability:** Understand how different denoising methods affect the reliability of acoustic biomarkers used for apnea detection
5. **Practical Recommendations:** Provide evidence-based recommendations for smartphone-based sleep monitoring applications

## Methodology Overview

### Data Source
- **Clinical Data:** PhysioNet sleep study recordings (EDF files) from 23 patients
- **Audio Duration:** ~115 hours total (5 hours average per patient)
- **Ground Truth:** Apnea event annotations from accompanying RML files
- **Audio Channel:** Microphone recordings during sleep

### Experimental Design
1. **Training Phase:**
   - Extract acoustic features from clean clinical audio (baseline "gold standard")
   - Train classification model on clean features and apnea labels
   - Features: RMS energy, MFCCs 1-13, spectral centroid, bandwidth, rolloff

2. **Testing Phase:**
   - Add controlled noise to clean audio (ESC-50 database: vacuum cleaner, cat sounds, door creaks)
   - Apply denoising methods: Spectral subtraction, Wiener filtering, deep learning denoisers
   - Extract same features from denoised audio
   - Test trained model on denoised features

3. **Evaluation Metrics:**
   - **Detection Performance:** F1-score, sensitivity, specificity, AUC-ROC
   - **Signal Quality:** SNR improvement, spectral distortion
   - **Robustness:** Performance degradation and recovery rates across noise conditions

### Denoising Methods to Compare
- Spectral Subtraction
- Wiener Filtering  
- Log-MMSE
- Deep learning-based denoisers (SpeechBrain, DeepFilterNet)

### Expected Outcomes
- Identification of optimal denoising method for smartphone-based sleep apnea detection
- Understanding of trade-offs between signal quality improvement and detection accuracy
- Evidence-based recommendations for real-world smartphone app deployment

## Preliminary Study and Methodological Refinements

### Initial Dataset Validation (2-Patient Pilot Study)

#### Dataset Characteristics
A preliminary validation study was conducted using a 2-patient subset (30,763 audio frames, 8.48% apnea prevalence) to validate the feature extraction pipeline and assess baseline model performance. The dataset exhibited:

- **Good data quality:** No missing values, no infinite values, balanced patient distribution
- **Appropriate class distribution:** 8.48% apnea rate (patient_01: 10.25%, patient_02: 6.81%)
- **Sufficient volume:** ~2 hours total duration, ~15,000 frames per patient

#### Critical Finding: Temporal Window Limitations

The initial approach using **1-second audio frames** revealed a fundamental methodological limitation:

**Feature-Target Correlations Analysis:**
- Maximum correlation: 0.0634 (clean_mfcc_11)
- All feature correlations < 0.07
- Baseline Random Forest F1-score: 0.24

**Root Cause Analysis:**
The extremely low feature-target correlations indicate that 1-second temporal windows are insufficient for capturing the physiological characteristics of sleep apnea events. This finding aligns with clinical understanding of apnea pathophysiology:

1. **Apnea Event Duration:** Clinical apnea events typically last 10+ seconds
2. **Breathing Pattern Context:** Apnea detection requires analysis of breathing rhythm changes over extended periods
3. **Temporal Dependencies:** Short frames miss the cessation-resumption patterns characteristic of apnea events

#### Methodological Refinements Based on Pilot Study

**Revised Frame Duration:**
- **Original:** 1-second frames
- **Revised:** 30-60 second frames to capture complete breathing cycles and apnea event patterns

**Enhanced Feature Engineering:**
- **Temporal Features:** Breathing rate variability, pause detection metrics
- **Rhythm Analysis:** Respiratory pattern regularity over extended windows
- **Context-Aware Features:** Before/during/after apnea event characteristics

**Alternative Modeling Approaches:**
- **Sequence Models:** LSTM/CNN architectures for temporal pattern recognition
- **Sliding Windows:** Overlapping analysis windows for improved temporal resolution
- **Multi-Scale Analysis:** Hierarchical feature extraction at multiple temporal scales

### Impact on Research Design

This preliminary finding significantly strengthens the research methodology by:

1. **Evidence-Based Parameter Selection:** Frame duration now grounded in empirical validation rather than arbitrary choice
2. **Clinical Relevance:** Temporal windows aligned with physiological characteristics of apnea events
3. **Improved Baseline Performance:** Expected substantial improvement in baseline model performance before denoising evaluation
4. **Methodological Rigor:** Demonstrates systematic validation and iterative refinement approach

### Updated Experimental Timeline

**Phase 1:** Implement revised temporal windowing (30-60 second frames)
**Phase 2:** Validate improved approach with 5-patient subset
**Phase 3:** Scale to full 23-patient local dataset upon validation
**Phase 4:** Noise injection and denoising method comparison
**Phase 5:** Full-scale evaluation with 212-patient PhysioNet dataset

This methodological refinement ensures that subsequent denoising method comparisons are conducted on a foundation of clinically meaningful apnea detection performance, thereby providing more reliable and actionable insights for smartphone-based sleep monitoring applications.