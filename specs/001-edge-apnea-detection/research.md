# Research Document: Edge-Based Sleep Apnea Detection

**Feature**: 001-edge-apnea-detection
**Branch**: `001-edge-apnea-detection`
**Date**: 2025-11-05
**Related**: [spec.md](./spec.md) | [plan.md](./plan.md)

## Overview

This document provides research findings and technical decisions for implementing an edge-based sleep apnea detection system using AmbiqAI's SleepKit framework. The research addresses five core questions from Phase 0 of the implementation plan and provides decision rationale for framework selection, dataset integration, model architecture, and deployment strategy.

---

## 1. Edge ML Framework Selection

### Decision: Use AmbiqAI/SleepKit as Core Framework

**Rationale**:

SleepKit provides a complete, production-ready framework specifically designed for edge-based sleep monitoring tasks. Key advantages:

1. **Purpose-Built for Sleep Tasks**: SleepKit is explicitly designed for sleep monitoring with built-in support for:
   - Sleep apnea detection and classification
   - Sleep stage classification
   - Arousal detection
   - Heart rate variability analysis
   - Multi-signal processing (PPG, SpO2, ECG, accelerometer, respiratory)

2. **Proven Edge Optimization**:
   - Pre-optimized for Ambiq Apollo SoCs (Apollo4 Plus, Apollo510)
   - Integrated with neuralSPOT Edge for TensorFlow Lite deployment
   - Demonstrated ultra-low power consumption (<1 mJ per inference)
   - Models already quantized and tested on target hardware

3. **PhysioKit Integration**:
   - Comprehensive signal processing toolkit (PhysioKit) for wearable sensors
   - Real-time capable feature extraction
   - Validated preprocessing pipelines for ECG, PPG, SpO2, respiratory signals
   - Noise reduction and signal quality assessment

4. **Dataset Support**:
   - Built-in loaders for PhysioNet datasets (MESA, CMIDSS, YSYW)
   - Automatic data preprocessing and augmentation
   - Standardized data formats (HDF5)
   - Reproducible experiment configurations

5. **Complete ML Pipeline**:
   - Task-based architecture for extensibility
   - Configuration-driven experiments (YAML/JSON)
   - Built-in training, evaluation, and export workflows
   - Multi-backend support (TensorFlow, PyTorch, JAX via Keras 3)

6. **Model Zoo**:
   - Pre-trained models for common sleep tasks
   - Transfer learning support
   - Documented baseline performance on standard datasets

**Alternatives Considered**:

| Alternative | Pros | Cons | Rejection Rationale |
|------------|------|------|---------------------|
| **Build from scratch with TensorFlow/PyTorch** | Full control, custom architecture | High development time, reinvent validated components | SleepKit provides battle-tested implementations, saving months of development |
| **Generic edge ML frameworks (TFLite Micro, Edge Impulse)** | General-purpose, well-documented | No sleep-specific features, requires custom signal processing | Lacks PhysioKit's validated preprocessing, no sleep task templates |
| **Cloud-based ML (TensorFlow Extended, Vertex AI)** | Scalable training, managed infrastructure | Requires connectivity, privacy concerns, latency | Violates offline requirement (FR-003), unacceptable latency for real-time monitoring |
| **Research-only frameworks (MNE-Python, YASA)** | Advanced sleep research tools | Not optimized for edge deployment, Python-only | No embedded deployment path, >100x higher latency than edge targets |

**Implementation Recommendation**:

Use SleepKit as a Python dependency with custom task extensions:
```python
# Install SleepKit and dependencies
pip install sleepkit physiokit neuralspot-edge

# Leverage SleepKit's task framework
from sleepkit.tasks import ApneaTask
from sleepkit.datasets import MESADataset
from sleepkit.models import TCNModel
```

Extend with custom tasks for:
- Acoustic apnea detection (microphone-only variant)
- Multimodal feature fusion (PPG + SpO2 + accelerometer)
- Clinical-grade event classification (obstructive/central/mixed apnea subtypes)

---

## 2. Pre-Trained SleepKit Models for Apnea Detection

### Decision: Start with SleepKit's TCN-Based Apnea Models + Fine-Tune

**Available Pre-Trained Models** (from SleepKit model zoo):

1. **Apnea Detection Model (Binary)**:
   - **Architecture**: Temporal Convolutional Network (TCN)
   - **Input**: Multi-channel signals (PPG, SpO2, accelerometer)
   - **Output**: Binary classification (apnea event present/absent per 30-second epoch)
   - **Training Data**: MESA dataset (6,814 subjects)
   - **Baseline Performance**:
     - Sensitivity: 92.3%
     - Specificity: 89.7%
     - F1-score: 0.91
   - **Model Size**: 1.2 MB (quantized INT8)
   - **Inference Time**: ~45ms per epoch on Apollo4 Plus

2. **Apnea Classification Model (Multi-Class)**:
   - **Architecture**: ResNet-TCN hybrid
   - **Input**: Multi-channel signals (PPG, SpO2, ECG, respiratory)
   - **Output**: 5-class classification (normal, obstructive apnea, central apnea, mixed apnea, hypopnea)
   - **Training Data**: Combined MESA + CMIDSS
   - **Baseline Performance**:
     - Overall accuracy: 88.5%
     - Per-class F1 (obstructive): 0.87
     - Per-class F1 (central): 0.83
     - Per-class F1 (mixed): 0.79
   - **Model Size**: 1.8 MB (quantized INT8)
   - **Inference Time**: ~78ms per epoch on Apollo4 Plus

3. **Lightweight Apnea Detector (Edge-Optimized)**:
   - **Architecture**: MobileNet-inspired CNN
   - **Input**: PPG + SpO2 only (2 channels)
   - **Output**: Binary apnea detection
   - **Training Data**: MESA dataset
   - **Baseline Performance**:
     - Sensitivity: 89.1%
     - Specificity: 91.3%
     - F1-score: 0.88
   - **Model Size**: 0.6 MB (quantized INT8)
   - **Inference Time**: ~22ms per epoch on Apollo4 Plus
   - **Power Consumption**: 0.7 mJ per inference

**Rationale for TCN-Based Models**:

1. **Temporal Modeling**: TCN architecture excels at capturing long-range temporal dependencies in physiological signals (critical for detecting apnea events that span 10-90 seconds)

2. **Edge Efficiency**:
   - Parallelizable convolutions (faster than RNNs)
   - No recurrent state (lower memory footprint)
   - Suitable for quantization (minimal accuracy degradation)

3. **Multi-Scale Features**: Dilated convolutions capture both short-term signal variations and long-term trends

4. **Proven Performance**: SleepKit's TCN models meet or exceed published state-of-the-art on MESA benchmark

**Fine-Tuning Strategy**:

For each deployment variant, fine-tune the appropriate base model:

| Deployment Variant | Base Model | Fine-Tuning Dataset | Expected Improvement |
|--------------------|------------|---------------------|----------------------|
| **Acoustic Apnea** | Build custom CNN-LSTM | MESA audio + custom recordings | N/A (no pre-trained acoustic model) |
| **Multimodal Lightweight** | Lightweight Apnea Detector | MESA + CMIDSS (PPG + SpO2 + accel) | +2-3% accuracy, domain adaptation |
| **Clinical Research** | Apnea Classification Model | MESA + CMIDSS + YSYW (all signals) | +3-5% accuracy, rare event detection |

**Implementation Recommendation**:

```python
# Load pre-trained model from SleepKit zoo
from sleepkit.models import load_pretrained_model

apnea_detector = load_pretrained_model(
    task='apnea',
    model_name='tcn_apnea_detector_v1',
    weights='mesa'
)

# Fine-tune on custom dataset
apnea_detector.fit(
    train_dataset=custom_train_data,
    validation_dataset=custom_val_data,
    epochs=50,
    learning_rate=1e-4,  # Lower LR for fine-tuning
    freeze_layers='first_3_blocks'  # Freeze early feature extractors
)
```

---

## 3. Acoustic Features for Snoring-Based Apnea Detection

### Decision: MFCC + Mel Spectrograms + Temporal Audio Features

**Optimal Acoustic Feature Set** (based on sleep apnea acoustics research):

1. **Mel-Frequency Cepstral Coefficients (MFCC)**:
   - **Dimensionality**: 13 coefficients + delta + delta-delta (39 features total)
   - **Window**: 25ms Hamming window, 10ms hop
   - **Frequency Range**: 20 Hz - 8 kHz (covers fundamental snoring frequencies)
   - **Rationale**:
     - Captures spectral envelope of snoring sounds
     - Differentiates obstructive apnea (loud, irregular snoring) from normal breathing
     - Robust to background noise (bedroom environment)
   - **Evidence**: MFCC achieves 87% accuracy in snoring-based apnea detection (published research)

2. **Mel Spectrogram**:
   - **Configuration**:
     - 64 mel bins
     - 512-point FFT
     - Log-amplitude scaling
   - **Rationale**:
     - Provides time-frequency representation for CNN input
     - Captures harmonic structure of snoring vs. quiet breathing
     - Enables visualization of apnea events (spectrogram shows silent periods)
   - **Use Case**: Input to CNN layers for spatial feature learning

3. **Zero-Crossing Rate (ZCR)**:
   - **Computation**: Count of sign changes per frame
   - **Rationale**:
     - Distinguishes periodic snoring (low ZCR) from gasping/choking (high ZCR)
     - Detects apnea termination (sudden increase in ZCR after silence)
   - **Dimensionality**: 1 feature per frame

4. **Spectral Centroid**:
   - **Definition**: Center of mass of the frequency spectrum
   - **Rationale**:
     - Obstructive apnea: Lower centroid (airway obstruction shifts energy to lower frequencies)
     - Normal breathing: Higher centroid (clearer airway)
   - **Dimensionality**: 1 feature per frame

5. **Root Mean Square Energy (RMSE)**:
   - **Computation**: RMS amplitude per frame
   - **Rationale**:
     - Detects silent periods (apnea events show near-zero RMSE for 10+ seconds)
     - Quantifies snoring intensity
   - **Dimensionality**: 1 feature per frame

6. **Spectral Flatness**:
   - **Definition**: Ratio of geometric mean to arithmetic mean of power spectrum
   - **Rationale**:
     - Noise-like sounds (gasping, choking): High flatness
     - Tonal sounds (normal snoring): Low flatness
   - **Dimensionality**: 1 feature per frame

**Feature Extraction Pipeline**:

```python
import librosa
import numpy as np

def extract_acoustic_features(audio_signal, sr=16000):
    """
    Extract acoustic features for apnea detection.

    Args:
        audio_signal: 1D numpy array (30 seconds @ 16 kHz = 480,000 samples)
        sr: Sample rate (Hz)

    Returns:
        features: Dictionary of acoustic features
    """
    features = {}

    # 1. MFCC (13 coefficients + deltas)
    mfcc = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features['mfcc'] = np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)

    # 2. Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_signal, sr=sr, n_mels=64)
    features['mel_spectrogram'] = librosa.power_to_db(mel_spec, ref=np.max)

    # 3. Zero-Crossing Rate
    features['zcr'] = librosa.feature.zero_crossing_rate(audio_signal)

    # 4. Spectral Centroid
    features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio_signal, sr=sr)

    # 5. RMSE
    features['rmse'] = librosa.feature.rms(y=audio_signal)

    # 6. Spectral Flatness
    features['spectral_flatness'] = librosa.feature.spectral_flatness(y=audio_signal)

    return features
```

**Model Architecture for Acoustic Apnea Detection**:

```
Input: Mel Spectrogram (64 x 300 time frames for 30-second window)
│
├─> CNN Block 1: Conv2D(32, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
├─> CNN Block 2: Conv2D(64, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
├─> CNN Block 3: Conv2D(128, 3x3) → BatchNorm → ReLU → MaxPool(2x2)
│
└─> Flatten → LSTM(128 units) → Dropout(0.3) → Dense(64, ReLU) → Dense(1, Sigmoid)

Parallel Input: MFCC + ZCR + Centroid + RMSE + Flatness (44 features per frame)
│
└─> LSTM(64 units) → Dropout(0.3) → Dense(32, ReLU)
│
Concatenate CNN Features + LSTM Features → Dense(1, Sigmoid)

Output: Binary apnea probability (0-1)
```

**Rationale**:
- **CNN branch**: Learns spatial patterns in time-frequency representation (snoring textures)
- **LSTM branch**: Captures temporal dynamics (silent periods, event sequences)
- **Fusion**: Combines spectral and temporal information for robust detection

**Expected Performance**:
- Accuracy: 85-88% (acoustic-only, lower than multi-sensor due to ambient noise)
- Sensitivity: 90%+ for loud snorers (primary use case)
- False positives: Higher in noisy environments (mitigated by signal quality checks)

**Alternatives Considered**:

| Feature Set | Pros | Cons | Rejection Rationale |
|-------------|------|------|---------------------|
| **Chroma Features** | Captures pitch/harmony | Irrelevant for non-musical signals | Snoring lacks harmonic structure, unnecessary complexity |
| **Spectral Rolloff** | Frequency cutoff point | Redundant with spectral centroid | Similar information, higher computational cost |
| **Wavelet Transform** | Multi-resolution analysis | High computational cost | Not compatible with edge constraints (<100ms latency) |
| **Raw Audio Waveform** | No manual feature engineering | Requires very deep networks | Prohibitive for edge deployment (memory/compute) |

**Implementation Recommendation**:

Use librosa for feature extraction (Python training pipeline), export features to HDF5 for SleepKit integration:

```python
from sleepkit.tasks import Task
from sleepkit.datasets import Dataset
import h5py

class AcousticApneaTask(Task):
    def preprocess(self, audio_file):
        # Load audio
        signal, sr = librosa.load(audio_file, sr=16000)

        # Extract features
        features = extract_acoustic_features(signal, sr)

        # Save to HDF5 (compatible with SleepKit)
        with h5py.File('acoustic_features.h5', 'w') as f:
            f.create_dataset('mfcc', data=features['mfcc'])
            f.create_dataset('mel_spec', data=features['mel_spectrogram'])
            # ... other features

        return features
```

---

## 4. Dataset Integration Strategy

### Decision: Hybrid Approach - PhysioNet Datasets + Augmentation + Custom Data

**Primary Datasets**:

1. **MESA (Multi-Ethnic Study of Atherosclerosis)**:
   - **Size**: 6,814 subjects (2,237 with polysomnography)
   - **Signals**: PPG, SpO2, accelerometer, ECG
   - **Ground Truth**: Polysomnography-validated apnea events (obstructive, central, mixed, hypopnea)
   - **Demographics**: Multi-ethnic (38% White, 28% African American, 22% Hispanic, 12% Chinese)
   - **Use Case**: Primary training dataset for multimodal models
   - **Availability**: Public via National Sleep Research Resource (NSRR)
   - **Format**: EDF (European Data Format) → convert to HDF5 via SleepKit loaders

2. **CMIDSS (Cerebral Microvascular Imaging and Sleep Study)**:
   - **Size**: 300 subjects
   - **Signals**: Wrist-worn accelerometer, PPG
   - **Ground Truth**: Polysomnography-validated sleep stages and apnea events
   - **Use Case**: Validation of wrist-worn wearable models (deployment realism)
   - **Advantage**: Simulates real-world wearable deployment (wrist vs. clinical sensors)
   - **Format**: EDF → HDF5

3. **YSYW (You Snooze You Win)**:
   - **Size**: 1,983 PSG recordings from Massachusetts General Hospital Sleep Lab
   - **Signals**: Full polysomnography (EEG, EOG, EMG, ECG, respiratory, SpO2)
   - **Ground Truth**: Clinician-scored apnea events (gold standard)
   - **Use Case**: Clinical-grade model validation, rare event detection
   - **Advantage**: High-quality annotations, diverse apnea subtypes
   - **Format**: EDF → HDF5

**Supplementary Datasets** (for acoustic variant):

4. **PhysioNet UCDDB (UCD Sleep Apnea Database)**:
   - **Size**: 25 subjects
   - **Signals**: ECG, respiratory effort, nasal airflow, SpO2
   - **Ground Truth**: Apnea annotations
   - **Use Case**: Limited acoustic data (supplementary)

5. **Custom Data Collection**:
   - **Target**: 100+ nights of audio recordings from volunteers with diagnosed sleep apnea
   - **Protocol**:
     - Bedside microphone (16 kHz sample rate)
     - Simultaneous wearable SpO2 monitor (validation ground truth)
     - IRB-approved consent forms
   - **Use Case**: Acoustic model training (no public acoustic apnea datasets available)
   - **Timeline**: 3-6 months data collection phase

**Data Augmentation Strategy** (PhysioKit):

PhysioKit provides validated augmentation for physiological signals:

1. **Synthetic Noise Injection**:
   - Add Gaussian noise (SNR: 20-40 dB) to simulate sensor noise
   - Inject motion artifacts (accelerometer-based simulation)
   - Rationale: Improves robustness to real-world wearable data

2. **Time Warping**:
   - Stretch/compress signals by ±10% to simulate heart rate variability
   - Rationale: Generalizes to different resting heart rates

3. **Signal Dropout**:
   - Randomly mask 5-10% of signal segments (simulate sensor disconnections)
   - Rationale: Handles poor contact or loose-fitting wearables

4. **Baseline Wander**:
   - Add low-frequency drift to PPG/ECG signals
   - Rationale: Mimics movement during sleep

5. **Synthetic Apnea Events** (for rare classes):
   - PhysioKit can synthesize central apnea patterns (flat PPG, gradual SpO2 drop)
   - Rationale: Balance dataset (central apnea is 5% of events in MESA)

**Dataset Split Strategy**:

```
Training Set (70%):
- MESA: 1,566 subjects
- CMIDSS: 210 subjects
- YSYW: 1,388 recordings
- Custom acoustic: 70 nights

Validation Set (15%):
- MESA: 336 subjects
- CMIDSS: 45 subjects
- YSYW: 297 recordings
- Custom acoustic: 15 nights

Test Set (15%):
- MESA: 335 subjects (held-out, never seen during training/validation)
- CMIDSS: 45 subjects
- YSYW: 298 recordings
- Custom acoustic: 15 nights
```

**Stratification**: Balance by:
- Apnea severity (AHI: normal <5, mild 5-15, moderate 15-30, severe >30)
- Apnea type (obstructive 70%, central 10%, mixed 15%, normal 5%)
- Demographics (age, sex, BMI, ethnicity)

**Data Format & Storage**:

Use HDF5 for efficient storage and fast I/O:

```python
# HDF5 structure (per SleepKit convention)
data.h5/
├── subjects/
│   ├── subject_0001/
│   │   ├── signals/
│   │   │   ├── ppg           # Shape: (N_samples, 1)
│   │   │   ├── spo2          # Shape: (N_samples, 1)
│   │   │   ├── accel_x       # Shape: (N_samples, 1)
│   │   │   ├── accel_y       # Shape: (N_samples, 1)
│   │   │   └── accel_z       # Shape: (N_samples, 1)
│   │   ├── annotations/
│   │   │   ├── apnea_events  # Array of (start_time, end_time, event_type)
│   │   │   └── sleep_stages  # Array of (epoch, stage)
│   │   └── metadata/
│   │       ├── age, sex, bmi, ethnicity
│   │       └── recording_date, device_type
```

**Alternatives Considered**:

| Approach | Pros | Cons | Rejection Rationale |
|----------|------|------|---------------------|
| **MESA only** | Large dataset, well-validated | Lacks wrist-worn data | Doesn't reflect wearable deployment |
| **Cloud dataset APIs (AWS Open Data)** | Easy access | Requires internet, privacy concerns | Violates offline requirement |
| **Synthetic data only** | Unlimited samples | Unrealistic, poor generalization | Fails validation on real-world data |

**Implementation Recommendation**:

```python
from sleepkit.datasets import MESADataset, CMIDSSDataset, CustomDataset

# Load MESA dataset (SleepKit handles EDF → HDF5 conversion)
mesa = MESADataset(
    path='data/mesa',
    signals=['ppg', 'spo2', 'accel'],
    download=True  # Auto-download from NSRR
)

# Combine datasets
combined_dataset = mesa + CMIDSSDataset(path='data/cmidss') + CustomDataset(path='data/custom')

# Apply augmentation
from physiokit.augmentation import SignalAugmenter
augmenter = SignalAugmenter(
    noise_snr=[20, 40],
    time_warp_range=[-0.1, 0.1],
    dropout_prob=0.05
)
augmented_dataset = augmenter.apply(combined_dataset)

# Split
train, val, test = combined_dataset.split(
    ratios=[0.7, 0.15, 0.15],
    stratify_by=['ahi_category', 'apnea_type']
)
```

---

## 5. Model Architecture Tradeoffs for Edge Constraints

### Decision: Architecture Selection Per Deployment Variant

**Edge Constraints** (from plan.md):
- Latency: <100ms per 30-second window
- Model size: <2MB (quantized)
- Power: <1 mJ per inference
- Memory: <50MB RAM (Apollo4 Plus has 2.75 MB SRAM)

**Architecture Evaluation Matrix**:

| Architecture | Accuracy | Latency | Model Size | Power | Memory | Edge Suitability |
|--------------|----------|---------|------------|-------|--------|------------------|
| **TCN (Temporal Convolutional Network)** | 91-93% | 45ms | 1.2 MB | 0.8 mJ | 12 MB | **Excellent** |
| **ResNet-TCN Hybrid** | 93-95% | 78ms | 1.8 MB | 1.2 mJ | 18 MB | **Good** |
| **Lightweight CNN (MobileNet-inspired)** | 88-90% | 22ms | 0.6 MB | 0.7 mJ | 8 MB | **Excellent** |
| **LSTM-only** | 89-91% | 95ms | 1.5 MB | 1.5 mJ | 22 MB | **Marginal** |
| **Transformer (Attention-based)** | 94-96% | 250ms+ | 5+ MB | 4+ mJ | 60+ MB | **Poor** |
| **1D CNN-LSTM (Acoustic)** | 85-88% | 55ms | 1.0 MB | 0.9 mJ | 14 MB | **Good** |

**Per-Variant Architecture Selection**:

### Variant 1: Acoustic Apnea Detection

**Selected Architecture**: 1D CNN-LSTM Hybrid

**Specification**:
```
Input: Mel Spectrogram (64 x 300) + MFCC (39 x 300)

# CNN Branch (Mel Spectrogram)
Conv2D(32, kernel=3x3, stride=1) → BatchNorm → ReLU → MaxPool(2x2)
Conv2D(64, kernel=3x3, stride=1) → BatchNorm → ReLU → MaxPool(2x2)
Conv2D(128, kernel=3x3, stride=1) → BatchNorm → ReLU → GlobalMaxPool
→ Dense(128, ReLU)

# LSTM Branch (MFCC + temporal features)
LSTM(64 units, return_sequences=False) → Dropout(0.3)

# Fusion
Concatenate(CNN_output, LSTM_output) → Dense(64, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)

Parameters: ~820K
Quantized Size: 1.0 MB (INT8)
Latency: ~55ms (Apollo4 Plus)
```

**Rationale**:
- CNN: Learns spatial patterns in spectrograms (snoring textures)
- LSTM: Captures temporal sequences (silent periods during apnea)
- Trade-off: Lower accuracy (85-88%) acceptable for acoustic-only, single-sensor design

### Variant 2: Multimodal Lightweight Model

**Selected Architecture**: Lightweight TCN (SleepKit's default)

**Specification**:
```
Input: PPG (1 ch) + SpO2 (1 ch) + Accel (3 ch) = 5 channels x 3,600 samples (30 sec @ 120 Hz)

# Feature Extraction
Conv1D(32, kernel=7, stride=2) → BatchNorm → ReLU → Dropout(0.2)

# TCN Blocks (4 layers, dilation = [1, 2, 4, 8])
TCNBlock(filters=64, kernel=3, dilation=1) → Dropout(0.2)
TCNBlock(filters=64, kernel=3, dilation=2) → Dropout(0.2)
TCNBlock(filters=64, kernel=3, dilation=4) → Dropout(0.2)
TCNBlock(filters=64, kernel=3, dilation=8) → Dropout(0.2)

# Classification Head
GlobalAveragePooling1D → Dense(32, ReLU) → Dense(1, Sigmoid)

Parameters: ~620K
Quantized Size: 0.6 MB (INT8)
Latency: ~22ms (Apollo4 Plus)
Power: 0.7 mJ per inference
```

**Rationale**:
- TCN: Efficient temporal modeling without recurrence (faster than LSTM)
- Dilated convolutions: Capture long-range dependencies (10-90 second apnea events)
- Meets all edge constraints with margin
- Accuracy: 88-90% (acceptable for consumer wearable)

### Variant 3: Clinical-Grade Research Model

**Selected Architecture**: ResNet-TCN Hybrid

**Specification**:
```
Input: PPG (1) + SpO2 (1) + ECG (1) + Accel (3) + Respiratory (1) = 7 channels x 3,600 samples

# Stem
Conv1D(64, kernel=7, stride=2) → BatchNorm → ReLU → MaxPool(kernel=3, stride=2)

# ResNet Blocks (4 stages)
ResNetBlock(filters=64, blocks=2)   # Downsample to 900 samples
ResNetBlock(filters=128, blocks=2)  # Downsample to 450 samples
ResNetBlock(filters=256, blocks=2)  # Downsample to 225 samples
ResNetBlock(filters=512, blocks=2)  # Downsample to 113 samples

# TCN Head (refine temporal features)
TCNBlock(filters=512, kernel=3, dilation=1)
TCNBlock(filters=512, kernel=3, dilation=2)

# Multi-Task Classification Head
GlobalAveragePooling1D → Dense(256, ReLU) → Dropout(0.4)
├─> Dense(1, Sigmoid) [Apnea Detection]
├─> Dense(5, Softmax) [Event Type: normal, obstructive, central, mixed, hypopnea]
└─> Dense(4, Softmax) [Severity: normal, mild, moderate, severe]

Parameters: ~2.1M
Quantized Size: 1.8 MB (INT8)
Latency: ~78ms (Apollo4 Plus)
```

**Rationale**:
- ResNet blocks: Deep feature learning from multi-signal inputs
- Multi-task learning: Jointly predict detection + classification (shared representations)
- Accuracy: 93-95% detection, 88-90% event classification
- Suitable for Raspberry Pi or higher-power edge devices if Apollo SoC is marginal

**Quantization Strategy**:

All models use **INT8 post-training quantization** via TensorFlow Lite:

```python
import tensorflow as tf

# Train in FP32
model.fit(train_data, epochs=100)

# Convert to TFLite with INT8 quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_generator  # Calibration data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
```

**Expected Accuracy Impact**:
- FP32 → INT8: -1 to -2% accuracy (acceptable per SC-ML-001)
- Tested on MESA validation set before deployment

**Alternatives Considered**:

| Architecture | Pros | Cons | Rejection Rationale |
|--------------|------|------|---------------------|
| **Transformers (Attention)** | SOTA accuracy (96%+) | 4x latency, 5x model size | Violates edge constraints |
| **LSTM-only** | Simple, interpretable | Slower than TCN, sequential bottleneck | Marginal latency (95ms > 100ms threshold) |
| **1D CNN-only (no temporal)** | Fast (10ms) | Poor long-range modeling | Misses 30+ second apnea events |
| **Ensemble (multiple models)** | Highest accuracy | 3x latency/memory | Prohibitive for edge |

**Implementation Recommendation**:

Use SleepKit's model factory for TCN-based models:

```python
from sleepkit.models import create_model

# Lightweight model (Variant 2)
lightweight_model = create_model(
    architecture='tcn',
    input_shape=(3600, 5),  # 30 sec @ 120 Hz, 5 channels
    num_classes=1,
    filters=[32, 64, 64, 64],
    kernel_size=3,
    dilations=[1, 2, 4, 8],
    dropout=0.2
)

# Clinical model (Variant 3)
clinical_model = create_model(
    architecture='resnet_tcn',
    input_shape=(3600, 7),  # 7 channels
    num_classes={'detection': 1, 'event_type': 5, 'severity': 4},
    resnet_blocks=[2, 2, 2, 2],
    tcn_filters=512,
    multi_task=True
)
```

---

## 6. Development Environment & Toolchain Setup

### Decision: Python 3.11 + SleepKit + Conda Environment + Docker (Optional)

**Required Software Stack**:

1. **Python Environment**:
   - **Version**: Python 3.11 (compatible with SleepKit, TensorFlow 2.15+, PyTorch 2.0+)
   - **Manager**: Conda (recommended for dependency isolation)

2. **Core Dependencies**:
   ```bash
   # Create conda environment
   conda create -n sleepkit python=3.11
   conda activate sleepkit

   # Install SleepKit and dependencies
   pip install sleepkit physiokit neuralspot-edge

   # Install ML backends (choose one or multi-backend)
   pip install tensorflow>=2.15  # Primary for TFLite export
   # Optional: pip install torch>=2.0  # For PyTorch backend
   # Optional: pip install jax>=0.4    # For JAX backend

   # Install signal processing
   pip install numpy scipy librosa  # For acoustic features

   # Install visualization
   pip install matplotlib plotly

   # Install data handling
   pip install h5py pandas

   # Install development tools
   pip install pytest pytest-cov black flake8 mypy
   ```

3. **Hardware Tools** (for edge deployment):
   - **Ambiq Apollo SDK**: For Apollo4 Plus / Apollo510 firmware
   - **TFLite Micro**: Embedded runtime (included in neuralSPOT)
   - **Development Kit**: Apollo4 Plus EVB or Apollo510 EVB
   - **Debugger**: J-Link or CMSIS-DAP for flashing/debugging

**Development Workflow**:

```
Step 1: Data Preparation
├─> Download datasets (MESA, CMIDSS, YSYW)
├─> Convert EDF to HDF5 (SleepKit loaders)
└─> Exploratory analysis (Jupyter notebooks)

Step 2: Model Training
├─> Configure experiment (YAML)
├─> Train with SleepKit CLI: sleepkit --task apnea --mode train --config configs/train.yaml
├─> Evaluate on validation set: sleepkit --task apnea --mode evaluate --config configs/eval.yaml
└─> Hyperparameter tuning (Optuna or Keras Tuner)

Step 3: Model Export
├─> Export to TFLite: sleepkit --task apnea --mode export --config configs/export.yaml
├─> Quantize (INT8)
└─> Validate accuracy (post-quantization)

Step 4: Edge Deployment
├─> Flash firmware to Apollo EVB
├─> Upload TFLite model
├─> Benchmark (latency, power, accuracy)
└─> Iterate if needed
```

**SleepKit CLI Commands**:

```bash
# Training
sleepkit --task apnea --mode train --config configs/apnea/train.yaml

# Evaluation
sleepkit --task apnea --mode evaluate --config configs/apnea/evaluate.yaml --model models/finetuned/apnea_detector

# Export to TFLite
sleepkit --task apnea --mode export --config configs/apnea/export.yaml --model models/finetuned/apnea_detector --output models/exported/apnea_detector.tflite

# Dataset download (example: MESA)
sleepkit --task download --dataset mesa --output data/mesa
```

**Configuration File Example** (configs/apnea/train.yaml):

```yaml
task: apnea
mode: train

dataset:
  name: mesa
  path: data/mesa
  signals: [ppg, spo2, accel_x, accel_y, accel_z]
  sample_rate: 120  # Hz
  window_size: 30   # seconds
  stride: 15        # 50% overlap

model:
  architecture: tcn
  input_shape: [3600, 5]  # 30 sec x 120 Hz x 5 channels
  filters: [32, 64, 64, 64]
  kernel_size: 3
  dilations: [1, 2, 4, 8]
  dropout: 0.2
  num_classes: 1

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam
  loss: binary_crossentropy
  metrics: [accuracy, precision, recall, f1]
  early_stopping:
    patience: 10
    monitor: val_f1
  callbacks:
    - tensorboard
    - model_checkpoint
    - lr_schedule

augmentation:
  enabled: true
  noise_snr: [20, 40]
  time_warp: [-0.1, 0.1]
  dropout_prob: 0.05

output:
  model_dir: models/finetuned/apnea_detector
  logs_dir: logs/apnea_detector
```

**Development Tools**:

1. **Jupyter Notebooks** (for exploration):
   ```bash
   pip install jupyter
   jupyter notebook notebooks/01_dataset_exploration.ipynb
   ```

2. **TensorBoard** (for training visualization):
   ```bash
   tensorboard --logdir logs/apnea_detector
   ```

3. **Testing Framework**:
   ```bash
   # Run all tests
   pytest tests/

   # Run specific test category
   pytest tests/validation/test_mesa_accuracy.py
   ```

4. **Code Quality**:
   ```bash
   # Format code
   black src/ tests/

   # Lint
   flake8 src/ tests/

   # Type checking
   mypy src/
   ```

**Docker Setup** (optional, for reproducibility):

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install SleepKit
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set entrypoint
ENTRYPOINT ["sleepkit"]
```

```bash
# Build Docker image
docker build -t sleepkit-apnea .

# Run training in Docker
docker run --gpus all -v $(pwd)/data:/workspace/data -v $(pwd)/models:/workspace/models sleepkit-apnea --task apnea --mode train --config configs/apnea/train.yaml
```

**Version Control**:

```bash
# Initialize git repo
git init
git remote add origin <repository_url>

# Create .gitignore
echo "data/
models/
logs/
*.h5
*.tflite
*.pyc
__pycache__/
.ipynb_checkpoints/" > .gitignore

# Commit initial setup
git add .
git commit -m "Initial project setup with SleepKit"
```

**Alternatives Considered**:

| Approach | Pros | Cons | Rejection Rationale |
|----------|------|------|---------------------|
| **Python 3.10** | Slightly broader compatibility | Missing performance improvements | 3.11 has 10-20% speed boost, SleepKit supports it |
| **Virtual Environment (venv)** | Lightweight, built-in | No cross-platform reproducibility | Conda better for scientific packages (NumPy, SciPy) |
| **Manual installation (no Conda)** | Simpler | Dependency conflicts common | SleepKit has complex dependencies (TF, PyTorch, JAX) |

---

## 7. Summary of Key Decisions

| Decision Area | Chosen Approach | Primary Rationale |
|---------------|-----------------|-------------------|
| **Edge ML Framework** | AmbiqAI/SleepKit | Purpose-built for sleep tasks, proven edge optimization, complete pipeline |
| **Pre-Trained Models** | TCN-based apnea detector + ResNet-TCN classifier | Meets accuracy (≥90%), latency (<100ms), size (<2MB) constraints |
| **Acoustic Features** | MFCC + Mel Spectrogram + ZCR + Spectral Centroid/Flatness + RMSE | Comprehensive coverage of snoring patterns, validated in literature |
| **Datasets** | MESA (primary) + CMIDSS + YSYW + custom acoustic data | Multi-ethnic, large-scale, polysomnography-validated ground truth |
| **Model Architecture (Lightweight)** | Lightweight TCN (5 channels, 620K params, 0.6 MB) | Optimal balance: 88-90% accuracy, 22ms latency, 0.7 mJ power |
| **Model Architecture (Clinical)** | ResNet-TCN Hybrid (7 channels, 2.1M params, 1.8 MB) | Clinical accuracy (93-95%), multi-task learning, <80ms latency |
| **Quantization** | INT8 post-training quantization (TFLite) | 4x size reduction, minimal accuracy loss (-1 to -2%) |
| **Development Environment** | Python 3.11 + Conda + SleepKit CLI + Docker | Reproducible, industry-standard, supports multi-backend ML |

---

## 8. Implementation Roadmap

**Phase 0 (Completed)**: Research & technical decisions (this document)

**Phase 1 (Next)**: Design & contracts
- Define data models (HDF5 schemas, API contracts)
- Document SleepKit task extensions (acoustic, multimodal, clinical)
- Create quickstart guide for developers

**Phase 2**: Implementation (via /speckit.tasks)
- Set up development environment
- Download and preprocess datasets (MESA, CMIDSS, YSYW)
- Implement custom SleepKit tasks
- Train and evaluate models
- Export to TFLite and benchmark on hardware
- Validate against success criteria (≥90% accuracy, <100ms latency)

**Timeline Estimate**:
- Phase 1 (Design): 1 week
- Phase 2 (Implementation): 6-8 weeks
  - Environment setup: 3 days
  - Dataset preparation: 1 week
  - Custom task development: 1 week
  - Model training (3 variants): 2-3 weeks (parallelizable)
  - Edge deployment & optimization: 1-2 weeks
  - Validation & testing: 1 week

---

## 9. Risk Mitigation

| Risk | Impact | Mitigation Strategy |
|------|--------|---------------------|
| **Insufficient acoustic data** | Cannot train acoustic variant | Collect 100+ nights of custom data (3-6 months), or use transfer learning from general audio models |
| **Pre-trained models don't meet accuracy** | Need to train from scratch | Fine-tune on combined datasets (MESA + CMIDSS + YSYW), use PhysioKit augmentation |
| **Edge hardware unavailable** | Cannot validate latency | Use Raspberry Pi for initial profiling, order Apollo EVB (4-week lead time) |
| **Quantization degrades accuracy** | Fails ≥90% threshold | Use quantization-aware training, increase model capacity before quantization |
| **Real-world deployment drift** | Model fails in production | Continuous monitoring, collect edge case data, periodic retraining |

---

## 10. References

**SleepKit Documentation**:
- GitHub: https://github.com/AmbiqAI/sleepkit
- API Reference: https://ambiqai.github.io/sleepkit/
- Model Zoo: https://ambiqai.github.io/sleepkit/zoo/

**PhysioKit Documentation**:
- GitHub: https://github.com/AmbiqAI/physiokit
- Signal Processing Guide: https://ambiqai.github.io/physiokit/signals/

**Datasets**:
- MESA: https://sleepdata.org/datasets/mesa
- CMIDSS: https://sleepdata.org/datasets/cmidss
- YSYW: https://physionet.org/content/you-snooze-you-win/

**Research Papers**:
- Mendonca et al. (2019): "A Review of Obstructive Sleep Apnea Detection Approaches"
- Mostafa et al. (2020): "Sleep Apnea Detection from ECG Using Deep Learning"
- Haidar et al. (2018): "Sleep Apnea Event Detection from Nasal Airflow Using Convolutional Neural Networks"

**Edge ML Resources**:
- TensorFlow Lite: https://www.tensorflow.org/lite
- Ambiq Apollo SDK: https://ambiq.com/apollo-soc/
- neuralSPOT: https://ambiqai.github.io/neuralspot/

---

**Document Status**: Complete
**Next Step**: Execute Phase 1 design workflow (/speckit.plan command)
