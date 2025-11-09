# SleepKit Quickstart Guide: Edge-Based Sleep Apnea Detection

**Feature**: `001-edge-apnea-detection` | **Version**: 2.0 | **Date**: 2025-11-05
**Framework**: [AmbiqAI/SleepKit](https://github.com/AmbiqAI/sleepkit)

## Overview

This quickstart guide walks you through the complete workflow for building an edge-based sleep apnea detection system using **SleepKit**, a production-ready framework for ultra-low power sleep monitoring. You'll learn to:

- Set up the SleepKit development environment
- Download and prepare PhysioNet datasets (MESA, CMIDSS, YSYW)
- Evaluate pre-trained apnea detection models
- Train custom models for three deployment variants
- Export models to TensorFlow Lite for edge deployment
- Deploy to Ambiq Apollo4/510 or Raspberry Pi
- Visualize and analyze results

**Total Time**: 4-8 hours (varies by bandwidth and compute resources)

---

## Table of Contents

1. [Prerequisites](#prerequisites) (5 min)
2. [Environment Setup](#step-1-environment-setup-10-15-minutes) (10-15 min)
3. [Dataset Download & Setup](#step-2-dataset-download--setup-1-2-hours) (1-2 hours)
4. [Quick Validation with Pre-trained Models](#step-3-quick-validation-with-pre-trained-models-30-minutes) (30 min)
5. [Training Deployment Variants](#step-4-training-deployment-variants-2-4-hours-per-variant) (2-4 hours)
6. [Model Evaluation](#step-5-model-evaluation-30-minutes) (30 min)
7. [TFLite Export](#step-6-tflite-export--optimization-15-minutes) (15 min)
8. [Edge Deployment](#step-7-edge-deployment-1-hour) (1 hour)
9. [Visualization & Analysis](#step-8-visualization--analysis-30-minutes) (30 min)
10. [Troubleshooting](#troubleshooting)
11. [Next Steps](#next-steps)

---

## Prerequisites

### Hardware Requirements

**Development Machine**:
- CPU: 4+ cores (8+ recommended for faster training)
- RAM: 16GB minimum (32GB recommended)
- Storage: 100GB free space (datasets + models)
- GPU: Optional but recommended (NVIDIA with CUDA support speeds up training 5-10x)

**Edge Deployment Target** (choose one):
- **Ambiq Apollo4 Plus** (ultra-low power, recommended)
- **Ambiq Apollo510** (next-gen, higher performance)
- **Nordic nRF5340** (alternative edge platform)
- **Raspberry Pi 4/5** (prototyping and development)

### Software Requirements

- **Python**: 3.11+ (SleepKit requires Python 3.10+)
- **Git**: For repository management
- **Operating System**: Linux (Ubuntu 22.04+), macOS (13+), or Windows with WSL2

### Accounts & Access

- **National Sleep Research Resource (NSRR)**: Free account required for PhysioNet dataset access
  - Sign up at: https://sleepdata.org/
  - Complete data use agreement for MESA, CMIDSS datasets

---

## Step 1: Environment Setup (10-15 minutes)

### 1.1 Create Project Directory

```bash
# Create project directory
mkdir -p ~/SleepApneaEdge
cd ~/SleepApneaEdge

# Create standard directory structure
mkdir -p configs data models notebooks scripts tests
```

### 1.2 Set Up Python Virtual Environment

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows WSL: source venv/bin/activate
                          # On Windows CMD: venv\Scripts\activate.bat
```

**Verification**:
```bash
python --version  # Should show Python 3.11.x
```

### 1.3 Install SleepKit and Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install SleepKit (includes PhysioKit and neuralspot-edge)
pip install sleepkit

# Install additional dependencies
pip install tensorflow==2.16.1 \
            keras==3.0.5 \
            numpy==1.26.4 \
            scipy==1.12.0 \
            matplotlib==3.8.3 \
            plotly==5.20.0 \
            pandas==2.2.1 \
            scikit-learn==1.4.1 \
            h5py==3.10.0 \
            pytest==8.1.1 \
            jupyterlab==4.1.5

# For edge deployment (optional, install later if needed)
# pip install tflite-runtime==2.16.1
```

**Expected output**:
```
Successfully installed sleepkit-X.X.X physiokit-X.X.X neuralspot-edge-X.X.X ...
```

### 1.4 Verify Installation

```bash
# Test SleepKit CLI
sleepkit --version

# Test Python imports
python -c "import sleepkit; import physiokit; print('SleepKit installed successfully')"
```

**Success criteria**:
- ✅ SleepKit version displayed (e.g., `sleepkit 1.5.0`)
- ✅ No import errors

**Troubleshooting**:
- If `sleepkit` command not found, ensure virtual environment is activated
- If import fails, try `pip install --upgrade sleepkit --no-cache-dir`

---

## Step 2: Dataset Download & Setup (1-2 hours)

SleepKit supports multiple PhysioNet datasets. We'll use **MESA** (Multi-Ethnic Study of Atherosclerosis) as the primary dataset.

### 2.1 Download MESA Dataset via SleepKit

```bash
# Create data directory
mkdir -p data/mesa

# Download MESA dataset (warning: ~50GB download)
sleepkit --mode download \
  --dataset mesa \
  --dest data/mesa \
  --num-workers 4
```

**Download time**: 1-2 hours depending on internet speed

**Expected output**:
```
Downloading MESA dataset...
  ├─ Downloading polysomnography data (6,814 subjects)
  ├─ Downloading annotations (sleep stages, apnea events)
  └─ Downloading metadata (demographics, AHI scores)
Progress: 100% [████████████████████████████████] 50.2GB/50.2GB
Download complete: data/mesa/
```

### 2.2 Explore Dataset Structure

```bash
# List downloaded files
ls -lh data/mesa/

# Expected structure:
# data/mesa/
# ├── edf/           # EDF signal files
# ├── annotations/   # XML annotation files
# ├── metadata.csv   # Subject metadata
# └── README.txt     # Dataset documentation
```

### 2.3 (Optional) Download Additional Datasets

**CMIDSS** (Child Mind Institute-Discovering Sleep States):
```bash
sleepkit --mode download \
  --dataset cmidss \
  --dest data/cmidss \
  --num-workers 4
```

**YSYW** (You Snooze, You Win):
```bash
sleepkit --mode download \
  --dataset ysyw \
  --dest data/ysyw \
  --num-workers 4
```

### 2.4 Verify Dataset Integrity

```bash
# Verify MESA dataset
sleepkit --mode validate \
  --dataset mesa \
  --path data/mesa

# Expected output:
# ✓ 6,814 subjects with valid polysomnography recordings
# ✓ 6,814 annotation files found
# ✓ No corrupted EDF files detected
# ✓ Metadata complete for all subjects
```

**Success criteria**:
- ✅ All subjects have matching EDF + annotation files
- ✅ No corruption errors reported

---

## Step 3: Quick Validation with Pre-trained Models (30 minutes)

Before training custom models, validate SleepKit's pre-trained apnea detector on MESA test data.

### 3.1 Download Pre-trained Model

```bash
# List available pre-trained models
sleepkit --mode list-models --task apnea

# Expected output:
# Available pre-trained models:
#   - apnea-detect-v1   : Binary apnea detector (PPG + SpO2)
#   - apnea-class-v1    : Apnea classifier (obstructive/central/mixed)
#   - apnea-multimodal  : Multimodal detector (PPG + SpO2 + IMU)

# Download binary apnea detector
sleepkit --mode download-model \
  --model apnea-detect-v1 \
  --dest models/pretrained
```

### 3.2 Run Evaluation on MESA Test Split

```bash
# Evaluate pre-trained model on MESA test set
sleepkit --mode evaluate \
  --task apnea \
  --model models/pretrained/apnea-detect-v1 \
  --dataset mesa \
  --data-path data/mesa \
  --split test \
  --output reports/pretrained_eval.json
```

**Expected runtime**: 15-30 minutes

**Expected output** (`reports/pretrained_eval.json`):
```json
{
  "model": "apnea-detect-v1",
  "dataset": "mesa",
  "split": "test",
  "num_subjects": 1022,
  "metrics": {
    "accuracy": 0.923,
    "precision": 0.891,
    "recall": 0.947,
    "f1_score": 0.918,
    "specificity": 0.901,
    "auc_roc": 0.968,
    "false_negative_rate_severe": 0.029
  },
  "per_class": {
    "normal": {"precision": 0.945, "recall": 0.901, "f1": 0.922},
    "apnea": {"precision": 0.891, "recall": 0.947, "f1": 0.918}
  },
  "confusion_matrix": [[7821, 865], [374, 6648]],
  "ahi_mae": 1.73
}
```

**Success criteria**:
- ✅ Accuracy ≥ 90% (meets SC-ML-001)
- ✅ Recall (sensitivity) ≥ 90% (meets SC-ML-002, SC-002)
- ✅ AHI MAE (Mean Absolute Error) ≤ 2 events/hour (meets SC-ML-008)
- ✅ False negative rate for severe apnea ≤ 5% (meets SC-ML-003)

### 3.3 Visualize Predictions

```bash
# Generate visualization dashboard
sleepkit --mode visualize \
  --task apnea \
  --model models/pretrained/apnea-detect-v1 \
  --dataset mesa \
  --data-path data/mesa \
  --subject mesa-0001 \
  --output visualizations/pretrained_demo.html
```

Open `visualizations/pretrained_demo.html` in your browser to see:
- Raw PPG and SpO2 signals
- Detected apnea events with timestamps
- Confidence scores over time
- AHI calculation summary

**Screenshot preview**:
```
[Signal Timeline Visualization]
PPG: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SpO2: ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Events:     ▼   ▼      ▼▼     ▼        ▼
         00:30 01:15  02:45  04:20   06:10
         (18s) (12s)  (15s)  (23s)   (14s)

AHI Score: 12.4 events/hour (Mild apnea)
```

---

## Step 4: Training Deployment Variants (2-4 hours per variant)

Train custom models for the three deployment paths outlined in `plan.md`.

### Variant 1: Acoustic Apnea Detection Prototype

**Goal**: Detect apnea from audio signals (snoring patterns).

#### 4.1.1 Create Configuration File

Create `configs/acoustic/train.yaml`:

```yaml
# configs/acoustic/train.yaml
task: apnea_acoustic
dataset: mesa
data_path: data/mesa
output_path: models/acoustic

# Signal configuration
signals:
  - audio  # Acoustic breathing sounds

# Feature extraction
features:
  type: acoustic
  mfcc_coefficients: 40
  mel_bands: 80
  window_size: 30  # seconds
  hop_size: 5      # seconds
  sample_rate: 16000

# Model architecture
model:
  type: cnn_lstm
  conv_layers:
    - filters: 32
      kernel_size: 5
      activation: relu
    - filters: 64
      kernel_size: 5
      activation: relu
  lstm_units: 128
  dropout: 0.3
  dense_units: [64, 32]

# Training parameters
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam
  early_stopping:
    patience: 10
    monitor: val_loss

# Data split
split:
  train: 0.7
  val: 0.15
  test: 0.15
  stratify: ahi_severity

# Success criteria
success_criteria:
  accuracy: 0.90
  recall: 0.90
  precision: 0.85
```

#### 4.1.2 Train Acoustic Model

```bash
# Train acoustic apnea detector
sleepkit --mode train \
  --config configs/acoustic/train.yaml \
  --seed 42

# Training will output progress:
# Epoch 1/50: loss=0.4521, acc=0.782, val_loss=0.3912, val_acc=0.821
# Epoch 2/50: loss=0.3654, acc=0.841, val_loss=0.3201, val_acc=0.867
# ...
# Epoch 34/50: loss=0.1234, acc=0.932, val_loss=0.1456, val_acc=0.918
# Early stopping triggered (patience=10)
# Best model saved: models/acoustic/best_model.keras
```

**Expected training time**:
- With GPU: 2-3 hours
- CPU only: 6-8 hours

**Success criteria**:
- ✅ Validation accuracy ≥ 90%
- ✅ Model converges (loss decreasing)

---

### Variant 2: Multimodal Lightweight Model (Edge Focus)

**Goal**: High-accuracy detector using PPG + SpO2 + IMU, optimized for edge inference.

#### 4.2.1 Create Configuration File

Create `configs/multimodal/train_lightweight.yaml`:

```yaml
# configs/multimodal/train_lightweight.yaml
task: apnea_multimodal_lightweight
dataset: mesa
data_path: data/mesa
output_path: models/multimodal_lightweight

# Multi-signal configuration
signals:
  - ppg       # Photoplethysmography
  - spo2      # Oxygen saturation
  - accel     # Accelerometer (respiratory effort)

# Feature fusion
features:
  type: multimodal_fusion
  ppg_features:
    - heart_rate_variability
    - pulse_amplitude
    - pulse_rate
  spo2_features:
    - oxygen_saturation
    - desaturation_events
  accel_features:
    - respiratory_rate
    - movement_intensity
  fusion_method: concatenate
  window_size: 30
  hop_size: 10

# Lightweight model (optimized for edge)
model:
  type: tcn_lightweight  # Temporal Convolutional Network
  num_filters: [32, 64, 64]
  kernel_size: 3
  dilations: [1, 2, 4, 8]
  dropout: 0.2
  dense_units: [32]
  activation: relu
  output_activation: sigmoid

# Training with quantization awareness
training:
  epochs: 60
  batch_size: 64
  learning_rate: 0.0005
  optimizer: adam
  quantization_aware: true  # QAT for better edge accuracy
  target_size_mb: 2.0
  early_stopping:
    patience: 12
    monitor: val_f1_score

# Edge optimization constraints
edge_constraints:
  max_latency_ms: 100
  max_model_size_mb: 2
  quantization: int8
  target_hardware: apollo4_plus

# Success criteria
success_criteria:
  accuracy: 0.90
  recall: 0.95
  precision: 0.85
  f1_score: 0.87
```

#### 4.2.2 Train Multimodal Lightweight Model

```bash
# Train with edge optimization
sleepkit --mode train \
  --config configs/multimodal/train_lightweight.yaml \
  --seed 42 \
  --verbose
```

**Expected training time**:
- With GPU: 3-4 hours
- CPU only: 10-12 hours

**Success criteria**:
- ✅ Validation accuracy ≥ 90%
- ✅ Model size < 2MB (after quantization)
- ✅ Quantization-aware training converges

---

### Variant 3: Clinical-Grade Apnea Research Model

**Goal**: Maximum accuracy for clinical research and regulatory validation.

#### 4.3.1 Create Configuration File

Create `configs/clinical/train_research.yaml`:

```yaml
# configs/clinical/train_research.yaml
task: apnea_clinical_research
datasets:
  - mesa      # Primary dataset
  - cmidss    # Wearable validation
  - ysyw      # Clinical validation
data_paths:
  mesa: data/mesa
  cmidss: data/cmidss
  ysyw: data/ysyw
output_path: models/clinical_research

# Comprehensive signal suite
signals:
  - ppg
  - spo2
  - ecg
  - respiratory_effort
  - airflow
  - accel

# Advanced feature extraction
features:
  type: clinical_comprehensive
  include_all_physiokit_features: true
  synthetic_augmentation: true  # PhysioKit synthetic data
  augmentation_ratio: 0.3

# Advanced architecture
model:
  type: resnet_tcn  # ResNet + Temporal Conv
  resnet_blocks: 4
  filters_per_block: [64, 128, 256, 512]
  tcn_layers: 6
  kernel_size: 5
  dropout: 0.4
  attention_mechanism: true
  dense_units: [256, 128, 64]

# Training for maximum accuracy
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0001
  optimizer: adamw
  weight_decay: 0.01
  learning_rate_schedule: cosine_decay
  early_stopping:
    patience: 20
    monitor: val_auc_roc

# Multi-class classification
classification:
  classes:
    - normal
    - obstructive_apnea
    - central_apnea
    - mixed_apnea
    - hypopnea
  loss: categorical_crossentropy

# Success criteria (clinical grade)
success_criteria:
  overall_accuracy: 0.95
  per_class_accuracy: 0.90
  ahi_mae: 1.5
  auc_roc: 0.98
```

#### 4.3.2 Train Clinical Research Model

```bash
# Train on combined datasets
sleepkit --mode train \
  --config configs/clinical/train_research.yaml \
  --seed 42 \
  --multi-gpu  # Use all available GPUs
```

**Expected training time**:
- With multi-GPU: 4-6 hours
- Single GPU: 12-16 hours
- CPU only: Not recommended (2-3 days)

**Success criteria**:
- ✅ Overall accuracy ≥ 95%
- ✅ Per-class accuracy ≥ 90% for all apnea types
- ✅ AHI MAE ≤ 1.5 events/hour

---

## Step 5: Model Evaluation (30 minutes)

Comprehensive evaluation of all trained models.

### 5.1 Evaluate All Variants

```bash
# Acoustic model
sleepkit --mode evaluate \
  --task apnea \
  --model models/acoustic/best_model.keras \
  --dataset mesa \
  --data-path data/mesa \
  --split test \
  --output reports/acoustic_eval.json

# Multimodal lightweight model
sleepkit --mode evaluate \
  --task apnea \
  --model models/multimodal_lightweight/best_model.keras \
  --dataset mesa \
  --data-path data/mesa \
  --split test \
  --output reports/multimodal_lightweight_eval.json

# Clinical research model
sleepkit --mode evaluate \
  --task apnea \
  --model models/clinical_research/best_model.keras \
  --dataset mesa \
  --data-path data/mesa \
  --split test \
  --metrics accuracy precision recall f1 auc_roc ahi_mae \
  --per-class \
  --output reports/clinical_research_eval.json
```

### 5.2 Generate Comparison Report

```bash
# Compare all models
sleepkit --mode compare \
  --models models/acoustic/best_model.keras \
           models/multimodal_lightweight/best_model.keras \
           models/clinical_research/best_model.keras \
  --dataset mesa \
  --data-path data/mesa \
  --split test \
  --output reports/model_comparison.html
```

Open `reports/model_comparison.html` to see side-by-side metrics:

```
Model Comparison - MESA Test Set (n=1022 subjects)
═══════════════════════════════════════════════════════════════════

Model                      Accuracy  Precision  Recall  F1    AUC    AHI MAE
─────────────────────────────────────────────────────────────────────────────
Acoustic                   0.891     0.847      0.923   0.883  0.941  2.34
Multimodal Lightweight     0.932     0.901      0.954   0.927  0.972  1.68
Clinical Research          0.957     0.941      0.968   0.954  0.989  1.21
─────────────────────────────────────────────────────────────────────────────

✓ All models meet ≥90% accuracy threshold (SC-ML-001)
✓ Clinical model achieves target AHI MAE ≤1.5 (SC-ML-008)
```

### 5.3 Generate Confusion Matrices

```bash
# Generate confusion matrix visualizations
sleepkit --mode plot-confusion \
  --models models/*/best_model.keras \
  --dataset mesa \
  --data-path data/mesa \
  --split test \
  --output visualizations/confusion_matrices.png
```

**Expected output**: `visualizations/confusion_matrices.png` showing:
- True Positives, False Positives, True Negatives, False Negatives
- Per-class precision/recall for clinical model

---

## Step 6: TFLite Export & Optimization (15 minutes)

Convert trained models to TensorFlow Lite for edge deployment.

### 6.1 Export Multimodal Lightweight Model

```bash
# Export with INT8 quantization
sleepkit --mode export \
  --task apnea \
  --model models/multimodal_lightweight/best_model.keras \
  --format tflite \
  --quantization int8 \
  --representative-dataset data/mesa \
  --output models/exported/multimodal_apnea_detector.tflite
```

**Expected output**:
```
Exporting model to TFLite...
  ├─ Loading model: models/multimodal_lightweight/best_model.keras
  ├─ Generating representative dataset (1000 samples)
  ├─ Applying INT8 quantization
  ├─ Optimizing for edge inference
  └─ Saved: models/exported/multimodal_apnea_detector.tflite

Model Statistics:
  Original size (Keras): 8.4 MB
  TFLite size (INT8):    1.8 MB (78.6% reduction)
  Compression ratio:     4.67x
```

**Success criteria**:
- ✅ Model size < 2MB (meets SC-ML-006)

### 6.2 Validate Quantized Model Accuracy

```bash
# Test quantized model accuracy
sleepkit --mode evaluate \
  --model models/exported/multimodal_apnea_detector.tflite \
  --dataset mesa \
  --data-path data/mesa \
  --split test \
  --output reports/tflite_eval.json
```

**Expected output** (`reports/tflite_eval.json`):
```json
{
  "model": "multimodal_apnea_detector.tflite",
  "quantization": "int8",
  "metrics": {
    "accuracy": 0.928,
    "precision": 0.897,
    "recall": 0.951,
    "f1_score": 0.923
  },
  "accuracy_drop": 0.004,
  "size_reduction": "78.6%"
}
```

**Success criteria**:
- ✅ Accuracy drop < 2% vs. FP32 model (meets SC-ML-005)

### 6.3 Export Other Models

```bash
# Export acoustic model
sleepkit --mode export \
  --model models/acoustic/best_model.keras \
  --format tflite \
  --quantization int8 \
  --output models/exported/acoustic_apnea_detector.tflite

# Export clinical model (may exceed 2MB, for research use)
sleepkit --mode export \
  --model models/clinical_research/best_model.keras \
  --format tflite \
  --quantization int8 \
  --output models/exported/clinical_apnea_classifier.tflite
```

---

## Step 7: Edge Deployment (1 hour)

Deploy TFLite models to target edge hardware.

### Option A: Ambiq Apollo4 Plus (Recommended)

#### 7.1 Install Ambiq SDK

```bash
# Clone Ambiq SDK
git clone https://github.com/AmbiqAI/ambiqsuite.git
cd ambiqsuite

# Install build tools
./install.sh
```

#### 7.2 Create Deployment Project

```bash
# Use SleepKit's edge deployment template
sleepkit --mode create-edge-project \
  --platform apollo4_plus \
  --model ../models/exported/multimodal_apnea_detector.tflite \
  --output apollo4_deployment
```

**Generated structure**:
```
apollo4_deployment/
├── src/
│   ├── main.c              # Application entry point
│   ├── model_runner.c      # TFLite inference
│   ├── sensor_driver.c     # PPG/SpO2/IMU drivers
│   └── apnea_detector.c    # Detection logic
├── include/
├── models/
│   └── multimodal_apnea_detector.tflite
├── CMakeLists.txt
└── README.md
```

#### 7.3 Build Firmware

```bash
cd apollo4_deployment
mkdir build && cd build

cmake .. -DBOARD=apollo4_evb
make -j8

# Output: build/apnea_detector.bin
```

#### 7.4 Flash to Device

```bash
# Connect Apollo4 EVB via USB

# Flash firmware
python ../scripts/flash.py \
  --port /dev/ttyUSB0 \
  --binary apnea_detector.bin \
  --verify
```

**Expected output**:
```
Flashing Apollo4 Plus...
  ├─ Erasing flash
  ├─ Writing firmware (324 KB)
  ├─ Verifying checksum
  └─ Done! Device rebooting...

Device ready for apnea detection.
```

### Option B: Raspberry Pi (Prototyping)

#### 7.5 Deploy to Raspberry Pi

```bash
# Copy TFLite model to Raspberry Pi
scp models/exported/multimodal_apnea_detector.tflite pi@raspberrypi.local:~/

# SSH into Pi
ssh pi@raspberrypi.local

# Install TFLite runtime
pip3 install tflite-runtime

# Run inference demo
python3 inference_demo.py \
  --model multimodal_apnea_detector.tflite \
  --test-data sample_signals.h5
```

### 7.6 Benchmark Edge Performance

```bash
# Profile inference latency on target hardware
sleepkit --mode benchmark \
  --model models/exported/multimodal_apnea_detector.tflite \
  --platform apollo4_plus \
  --port /dev/ttyUSB0 \
  --iterations 1000 \
  --output reports/edge_benchmark.json
```

**Expected output** (`reports/edge_benchmark.json`):
```json
{
  "platform": "apollo4_plus",
  "model": "multimodal_apnea_detector.tflite",
  "iterations": 1000,
  "performance": {
    "inference_time_mean_ms": 73.4,
    "inference_time_std_ms": 2.1,
    "inference_time_max_ms": 81.2,
    "throughput_fps": 13.6,
    "memory_usage_kb": 1847,
    "power_consumption_mw": 138.7,
    "energy_per_inference_mj": 0.76
  }
}
```

**Success criteria**:
- ✅ Inference latency < 100ms (meets SC-ML-004)
- ✅ Energy per inference < 1 millijoule (power efficiency target)

---

## Step 8: Visualization & Analysis (30 minutes)

### 8.1 Generate Sleep Report for Test Subject

```bash
# Process one night of sleep data
sleepkit --mode infer \
  --model models/exported/multimodal_apnea_detector.tflite \
  --dataset mesa \
  --data-path data/mesa \
  --subject mesa-0042 \
  --output results/mesa-0042_report.json

# Generate visualization
sleepkit --mode visualize \
  --task apnea \
  --report results/mesa-0042_report.json \
  --output visualizations/mesa-0042_sleep_report.html
```

Open `visualizations/mesa-0042_sleep_report.html` to see:

```
┌────────────────────────────────────────────────────────────┐
│  Sleep Apnea Detection Report                              │
│  Subject: mesa-0042 | Date: 2024-03-15                    │
│  Total Sleep Time: 7.2 hours                               │
└────────────────────────────────────────────────────────────┘

AHI Score: 18.7 events/hour (Moderate Apnea)

Event Breakdown:
  ┌─────────────────────────────────────────┐
  │ Obstructive Apnea:  89 events (66.4%)  │
  │ Central Apnea:      32 events (23.9%)  │
  │ Mixed Apnea:        13 events (9.7%)   │
  │ Total:              134 events          │
  └─────────────────────────────────────────┘

Signal Timeline:
  PPG   │━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
  SpO2  │━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
  Events│    ▼  ▼   ▼▼ ▼    ▼   ▼▼  ▼  ▼    │
        └────────────────────────────────────┘
        22:00  00:00  02:00  04:00  06:00

Longest Event: 38 seconds (03:24 AM)
Average Event Duration: 16.2 seconds
SpO2 Minimum: 84% (during 38s event)
```

### 8.2 Calculate AHI Accuracy

```bash
# Compare calculated AHI to clinical ground truth
sleepkit --mode validate-ahi \
  --reports results/*.json \
  --ground-truth data/mesa/annotations/ \
  --output reports/ahi_validation.csv
```

**Expected output** (`reports/ahi_validation.csv`):
```csv
subject_id,predicted_ahi,ground_truth_ahi,absolute_error,relative_error
mesa-0001,5.2,6.1,0.9,14.8%
mesa-0002,18.7,17.2,1.5,8.7%
mesa-0003,32.4,30.9,1.5,4.9%
...
mean,,,1.68,10.2%
```

**Success criteria**:
- ✅ Mean AHI absolute error ≤ 2 events/hour (meets SC-ML-008)

### 8.3 Generate Trend Analysis (Multi-Night)

```bash
# Process 30 nights of data for longitudinal analysis
sleepkit --mode batch-infer \
  --model models/exported/multimodal_apnea_detector.tflite \
  --dataset mesa \
  --data-path data/mesa \
  --subjects mesa-0042 \
  --nights 30 \
  --output results/longitudinal/

# Generate trend visualization
sleepkit --mode plot-trends \
  --reports results/longitudinal/*.json \
  --output visualizations/30_night_trends.html
```

**Visualization shows**:
- AHI scores over 30 nights (time series)
- Event frequency patterns (weekday vs. weekend)
- Severity distribution (mild/moderate/severe nights)
- Sleep quality correlation

---

## Troubleshooting

### Issue: MESA dataset download fails

**Symptoms**:
```
Error: Connection timeout while downloading from sleepdata.org
```

**Solutions**:
1. **Check NSRR account**: Ensure data use agreement is signed
2. **Retry with fewer workers**: `--num-workers 1`
3. **Download manually**:
   ```bash
   # Download via browser from sleepdata.org
   # Extract to data/mesa/
   sleepkit --mode validate --dataset mesa --path data/mesa
   ```
4. **Use alternative dataset**: Try CMIDSS or YSYW if MESA unavailable

---

### Issue: Training accuracy plateaus below 90%

**Symptoms**:
```
Epoch 50/50: val_acc=0.876 (not improving)
```

**Solutions**:
1. **Increase training data**: Add CMIDSS or YSYW datasets
   ```yaml
   datasets:
     - mesa
     - cmidss  # Add this
   ```
2. **Enable data augmentation**:
   ```yaml
   features:
     synthetic_augmentation: true
     augmentation_ratio: 0.5  # Increase from 0.3
   ```
3. **Tune hyperparameters**:
   ```yaml
   training:
     learning_rate: 0.0005  # Reduce from 0.001
     batch_size: 16         # Reduce from 32
   ```
4. **Try different architecture**:
   ```yaml
   model:
     type: resnet_tcn  # Instead of cnn_lstm or tcn_lightweight
   ```
5. **Check data quality**:
   ```bash
   sleepkit --mode analyze-dataset \
     --dataset mesa \
     --path data/mesa \
     --check-labels \
     --check-balance
   ```

---

### Issue: TFLite export accuracy drops >2%

**Symptoms**:
```
TFLite accuracy: 0.863 (FP32 accuracy: 0.932)
Accuracy drop: 6.9% ✗ (exceeds 2% threshold)
```

**Solutions**:
1. **Use quantization-aware training (QAT)**:
   ```yaml
   training:
     quantization_aware: true  # Train with quantization simulation
   ```
   Re-train model with QAT enabled, then export again.

2. **Increase representative dataset size**:
   ```bash
   sleepkit --mode export \
     --model models/multimodal_lightweight/best_model.keras \
     --format tflite \
     --quantization int8 \
     --representative-dataset data/mesa \
     --representative-samples 5000  # Increase from default 1000
   ```

3. **Try dynamic range quantization** (less aggressive):
   ```bash
   sleepkit --mode export \
     --quantization dynamic  # Instead of int8
   ```

4. **Use float16 quantization** (if edge device supports):
   ```bash
   sleepkit --mode export \
     --quantization float16
   ```

---

### Issue: Edge inference latency exceeds 100ms

**Symptoms**:
```
Apollo4 benchmark: 142.3ms ± 5.1ms ✗ (exceeds 100ms target)
```

**Solutions**:
1. **Reduce model complexity**:
   ```yaml
   model:
     num_filters: [16, 32, 32]  # Reduce from [32, 64, 64]
     dense_units: [16]          # Reduce from [32]
   ```

2. **Enable hardware acceleration**:
   ```cmake
   # In CMakeLists.txt
   set(TFLITE_USE_CMSIS_NN ON)  # Enable CMSIS-NN optimizations
   set(TFLITE_USE_ETHOS_U ON)   # If using Arm Ethos-U NPU
   ```

3. **Optimize window size**:
   ```yaml
   features:
     window_size: 15  # Reduce from 30 seconds
     hop_size: 5      # Keep hop size for overlap
   ```

4. **Profile bottlenecks**:
   ```bash
   sleepkit --mode profile \
     --model models/exported/multimodal_apnea_detector.tflite \
     --platform apollo4_plus \
     --detailed
   # Output shows which layers are slowest
   ```

5. **Switch to faster architecture**:
   - TCN (Temporal Convolutional Network) is faster than LSTM
   - MobileNet-inspired architectures for edge devices

---

### Issue: Model accuracy is high but AHI calculation is inaccurate

**Symptoms**:
```
Model accuracy: 0.932 ✓
AHI MAE: 4.2 events/hour ✗ (exceeds ±2 threshold)
```

**Root cause**: Good event-level detection but poor aggregation/calibration.

**Solutions**:
1. **Calibrate detection threshold**:
   ```bash
   sleepkit --mode calibrate-threshold \
     --model models/multimodal_lightweight/best_model.keras \
     --dataset mesa \
     --metric ahi_mae \
     --output configs/calibrated_threshold.yaml
   ```
   This finds optimal confidence threshold for AHI calculation.

2. **Add post-processing rules**:
   ```yaml
   inference:
     post_processing:
       min_event_duration: 10  # Seconds (clinical definition)
       merge_events_within: 5  # Merge events <5s apart
       min_confidence: 0.8     # Higher threshold reduces false positives
   ```

3. **Train on AHI-weighted loss**:
   ```yaml
   training:
     loss: custom_ahi_loss  # Directly optimize AHI error
     ahi_loss_weight: 0.5
   ```

4. **Validate on per-severity subsets**:
   ```bash
   # Check if error is worse for specific severity levels
   sleepkit --mode evaluate \
     --model models/multimodal_lightweight/best_model.keras \
     --dataset mesa \
     --split test \
     --stratify-by ahi_severity \
     --output reports/per_severity_eval.json
   ```

---

### Issue: Sensor connection failures on edge device

**Symptoms**:
```
[ERROR] MAX30102 initialization failed (I2C timeout)
```

**Solutions**:
1. **Check I2C wiring**:
   - Verify SDA/SCL connections
   - Ensure pullup resistors (4.7kΩ) are present
   - Measure voltage: should be 3.3V

2. **Reduce I2C bus speed**:
   ```c
   // In sensor_driver.c
   i2c_config.frequency = 100000;  // Reduce from 400kHz to 100kHz
   ```

3. **Enable I2C debugging**:
   ```c
   #define I2C_DEBUG 1
   // Rebuild firmware to see detailed I2C transactions
   ```

4. **Test sensor individually**:
   ```bash
   # Use vendor example code to verify sensor hardware
   cd ambiqsuite/examples/sensors/max30102_test
   make flash
   ```

5. **Power supply issues**:
   - MAX30102 peak current: ~50mA (LED on)
   - Use external power supply or larger capacitor (100µF)

---

### Issue: High battery drain on edge device

**Symptoms**:
```
Expected: 8-10 hour battery life
Actual: 3-4 hour battery life
```

**Solutions**:
1. **Enable adaptive sampling**:
   ```c
   // In apnea_detector.c
   #define ENABLE_ADAPTIVE_SAMPLING 1
   #define IDLE_SAMPLING_RATE 1  // Hz (vs. 25 Hz active)
   ```

2. **Reduce PPG LED brightness**:
   ```c
   max30102_set_led_current(MAX30102_LED_CURRENT_6_4MA);  // Reduce from 50mA
   ```
   Note: May affect signal quality, validate accuracy after change.

3. **Use sensor gating**:
   ```c
   // Power down sensors during stable periods (no movement, stable HR)
   if (is_stable_period()) {
       max30102_sleep_mode();
       adxl345_sleep_mode();
   }
   ```

4. **Optimize inference scheduling**:
   ```c
   // Run inference every 30s instead of continuous
   #define INFERENCE_INTERVAL_MS 30000
   ```

5. **Profile power consumption**:
   ```bash
   # Measure actual current draw
   sleepkit --mode power-profile \
     --platform apollo4_plus \
     --port /dev/ttyUSB0 \
     --duration 3600  # 1 hour test
   ```

---

## Next Steps

### Immediate Next Steps

1. **Clinical Validation**
   - Test on real subjects with concurrent polysomnography
   - Collect 50+ nights of validation data
   - Calculate sensitivity/specificity vs. clinical gold standard
   - Document results for regulatory submission

2. **Model Optimization**
   - Experiment with model pruning (reduce size further)
   - Try knowledge distillation (teacher-student training)
   - Explore neural architecture search (NAS) for optimal edge models

3. **Production Deployment**
   - Design PCB for custom wearable device
   - Develop mobile app for data visualization and sync
   - Implement OTA firmware updates
   - Add cloud backend for long-term data storage (optional)

### Advanced Features

4. **Real-Time Alerts**
   - Add haptic feedback for severe apnea events
   - Implement smart alarm (wake user after severe event)
   - Bluetooth notifications to smartphone

5. **Multi-User Support**
   - User profiles and data isolation
   - Family account management
   - Healthcare provider dashboard integration

6. **Additional Sleep Disorders**
   - Sleep stage classification (REM, deep, light, awake)
   - Periodic limb movement detection
   - Snoring intensity measurement
   - Sleep quality scoring

### Regulatory & Compliance

7. **FDA 510(k) Submission** (if marketing as medical device in US)
   - Prepare technical documentation
   - Conduct clinical trials (510+ subjects recommended)
   - Submit pre-market notification
   - Estimated timeline: 6-12 months

8. **CE Marking** (if marketing in EU)
   - Determine device class (likely Class IIa for apnea monitor)
   - ISO 13485 quality management system
   - ISO 14971 risk management
   - Notified Body review

9. **HIPAA Compliance** (if storing patient data)
   - Implement data encryption (AES-256)
   - Secure authentication (OAuth 2.0)
   - Audit logging
   - Business Associate Agreements (BAAs)

---

## Additional Resources

### Documentation

- **SleepKit Official Docs**: https://github.com/AmbiqAI/sleepkit/wiki
- **PhysioKit API Reference**: https://ambiqai.github.io/physiokit
- **neuralSPOT Edge Guide**: https://github.com/AmbiqAI/neuralspot-edge
- **Keras 3 Documentation**: https://keras.io/keras_3/
- **TensorFlow Lite for Microcontrollers**: https://www.tensorflow.org/lite/microcontrollers

### Project-Specific Docs

- **API Contracts**: `contracts/sleepkit-task-api.md`, `contracts/inference-api.md`
- **Data Model**: `data-model.md`
- **Research Notes**: `research.md`
- **Implementation Plan**: `plan.md`
- **Feature Spec**: `spec.md`

### Datasets

- **MESA**: https://sleepdata.org/datasets/mesa
- **CMIDSS**: https://sleepdata.org/datasets/cmidss
- **YSYW**: https://sleepdata.org/datasets/ysyw
- **PhysioNet Sleep Databases**: https://physionet.org/about/database/#sleep

### Hardware

- **Ambiq Apollo4 Plus**: https://ambiq.com/apollo4-plus/
- **Ambiq Apollo510**: https://ambiq.com/apollo5/
- **Nordic nRF5340**: https://www.nordicsemi.com/Products/nRF5340
- **MAX30102 Datasheet**: https://www.maximintegrated.com/en/products/sensors/MAX30102.html

### Community & Support

- **SleepKit GitHub Issues**: https://github.com/AmbiqAI/sleepkit/issues
- **SleepKit Discussions**: https://github.com/AmbiqAI/sleepkit/discussions
- **Ambiq Developer Forum**: https://ambiq.com/community/
- **PhysioNet Forum**: https://groups.google.com/g/physionet-users

### Academic References

1. **AASM Scoring Manual**: Berry RB, et al. "The AASM Manual for the Scoring of Sleep and Associated Events." (2020)
2. **MESA Study**: Chen X, et al. "Racial/Ethnic Differences in Sleep Disturbances: The Multi-Ethnic Study of Atherosclerosis (MESA)." Sleep (2015)
3. **Edge ML for Sleep**: Fonseca P, et al. "Sleep stage classification with ECG and respiratory effort." Physiol Meas (2015)
4. **Apnea Detection Review**: Mendonça F, et al. "A Review of Obstructive Sleep Apnea Detection Approaches." IEEE JBHI (2019)

---

## Appendix: Quick Reference Commands

### Training
```bash
# Train acoustic model
sleepkit --mode train --config configs/acoustic/train.yaml --seed 42

# Train multimodal lightweight model
sleepkit --mode train --config configs/multimodal/train_lightweight.yaml --seed 42

# Train clinical research model
sleepkit --mode train --config configs/clinical/train_research.yaml --seed 42 --multi-gpu
```

### Evaluation
```bash
# Evaluate single model
sleepkit --mode evaluate \
  --model models/multimodal_lightweight/best_model.keras \
  --dataset mesa --data-path data/mesa --split test \
  --output reports/eval.json

# Compare multiple models
sleepkit --mode compare \
  --models models/*/best_model.keras \
  --dataset mesa --data-path data/mesa --split test \
  --output reports/comparison.html
```

### Export
```bash
# Export to TFLite with INT8 quantization
sleepkit --mode export \
  --model models/multimodal_lightweight/best_model.keras \
  --format tflite --quantization int8 \
  --representative-dataset data/mesa \
  --output models/exported/detector.tflite
```

### Inference
```bash
# Single subject inference
sleepkit --mode infer \
  --model models/exported/detector.tflite \
  --dataset mesa --data-path data/mesa \
  --subject mesa-0042 \
  --output results/mesa-0042.json

# Batch inference
sleepkit --mode batch-infer \
  --model models/exported/detector.tflite \
  --dataset mesa --data-path data/mesa \
  --subjects mesa-* \
  --output results/batch/
```

### Visualization
```bash
# Generate sleep report
sleepkit --mode visualize \
  --task apnea \
  --report results/mesa-0042.json \
  --output visualizations/report.html

# Plot trends
sleepkit --mode plot-trends \
  --reports results/batch/*.json \
  --output visualizations/trends.html
```

### Benchmarking
```bash
# Benchmark on edge device
sleepkit --mode benchmark \
  --model models/exported/detector.tflite \
  --platform apollo4_plus \
  --port /dev/ttyUSB0 \
  --iterations 1000 \
  --output reports/benchmark.json
```

---

**Last Updated**: 2025-11-05 | **Version**: 2.0 (SleepKit-based)
**Feedback**: Please report issues or suggestions via GitHub Issues
