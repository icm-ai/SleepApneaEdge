# Quickstart Guide: Edge-Based Sleep Apnea Detection

**Feature**: `001-edge-apnea-detection` | **Version**: 1.0 | **Date**: 2025-11-04

## Overview

This quickstart guide helps developers set up the development environment, train models, and deploy the edge-based sleep apnea detection system to Nordic nRF5340 or Analog Devices MAX32664 hardware.

**Time to Complete**: ~2-3 hours (excluding model training)

---

## Prerequisites

### Hardware
- **Development Board**: Nordic nRF5340 DK or Analog Devices MAX32664 evaluation kit
- **Sensors**:
  - MAX30102 or MAX30105 (integrated PPG + SpO2)
  - ADXL345 or equivalent 3-axis accelerometer
  - Connecting wires and breadboard (if not using integrated sensor hub)
- **USB Cable**: For firmware flashing and debugging
- **Computer**: Linux, macOS, or Windows with WSL2

### Software
- **Python**: 3.11+ (for ML model development)
- **ARM GCC Toolchain**: 10.3+ (for firmware compilation)
- **Nordic nRF Command Line Tools**: v10.15+ (if using nRF5340)
- **Git**: For repository management
- **Optional**: MATLAB R2023a+ (for advanced signal analysis)

### Accounts
- **National Sleep Research Resource (NSRR)**: Free account for dataset access

---

## Step 1: Environment Setup

### 1.1 Clone Repository

```bash
git clone https://github.com/your-org/SleepApneaEdge.git
cd SleepApneaEdge
git checkout 001-edge-apnea-detection
```

### 1.2 Install Python Dependencies

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Key dependencies** (requirements.txt):
```text
tensorflow==2.14.0
tensorflow-lite-tools==0.0.1
numpy==1.26.0
scipy==1.11.3
pandas==2.1.1
scikit-learn==1.3.1
matplotlib==3.8.0
pytest==7.4.2
h5py==3.10.0
```

### 1.3 Install ARM GCC Toolchain

**macOS (Homebrew)**:
```bash
brew install --cask gcc-arm-embedded
```

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get update
sudo apt-get install gcc-arm-none-eabi gdb-multiarch
```

**Windows**: Download from [ARM Developer](https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain)

### 1.4 Install nRF Command Line Tools (if using nRF5340)

Download from [Nordic Semiconductor](https://www.nordicsemi.com/Products/Development-tools/nrf-command-line-tools)

Verify installation:
```bash
nrfjprog --version
```

---

## Step 2: Dataset Preparation

### 2.1 Download NSRR Sleep Heart Health Study

1. Create account at [NSRR](https://sleepdata.org/)
2. Navigate to Sleep Heart Health Study dataset
3. Download EDF files (polysomnography recordings) and annotations

```bash
# Using NSRR gem (recommended)
gem install nsrr
nsrr download shhs/polysomnography/edfs --shallow

# Place downloaded files in project directory
mkdir -p data/raw/shhs
mv ~/downloads/shhs1-*.edf data/raw/shhs/
```

### 2.2 Preprocess Dataset

Extract PPG, SpO2, and respiratory signals from EDF files:

```bash
python src/preprocessing/extract_signals.py \
  --input data/raw/shhs \
  --output data/processed/shhs \
  --signals ppg spo2 respiratory \
  --window-size 30 \
  --sampling-rate 25
```

**Expected output**:
- `data/processed/shhs/features.h5`: Preprocessed feature vectors
- `data/processed/shhs/labels.csv`: Apnea event annotations
- `data/processed/shhs/metadata.json`: Dataset statistics

### 2.3 Create Train/Val/Test Splits

```bash
python src/preprocessing/split_dataset.py \
  --input data/processed/shhs \
  --output data/splits \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --stratify-by ahi_severity \
  --random-seed 42
```

---

## Step 3: Model Training

### 3.1 Train Apnea Detection Model

Train the binary classifier (apnea vs. normal breathing):

```bash
python src/cli/train.py \
  --config configs/detector_config.yaml \
  --data data/splits \
  --output models/detector \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --early-stopping-patience 10
```

**Expected training time**: 2-4 hours on GPU, 8-12 hours on CPU

**Monitor training**:
```bash
tensorboard --logdir models/detector/logs
```

**Success criteria**:
- Validation accuracy ≥90% (SC-ML-001)
- Recall (sensitivity) ≥90% (SC-ML-002)
- Precision ≥85% (SC-ML-002)
- F1 score ≥87% (SC-ML-002)

### 3.2 Train Event Classification Model

Train the multi-class classifier (obstructive/central/mixed):

```bash
python src/cli/train.py \
  --config configs/classifier_config.yaml \
  --data data/splits \
  --output models/classifier \
  --epochs 50 \
  --batch-size 32
```

**Success criteria**:
- Validation accuracy ≥90% (SC-ML-007)

### 3.3 Evaluate Models

Run evaluation on test set:

```bash
python src/cli/evaluate.py \
  --detector models/detector/best_model.h5 \
  --classifier models/classifier/best_model.h5 \
  --test-data data/splits/test \
  --output reports/evaluation.json
```

**Expected output** (reports/evaluation.json):
```json
{
  "detector": {
    "accuracy": 0.952,
    "precision": 0.887,
    "recall": 0.931,
    "f1": 0.908,
    "false_negative_rate_severe": 0.03
  },
  "classifier": {
    "accuracy": 0.914,
    "per_class": {
      "obstructive": {"precision": 0.92, "recall": 0.95},
      "central": {"precision": 0.88, "recall": 0.85},
      "mixed": {"precision": 0.89, "recall": 0.91}
    }
  }
}
```

---

## Step 4: Model Conversion for Edge Deployment

### 4.1 Convert to TensorFlow Lite

Convert trained models to TFLite format with INT8 quantization:

```bash
python src/models/export.py \
  --input models/detector/best_model.h5 \
  --output models/v1.0/detector.tflite \
  --quantization int8 \
  --representative-dataset data/splits/calibration
```

Repeat for classifier:
```bash
python src/models/export.py \
  --input models/classifier/best_model.h5 \
  --output models/v1.0/classifier.tflite \
  --quantization int8 \
  --representative-dataset data/splits/calibration
```

### 4.2 Validate Quantized Models

Verify accuracy degradation is <2% (SC-ML-005):

```bash
python src/cli/evaluate.py \
  --detector models/v1.0/detector.tflite \
  --classifier models/v1.0/classifier.tflite \
  --test-data data/splits/test \
  --output reports/quantized_evaluation.json
```

**Acceptance criteria**:
- Accuracy drop ≤2% compared to FP32 models
- Model size ≤50MB (SC-ML-006)

### 4.3 Profile on Target Hardware

Test inference latency on nRF5340:

```bash
python src/cli/profile.py \
  --detector models/v1.0/detector.tflite \
  --classifier models/v1.0/classifier.tflite \
  --hardware nrf5340 \
  --port /dev/ttyACM0 \
  --iterations 100
```

**Expected output**:
```text
Detector Inference: 87.3ms ± 3.2ms (avg ± std)
Classifier Inference: 62.1ms ± 2.8ms
Total Pipeline: 149.4ms ± 4.5ms
Memory Usage: 12.3 MB
Power Consumption: 42 mA @ 3.3V
```

**Acceptance criteria**:
- Inference latency <100ms per model (SC-ML-004)
- Total pipeline <200ms for real-time processing

---

## Step 5: Firmware Compilation

### 5.1 Build Firmware

Compile firmware for nRF5340:

```bash
cd firmware/nrf5340
mkdir build && cd build

cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/arm-none-eabi.cmake \
  -DBOARD=nrf5340dk_nrf5340_cpuapp \
  -DTFLITE_MODEL_DETECTOR=../../../models/v1.0/detector.tflite \
  -DTFLITE_MODEL_CLASSIFIER=../../../models/v1.0/classifier.tflite

make -j8
```

**Output**: `build/zephyr/zephyr.hex` (firmware image)

### 5.2 Flash Firmware

Flash firmware to nRF5340 development kit:

```bash
nrfjprog --program zephyr.hex --chiperase --verify
nrfjprog --reset
```

---

## Step 6: Hardware Setup

### 6.1 Connect Sensors

**MAX30102 (PPG + SpO2)**:
- VCC → 3.3V
- GND → GND
- SDA → P0.26 (I2C SDA)
- SCL → P0.27 (I2C SCL)
- INT → P0.28 (interrupt)

**ADXL345 (Accelerometer)**:
- VCC → 3.3V
- GND → GND
- SDA → P0.26 (shared I2C)
- SCL → P0.27 (shared I2C)

### 6.2 Sensor Calibration

Run calibration routine:

```bash
python scripts/calibrate_sensors.py \
  --port /dev/ttyACM0 \
  --sensors max30102 adxl345 \
  --duration 60
```

**Follow on-screen instructions**: Keep device still, then move gently, then place on test subject's wrist.

---

## Step 7: End-to-End Testing

### 7.1 Simulated Sleep Session

Run a test session with pre-recorded data:

```bash
python tests/integration/test_session.py \
  --hardware nrf5340 \
  --port /dev/ttyACM0 \
  --test-data tests/fixtures/apnea_test_recording.h5 \
  --duration 300
```

**Expected output**:
- Session starts successfully
- Events detected with timestamps
- Session summary generated with AHI score

### 7.2 Validate Against Ground Truth

Compare detected events to clinical annotations:

```bash
python tests/integration/validate_session.py \
  --session-export exports/test_session.json \
  --ground-truth tests/fixtures/apnea_test_annotations.csv \
  --metrics sensitivity specificity ahi_accuracy
```

**Acceptance criteria**:
- Sensitivity ≥95% (SC-002)
- Specificity ≥90% (SC-003)
- AHI accuracy within ±2 events/hour (SC-ML-008)

### 7.3 Battery Life Test

Measure power consumption during 8-hour simulated session:

```bash
python src/cli/profile.py \
  --hardware nrf5340 \
  --port /dev/ttyACM0 \
  --power-profiling \
  --duration 28800  # 8 hours in seconds
```

**Expected results**:
- Average current: <50 mA during active inference
- Total energy: <400 mAh for 8-hour session
- Battery life projection: 2-3 nights on 1000 mAh battery (SC-001)

---

## Step 8: Deployment

### 8.1 Generate Model Metadata

Create model card documenting performance:

```bash
python scripts/generate_model_card.py \
  --detector models/v1.0/detector.tflite \
  --classifier models/v1.0/classifier.tflite \
  --evaluation reports/quantized_evaluation.json \
  --output models/v1.0/MODEL_CARD.md
```

### 8.2 Package Release

Create release bundle:

```bash
python scripts/package_release.py \
  --firmware build/zephyr/zephyr.hex \
  --models models/v1.0/ \
  --output releases/SleepApneaEdge-v1.0.zip
```

**Release contents**:
- Firmware image (.hex)
- TFLite models (detector.tflite, classifier.tflite)
- Model metadata (MODEL_CARD.md, metadata.json)
- Flashing instructions (INSTALL.md)
- Checksums (SHA256SUMS)

---

## Troubleshooting

### Issue: Model accuracy below 90% threshold

**Solutions**:
1. Increase training data (use more NSRR datasets)
2. Tune hyperparameters (learning rate, model architecture)
3. Verify data preprocessing (check for label errors)
4. Try quantization-aware training instead of post-training quantization

### Issue: Inference latency exceeds 100ms

**Solutions**:
1. Reduce model size (fewer layers, smaller hidden dimensions)
2. Verify CMSIS-NN optimizations are enabled in build
3. Increase CPU clock frequency (balance with power consumption)
4. Profile bottlenecks with ARM Development Studio

### Issue: High false positive rate (>10%)

**Solutions**:
1. Adjust detection threshold (increase from 0.7 to 0.8)
2. Improve signal quality assessment (reject noisy windows)
3. Add temporal consistency check (require multiple consecutive detections)
4. Collect more diverse training data (different demographics)

### Issue: Sensor connection failures

**Solutions**:
1. Verify I2C wiring (SDA/SCL pullup resistors may be needed)
2. Check sensor power supply (stable 3.3V, sufficient current)
3. Test sensors individually with vendor example code
4. Enable I2C debugging in firmware configuration

### Issue: Battery depletes faster than expected

**Solutions**:
1. Verify adaptive sampling is enabled (check firmware config)
2. Reduce PPG LED brightness (may affect signal quality)
3. Lower baseline sampling rate (test impact on accuracy)
4. Enable sensor gating during stable periods

---

## Next Steps

After completing this quickstart:

1. **Clinical Validation**: Test on real subjects with concurrent polysomnography
2. **Regulatory Compliance**: Prepare documentation for FDA 510(k) or CE marking
3. **User Interface**: Develop companion mobile app for data visualization
4. **Advanced Features**: Add real-time alerts, cloud sync, multi-user profiles
5. **Optimization**: Fine-tune power management, explore model pruning

---

## Additional Resources

- **API Documentation**: See `contracts/inference-api.md`
- **Data Model**: See `data-model.md`
- **Research Notes**: See `research.md`
- **Constitution**: See `.specify/memory/constitution.md` for development principles

**Support**:
- GitHub Issues: https://github.com/your-org/SleepApneaEdge/issues
- Email: support@your-org.com
- Slack: #sleep-apnea-edge

---

**Last Updated**: 2025-11-04 | **Version**: 1.0
