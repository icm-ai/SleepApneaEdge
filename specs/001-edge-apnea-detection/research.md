# Research Document: Edge-Based Sleep Apnea Detection

**Feature**: `001-edge-apnea-detection` | **Date**: 2025-11-04 | **Phase**: 0 (Research)

## Overview

This document provides research-backed decisions for technical unknowns identified in the implementation plan. Each section addresses a specific clarification need from the Technical Context, providing a recommended decision, rationale, and alternatives considered.

---

## 1. Edge ML Framework Selection

### Decision: TensorFlow Lite Micro (TFLM)

**Rationale**: TensorFlow Lite excels in embedded device deployment with superior hardware acceleration support through its delegate system (CPU, GPU, DSP, specialized chips), smaller binary sizes (4MB vs 12MB for equivalent models), and mature optimization features including 8-bit and 16-bit quantization. TensorFlow Lite Micro specifically targets microcontrollers with kilobytes of memory, making it ideal for wearable sleep monitoring devices. TFLM kernels automatically use CMSIS-NN optimizations on ARM Cortex-M processors with no additional developer work, maximizing performance while minimizing power consumption.

**Alternatives Considered**:
- **PyTorch Mobile**: Rejected due to larger binary sizes, limited hardware acceleration options beyond CPU/GPU, and heavier runtime requirements. While PyTorch Mobile maintains a more intuitive development workflow, TFLite's power efficiency and optimization features are critical for 8-10 hour battery life requirements.
- **ONNX Runtime**: Rejected due to less mature embedded tooling compared to TFLite Micro and lack of specific microcontroller-optimized variants.

---

## 2. Target Hardware Platform

### Decision: Nordic nRF5340 (dual ARM Cortex-M33) or Analog Devices MAX32664 Sensor Hub

**Rationale**: For sleep monitoring wearables, the nRF5340 offers dual Cortex-M33 processors (one for network operations like BLE, one for ML inference), 29% lower TX power and 41% lower RX power versus nRF52840, and sleep current as low as 1.1 µA enabling multi-day battery life. Alternatively, the MAX32664 is purpose-built as a biometric sensor hub with embedded sleep monitoring algorithms, ARM Cortex-M4 at 96MHz, ultra-low power consumption, and tiny form factor (1.6mm x 1.6mm). Both platforms support TFLite Micro with CMSIS-NN acceleration and provide sufficient memory for 50MB model constraints.

**Alternatives Considered**:
- **Raspberry Pi Zero**: Rejected due to significantly higher power consumption (100mA+ idle) making 8-10 hour battery operation impractical for wearables. Better suited for prototyping than production wearables.
- **STM32 Cortex-M4**: Viable alternative with X-CUBE-AI support for NN optimization, but lacks integrated sensor hub capabilities and requires more external components than MAX32664 solution.
- **Ambiq Apollo4**: Strong alternative with industry-leading power efficiency (SPOT technology), purpose-built for wearable AI applications, but less mature ecosystem than Nordic for BLE-connected health devices.

---

## 3. Embedded Firmware Language

### Decision: C (ANSI C99)

**Rationale**: C is the dominant language for ML inference on ARM Cortex-M processors in wearable devices. TensorFlow Lite Micro generates ANSI C code optimized for embedded targets, CMSIS-NN neural network kernels are written in C, and frameworks like X-CUBE-AI export to ANSI C libraries. C provides predictable memory usage, minimal runtime overhead, and universal compiler support across all embedded platforms. For resource-constrained wearables requiring deterministic behavior and maximum battery efficiency, C is the industry standard.

**Alternatives Considered**:
- **C++**: Supported by all embedded toolchains and some ML frameworks (emlearn supports C++), but adds runtime complexity (constructors, exceptions, vtables) that can impact determinism and increase code size. Acceptable for higher-level firmware if memory permits, but C preferred for ML inference path.
- **Rust**: Emerging in embedded space with memory safety guarantees, but lacks mature TinyML framework support and has limited ARM Cortex-M ecosystem compared to C. Too experimental for production medical wearable.

---

## 4. Sensor Selection and Data Acquisition

### Decision: Multi-sensor approach combining PPG, SpO2, and Accelerometer

**Rationale**: Research shows combining multiple sensor modalities provides superior sleep apnea detection accuracy. PPG (photoplethysmography) measures blood volume changes non-invasively and estimates autonomic variability. SpO2 (pulse oximetry) detects oxygen desaturation events that characterize apnea episodes and has been shown to outperform ECG/PPG alone in classification tasks. Accelerometer captures respiratory effort through diaphragm/chest movement and body position, enabling differentiation between obstructive (effort present) and central (no effort) apnea. This sensor fusion approach enables the ≥90% classification accuracy requirement for event types.

**Alternatives Considered**:
- **Single SpO2 sensor**: Simpler but cannot distinguish obstructive from central apnea (requires respiratory effort signal). Insufficient for FR-006 event classification requirement.
- **Nasal airflow sensor**: Provides direct respiratory signal but impractical for wearable devices (user compliance issues with nasal cannula during sleep). Better suited for clinical polysomnography.
- **ECG + PPG only**: Widely used but accelerometer adds critical respiratory effort information at minimal power cost, improving classification accuracy with negligible complexity increase.

**Sensor Data Acquisition Libraries**:
- **Nordic nRF SDK**: For nRF5340, provides BLE connectivity and sensor interface drivers
- **MAX32664 Embedded Algorithms**: Pre-validated biometric algorithms for PPG/SpO2 processing
- **CMSIS-DSP**: ARM digital signal processing library for filtering and feature extraction

---

## 5. Sleep Apnea Validation Dataset

### Decision: National Sleep Research Resource (NSRR) - Sleep Heart Health Study

**Rationale**: NSRR provides free, de-identified polysomnography data from well-characterized research cohorts. The Sleep Heart Health Study dataset contains 9,736 overnight PSG recordings with clinical annotations including apnea event labels, types, and AHI scores. Data includes ECG, SpO2, respiratory signals, and sleep staging - matching our target sensor modalities. This established dataset enables validation against clinical ground truth (SC-002, SC-ML-001) and comparison with published research. NSRR's scale supports robust train/validation/test splits for ≥90% accuracy targets.

**Alternatives Considered**:
- **PSG-Audio Dataset**: Only 50 patients with audio recordings, insufficient scale for training deep learning models requiring thousands of examples. Useful for augmentation but not primary dataset.
- **NCH Sleep DataBank**: Recently published (April 2025) with 100 patients and wearable device data, but smaller scale than NSRR and less established for benchmarking. Good candidate for additional validation after initial development.
- **Clinical Partnership**: Most representative of target population but requires IRB approval, data access agreements, and delays initial development. Recommended for final clinical validation phase, not initial model training.

---

## 6. Edge Hardware Testing Framework

### Decision: pytest + Custom Edge Profiling Tools

**Rationale**: Python-based pytest framework handles model development testing (preprocessing, training, accuracy validation) with rich ecosystem of assertion libraries and fixtures. For edge device testing, custom profiling tools built with MATLAB/Simulink or Edge Impulse CLI measure on-device inference latency, memory usage, and power consumption. MLPerf Tiny benchmark (v0.5+) provides standardized embedded ML benchmarking for comparing performance across platforms. This hybrid approach maintains development velocity during training while ensuring rigorous edge performance validation before deployment.

**Alternatives Considered**:
- **Unity Test Framework**: C-based unit testing for embedded systems, but requires testing directly on target hardware, slowing development iteration. Better suited for firmware-level unit tests than ML model validation.
- **Edge Impulse Platform**: End-to-end solution for TinyML with integrated testing, but proprietary platform lock-in and less flexibility for custom algorithms. Useful for prototyping but not recommended as sole testing framework.
- **MATLAB/Simulink Only**: Comprehensive tooling for embedded ML workflow including quantization and profiling, but expensive licensing and overkill for this single-feature project. Recommended only if organization already has licenses.

**Testing Strategy**:
- **Python (pytest)**: Unit tests, integration tests, model accuracy validation on development machine
- **Custom CLI Tools**: Profiling scripts that deploy models to target hardware and measure latency/power
- **MLPerf Tiny**: Standardized benchmarking for cross-platform performance comparison

---

## 7. Model Quantization Strategy

### Decision: INT8 Post-Training Quantization with CMSIS-NN

**Rationale**: INT8 (8-bit integer) quantization reduces model size by 4x versus FP32 and accelerates inference through hardware-optimized integer operations on ARM Cortex-M processors. TensorFlow Lite kernels automatically use CMSIS-NN optimized implementations for INT8 operations, maximizing performance on target Cortex-M33/M4 devices. CMSIS-NN follows TFLite quantization specification and leverages DSP and M-Profile Vector Extension (MVE) instructions for matrix operations. Post-training quantization (PTQ) requires no retraining, simplifying deployment while typically maintaining <2% accuracy degradation per SC-ML-005.

**Alternatives Considered**:
- **FP32 (32-bit float)**: Highest accuracy but 4x larger model size and slower inference. Incompatible with 50MB model constraint and <100ms latency requirement on microcontroller hardware.
- **INT16 (16-bit integer)**: Moderate compression (2x reduction) with better accuracy preservation, but CMSIS-NN optimizations focus on INT8. Consider if INT8 accuracy degradation exceeds 2% threshold.
- **Quantization-Aware Training (QAT)**: Simulates quantization during training for better accuracy retention, but adds training complexity. Use only if PTQ fails to meet ≥90% accuracy requirement.

---

## 8. Power Management Strategy

### Decision: Adaptive Sampling with Sensor Gating

**Rationale**: To achieve 8-10 hour battery life, implement adaptive sampling that increases sensor polling frequency only during suspected apnea events. During normal breathing, reduce PPG/accelerometer sampling to 10-25 Hz baseline. When SpO2 drops or breathing irregularity detected, increase to 50-100 Hz for detailed event characterization. Gate power-hungry sensors (PPG LED) off during stable periods. Nordic nRF5340's dual-core architecture enables network processor to sleep while application processor handles inference. Combined with 1.1 µA sleep current, adaptive sampling can extend battery life 2-3x versus continuous high-rate sampling.

**Alternatives Considered**:
- **Continuous High-Rate Sampling**: Simplest approach (50-100 Hz all sensors) but depletes battery in 3-5 hours, failing FR-012 requirement. Rejected due to power constraints.
- **Wake-on-Movement Only**: Minimal power but misses apnea events during still sleep periods (majority of sleep). Insufficient for continuous monitoring requirement.
- **External Power Bank**: Bypasses wearable constraint, making device impractical for nightly use. Defeats wearable form factor advantage.

---

## Implementation Recommendations

Based on this research, proceed with:

1. **Platform**: Nordic nRF5340 development kit for prototyping (or MAX32664 if partnering with sensor manufacturer)
2. **Framework**: TensorFlow Lite Micro with INT8 quantization
3. **Language**: C99 for inference firmware, Python for model training
4. **Sensors**: PPG + SpO2 + 3-axis accelerometer (evaluate MAX30102 or MAX32664 integrated solution)
5. **Dataset**: NSRR Sleep Heart Health Study for initial training/validation
6. **Testing**: pytest for Python development, custom edge profiling tools for on-device validation

These choices prioritize simplicity (Constitution Principle I) while meeting all functional requirements and success criteria.
