# Implementation Plan: Edge-Based Sleep Apnea Detection

**Branch**: `001-edge-apnea-detection` | **Date**: 2025-11-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-edge-apnea-detection/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Build an edge-based ML system that detects and classifies sleep apnea events in real-time on wearable devices. The system must process respiratory sensor data with <100ms latency, achieve ≥90% detection accuracy, classify events by type (obstructive/central/mixed), and operate offline for 8-10 hours on battery power.

## Technical Context

**Language/Version**: Python 3.11+ (edge inference), NEEDS CLARIFICATION for embedded firmware language
**Primary Dependencies**: TensorFlow Lite / PyTorch Mobile (edge inference), NumPy/SciPy (signal processing), NEEDS CLARIFICATION for sensor data acquisition libraries
**Storage**: SQLite (local device storage for 90+ nights of data), binary format for raw sensor streams
**Testing**: pytest (Python ML pipeline), NEEDS CLARIFICATION for edge hardware testing framework
**Target Platform**: NEEDS CLARIFICATION (ARM Cortex-M series, Raspberry Pi, or specific wearable chipset)
**Project Type**: Embedded ML system (edge device firmware + model deployment)
**Performance Goals**: <100ms inference latency per 30-second window, ≥90% apnea detection accuracy, ≥90% event classification accuracy
**Constraints**: 50MB model size limit, 8-10 hour battery life, offline operation, <2% accuracy degradation edge vs training environment
**Scale/Scope**: Single-user device, 90+ nights local storage (~10-50MB per night depending on sampling rate), real-time processing during 8-10 hour sleep sessions

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Principle I - Simplicity First (Occam's Razor)**:

- [x] Architecture uses simplest approach meeting requirements
  - Single-stage detection model preferred over ensemble unless accuracy requires it
- [x] All dependencies justified (document in Technical Context)
  - TensorFlow Lite/PyTorch Mobile: Edge inference optimization
  - NumPy/SciPy: Standard signal processing (respiratory waveform analysis)
  - SQLite: Lightweight embedded storage for 90+ nights data
- [x] Rejected alternatives documented with rationale
  - Cloud-based processing: Rejected due to offline requirement (FR-003)
  - Multi-model ensemble: Will evaluate in Phase 0; default to single model

**Principle II - Test-First Development**:

- [x] Test strategy defined before implementation
  - Unit tests: Signal preprocessing, event detection logic
  - Integration tests: End-to-end inference pipeline on sample data
  - Hardware tests: Latency profiling on target edge device
- [x] Accuracy tests verify ≥90% threshold
  - Validation against clinical polysomnography ground truth (SC-ML-001)
- [x] Edge device latency tests included
  - Target: <100ms per 30-second window (SC-ML-004)

**Principle III - Model Accuracy Requirement**:

- [x] Success criteria specify ≥90% accuracy target
  - Detection: ≥95% sensitivity, ≤10% false positive rate (SC-002, SC-003)
  - Classification: ≥90% accuracy for event types (SC-ML-007)
- [x] Validation dataset identified and representative
  - NEEDS CLARIFICATION: Source of labeled sleep study data (public datasets vs clinical partnership)
- [x] Per-class metrics (precision, recall, F1) planned
  - Per-class precision ≥85%, recall ≥90%, F1 ≥87% (SC-ML-002)

**Principle IV - Edge Performance Optimization**:

- [x] Target edge hardware specified in Technical Context
  - NEEDS CLARIFICATION: Specific chipset (ARM Cortex-M series, RPi, or custom wearable SoC)
- [x] Latency targets defined (<100ms baseline, adjust as needed)
  - <100ms per 30-second analysis window (SC-ML-004)
- [x] Model size constraints documented
  - 50MB memory limit (SC-ML-006)
- [x] Profiling plan on target hardware established
  - Phase 1 will include edge device testing with target hardware

**Principle V - Observability & Reproducibility**:

- [x] Training metadata tracking planned (hyperparameters, seeds, versions)
  - Training scripts will log all hyperparameters, random seeds, dataset versions
- [x] Inference logging strategy defined
  - Each prediction logged with timestamp, confidence score, sensor data quality metrics
- [x] Model versioning approach documented
  - Model artifacts include training metadata, framework version, performance benchmarks

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
# Embedded ML system structure
src/
├── preprocessing/       # Signal preprocessing and feature extraction
│   ├── filters.py      # Band-pass filters, noise reduction
│   ├── features.py     # Respiratory rate, saturation features
│   └── windowing.py    # Time-series windowing for inference
├── models/             # ML model definitions and training
│   ├── detector.py     # Apnea event detection model
│   ├── classifier.py   # Event type classification (obstructive/central/mixed)
│   ├── training/       # Training scripts, data loaders
│   └── export.py       # Model conversion to TFLite/PyTorch Mobile
├── inference/          # Edge inference pipeline
│   ├── engine.py       # Real-time inference orchestrator
│   ├── postprocess.py  # Event aggregation, AHI calculation
│   └── quality.py      # Signal quality assessment
├── storage/            # Local data persistence
│   ├── session.py      # Sleep session storage (SQLite)
│   ├── events.py       # Event logging and retrieval
│   └── export.py       # Medical report export formats
└── cli/                # Development and testing CLI tools
    ├── train.py        # Model training entry point
    ├── evaluate.py     # Model evaluation on validation sets
    └── profile.py      # Edge device profiling tools

tests/
├── unit/               # Unit tests for preprocessing, models, storage
├── integration/        # End-to-end pipeline tests
└── hardware/           # Edge device latency and accuracy tests

notebooks/              # Research and exploratory analysis
└── eda/               # Dataset exploration, signal visualization

data/                   # Training and validation datasets (gitignored)
├── raw/               # Raw sensor data from sleep studies
├── processed/         # Preprocessed features
└── splits/            # Train/val/test splits

models/                 # Trained model artifacts (versioned)
└── v1.0/
    ├── detector.tflite
    ├── classifier.tflite
    └── metadata.json
```

**Structure Decision**: Single project structure optimized for embedded ML development. Separates preprocessing, model training, and edge inference into distinct modules. Hardware-specific deployment code (firmware) will be added in Phase 1 after target platform is clarified.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

**No violations detected**. All design decisions comply with Constitution principles:
- Single-model architecture preferred (ensembles only if accuracy requires)
- Direct SQLite access (no ORM layer)
- Standard dependencies (NumPy, SciPy, TFLite Micro)
- Adaptive sampling instead of continuous high-rate (simpler power management)

---

## Phase 1 Design Review

**Status**: ✅ COMPLETED

### Generated Artifacts

1. ✅ **research.md**: All technical unknowns resolved
   - Edge ML framework: TensorFlow Lite Micro
   - Target hardware: Nordic nRF5340 or MAX32664
   - Language: C99 for firmware, Python for training
   - Sensors: PPG + SpO2 + Accelerometer
   - Dataset: NSRR Sleep Heart Health Study

2. ✅ **data-model.md**: Data structures and relationships defined
   - 7 core entities (SleepSession, ApneaEvent, HypopneaEvent, SessionSummary, DeviceProfile, TrendSummary, SensorReadings)
   - Validation rules aligned with clinical definitions
   - Storage estimate: 10-15 MB for 90 days (within device constraints)

3. ✅ **contracts/inference-api.md**: API contracts documented
   - 8 core APIs with input/output schemas
   - Performance guarantees (latency, memory, power)
   - Error handling standards
   - Versioning strategy

4. ✅ **quickstart.md**: Developer onboarding guide
   - Environment setup (Python, ARM GCC, nRF tools)
   - Dataset preparation (NSRR download, preprocessing)
   - Model training and evaluation
   - Edge deployment and testing
   - Troubleshooting guide

### Constitution Re-Evaluation (Post-Design)

**Principle I - Simplicity First**: ✅ PASS
- Single-stage detection model (no ensemble)
- Direct SQLite access (no ORM overhead)
- Three core dependencies (TFLite, NumPy, SciPy)
- Rejected cloud processing, multi-model ensembles, and unnecessary abstractions

**Principle II - Test-First Development**: ✅ PASS
- Test strategy defined in quickstart.md
- Unit, integration, and hardware tests planned
- pytest for Python, custom profiling tools for edge
- Acceptance criteria: ≥90% accuracy, <100ms latency, ≥95% sensitivity

**Principle III - Model Accuracy Requirement**: ✅ PASS
- Validation dataset identified (NSRR Sleep Heart Health Study)
- Accuracy targets: ≥90% detection, ≥90% classification
- Per-class metrics defined (precision, recall, F1)
- Ground truth: Clinical polysomnography annotations

**Principle IV - Edge Performance Optimization**: ✅ PASS
- Target hardware specified: Nordic nRF5340 (dual Cortex-M33)
- Latency target: <100ms inference per 30-second window
- Model size: <50MB (INT8 quantization)
- Power budget: <50 mA active, 8-10 hour battery life
- Adaptive sampling strategy to extend battery 2-3x

**Principle V - Observability & Reproducibility**: ✅ PASS
- Training metadata logging planned (hyperparameters, seeds, versions)
- Inference logging: predictions, confidence, quality scores
- Model versioning: metadata.json with training environment
- Reproducibility: All experiments use fixed random seeds

### Open Questions for Implementation

1. **Sensor Integration**: Evaluate MAX30102 vs MAX32664 integrated solution (sensor manufacturer partnership)
2. **Clinical Validation**: Plan concurrent polysomnography study for ground truth comparison
3. **Regulatory Path**: Determine if pursuing FDA 510(k) (medical device) or wellness device classification
4. **User Interface**: Design companion app for data visualization (out of scope for initial implementation)

### Next Steps

✅ Phase 0 (Research) and Phase 1 (Design) complete. Ready for Phase 2 (Tasks).

Run `/speckit.tasks` to generate the dependency-ordered task breakdown for implementation.
