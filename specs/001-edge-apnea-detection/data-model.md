# Data Model: Edge-Based Sleep Apnea Detection

**Feature**: `001-edge-apnea-detection` | **Date**: 2025-11-04 | **Phase**: 1 (Design)

## Overview

This document defines the comprehensive data structures, storage formats, relationships, and validation rules for the SleepKit-based sleep apnea detection system. The system uses a hybrid storage approach optimized for edge deployment:

- **HDF5**: Preprocessed signals, extracted features, and training datasets
- **SQLite**: Session metadata, events, summaries, and device profiles
- **TFLite**: Quantized models with embedded metadata
- **YAML/JSON**: Experiment configurations and model parameters

All data is stored locally on the device with a 90-day retention policy for raw sessions and indefinite retention for aggregated summaries.

---

## Storage Architecture

### File Organization

```text
/storage/
├── datasets/                  # HDF5 training datasets
│   ├── mesa.h5               # MESA dataset (6,814 subjects)
│   ├── cmidss.h5             # CMIDSS dataset (300 subjects)
│   ├── ysyw.h5               # YSYW dataset (1,983 recordings)
│   └── custom/               # Self-collected datasets
│       └── session_*.h5      # Individual session HDF5 files
│
├── features/                  # Extracted features (HDF5)
│   ├── acoustic/             # MFCC, Mel spectrograms
│   │   └── mfcc_features.h5
│   ├── multimodal/           # Fused PPG+SpO2+IMU features
│   │   └── fusion_features.h5
│   └── clinical/             # Full signal suite features
│       └── clinical_features.h5
│
├── models/                    # Trained models and metadata
│   ├── pretrained/           # Downloaded from SleepKit zoo
│   │   ├── apnea_detector_v1.tflite
│   │   └── apnea_detector_v1.json
│   ├── finetuned/            # Custom trained models
│   │   ├── acoustic_apnea.tflite
│   │   ├── acoustic_apnea.json
│   │   ├── multimodal_lightweight.tflite
│   │   └── multimodal_lightweight.json
│   └── active/               # Currently deployed model
│       ├── current.tflite
│       └── current.json
│
├── configs/                   # Experiment configurations
│   ├── apnea/
│   │   ├── train.yaml
│   │   ├── evaluate.yaml
│   │   └── export.yaml
│   ├── acoustic/
│   │   └── train_acoustic.yaml
│   └── multimodal/
│       └── train_multimodal.yaml
│
└── sessions/                  # Runtime session data
    ├── sessions.db           # SQLite database (metadata, events, summaries)
    └── signals/              # Raw signal buffers (temporary)
        └── session_*.h5      # Deleted after session processing
```

---

## Part I: HDF5 Data Structures

### 1. Signal Dataset Schema (PhysioKit Compatible)

HDF5 files for raw and preprocessed physiological signals follow PhysioKit conventions.

#### Structure

```python
session_YYYYMMDD_HHMMSS.h5
├── /metadata                   # Session metadata (attributes)
│   ├── session_id: UUID
│   ├── start_time: ISO8601 timestamp
│   ├── duration_seconds: float
│   ├── device_id: string
│   ├── firmware_version: string
│   ├── sampling_rates: JSON dict
│   └── signal_quality: float [0.0-1.0]
│
├── /signals                    # Raw sensor signals (datasets)
│   ├── ppg                    # Photoplethysmography
│   │   ├── data: float32[n_samples]
│   │   ├── attrs: {sampling_rate: 25, units: "AU", sensor: "MAX30102"}
│   │   └── timestamps: float64[n_samples]  # Unix timestamps
│   │
│   ├── spo2                   # Blood oxygen saturation
│   │   ├── data: float32[n_samples]
│   │   ├── attrs: {sampling_rate: 1, units: "percent", range: [0, 100]}
│   │   └── timestamps: float64[n_samples]
│   │
│   ├── accelerometer          # 3-axis acceleration
│   │   ├── data: float32[n_samples, 3]  # [x, y, z]
│   │   ├── attrs: {sampling_rate: 25, units: "m/s^2", axes: ["x", "y", "z"]}
│   │   └── timestamps: float64[n_samples]
│   │
│   ├── audio (optional)       # Acoustic signals for snoring detection
│   │   ├── data: float32[n_samples]
│   │   ├── attrs: {sampling_rate: 8000, units: "Pa", codec: "PCM16"}
│   │   └── timestamps: float64[n_samples]
│   │
│   └── respiratory_effort (optional)  # Chest/abdominal effort bands
│       ├── data: float32[n_samples]
│       ├── attrs: {sampling_rate: 10, units: "AU", sensor: "piezo"}
│       └── timestamps: float64[n_samples]
│
├── /preprocessed              # Filtered/normalized signals
│   ├── ppg_filtered          # Bandpass filtered PPG (0.5-8 Hz)
│   │   ├── data: float32[n_samples]
│   │   └── attrs: {filter: "butterworth", order: 4, lowcut: 0.5, highcut: 8.0}
│   │
│   ├── spo2_interpolated     # Gap-filled SpO2
│   │   ├── data: float32[n_samples]
│   │   └── attrs: {method: "linear", max_gap_seconds: 5}
│   │
│   └── accel_magnitude       # Vector magnitude of acceleration
│       ├── data: float32[n_samples]
│       └── attrs: {formula: "sqrt(x^2 + y^2 + z^2)"}
│
├── /annotations               # Ground truth labels (for training data)
│   ├── apnea_events          # Apnea event windows
│   │   ├── start_indices: int64[n_events]
│   │   ├── end_indices: int64[n_events]
│   │   ├── event_types: string[n_events]  # ["obstructive", "central", "mixed"]
│   │   └── attrs: {source: "polysomnography", annotator: "clinician_id"}
│   │
│   ├── hypopnea_events       # Hypopnea event windows
│   │   ├── start_indices: int64[n_events]
│   │   ├── end_indices: int64[n_events]
│   │   ├── reduction_percent: float32[n_events]
│   │   └── attrs: {source: "polysomnography"}
│   │
│   └── sleep_stages (optional)  # Sleep staging annotations
│       ├── stage_labels: int8[n_epochs]  # 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
│       ├── epoch_duration_seconds: 30
│       └── attrs: {source: "polysomnography"}
│
└── /quality                   # Signal quality metrics
    ├── ppg_quality           # Per-window quality scores
    │   ├── scores: float32[n_windows]
    │   ├── window_indices: int64[n_windows, 2]  # [start, end]
    │   └── attrs: {threshold: 0.7, method: "SNR"}
    │
    └── overall_quality       # Session-level quality
        └── attrs: {mean_quality: 0.87, low_quality_percentage: 5.2}
```

#### Validation Rules

- All `data` datasets must have corresponding `timestamps` with matching length
- `sampling_rate` attribute must match actual temporal resolution of timestamps
- Signal units must follow PhysioKit conventions (AU for unitless, percent, m/s^2, etc.)
- Missing data represented by NaN values (not -999 or sentinel values)
- Annotations must reference valid indices within signal data range
- Quality scores must be in range [0.0, 1.0]

#### Example Access (Python)

```python
import h5py
import numpy as np

with h5py.File('session_20251104_223000.h5', 'r') as f:
    # Read PPG signal
    ppg_data = f['/signals/ppg/data'][:]
    ppg_sampling_rate = f['/signals/ppg'].attrs['sampling_rate']

    # Read apnea annotations
    apnea_starts = f['/annotations/apnea_events/start_indices'][:]
    apnea_types = f['/annotations/apnea_events/event_types'][:]

    # Session metadata
    session_id = f['/metadata'].attrs['session_id']
    signal_quality = f['/metadata'].attrs['signal_quality']
```

---

### 2. Feature Dataset Schema

Extracted features for model training, stored in HDF5 format.

#### Structure

```python
features_VARIANT.h5
├── /metadata
│   ├── feature_version: "1.0"
│   ├── extraction_date: ISO8601
│   ├── source_dataset: "mesa"  # or "cmidss", "ysyw", "custom"
│   ├── window_size_seconds: 30
│   ├── hop_size_seconds: 15
│   └── feature_extractor: "acoustic_mfcc_v1"
│
├── /subjects                  # Subject-level organization
│   ├── subject_0001
│   │   ├── features: float32[n_windows, n_features]
│   │   ├── labels: int8[n_windows]  # 0=normal, 1=apnea, 2=hypopnea
│   │   ├── timestamps: float64[n_windows]
│   │   └── attrs: {session_ids: ["uuid1", "uuid2"], ahi_score: 18.5}
│   │
│   ├── subject_0002
│   │   └── ...
│   └── ...
│
├── /feature_names             # Feature metadata
│   └── attrs: {
│       names: ["mfcc_1", "mfcc_2", ..., "spectral_centroid"],
│       descriptions: ["First MFCC coefficient", ...],
│       units: ["AU", "AU", ..., "Hz"]
│   }
│
└── /splits                    # Train/val/test splits
    ├── train_subjects: string[n_train]  # ["subject_0001", ...]
    ├── val_subjects: string[n_val]
    ├── test_subjects: string[n_test]
    └── attrs: {split_method: "stratified", random_seed: 42}
```

#### Feature Types by Variant

**Acoustic Features** (`acoustic/mfcc_features.h5`):
- MFCC coefficients (13 coefficients)
- Mel spectrogram statistics (mean, std, max per frequency band)
- Spectral features (centroid, bandwidth, rolloff)
- Zero-crossing rate
- RMS energy
- Feature shape: `[n_windows, 64]`

**Multimodal Features** (`multimodal/fusion_features.h5`):
- PPG features: heart rate, HRV metrics, pulse morphology (20 features)
- SpO2 features: mean, std, min, desaturation events (8 features)
- Accelerometer features: movement intensity, sleep position, turn count (12 features)
- Time-domain cross-correlations (PPG-SpO2, PPG-accel) (6 features)
- Feature shape: `[n_windows, 46]`

**Clinical Features** (`clinical/clinical_features.h5`):
- All multimodal features (46 features)
- Respiratory effort features (if available): amplitude, frequency, regularity (10 features)
- Advanced HRV features: spectral power, Poincaré plot metrics (12 features)
- Signal quality indicators (4 features)
- Feature shape: `[n_windows, 72]`

#### Validation Rules

- `n_features` dimension must match `len(feature_names.attrs['names'])`
- Labels must be in range [0, 2] (normal, apnea, hypopnea)
- All subjects in splits must exist in `/subjects` group
- Train/val/test splits must be mutually exclusive
- Feature values should be normalized (mean ≈ 0, std ≈ 1) or in documented range

---

### 3. Dataset Manifest Schema

Master index for training datasets (MESA, CMIDSS, YSYW).

#### Structure

```python
mesa.h5
├── /manifest
│   ├── dataset_name: "MESA"
│   ├── version: "2018"
│   ├── total_subjects: 6814
│   ├── total_recordings: 6814
│   ├── source_url: "https://sleepdata.org/datasets/mesa"
│   ├── citation: "Chen et al., 2015, Sleep"
│   └── preprocessing_version: "sleepkit_v1.0"
│
├── /subjects
│   ├── subject_0001
│   │   ├── demographics
│   │   │   └── attrs: {age: 62, sex: "M", bmi: 28.3, ahi_clinical: 15.2}
│   │   ├── recordings
│   │   │   ├── recording_001
│   │   │   │   ├── signals -> /storage/datasets/mesa/subject_0001_rec_001.h5
│   │   │   │   ├── features -> /storage/features/multimodal/mesa_subject_0001.h5
│   │   │   │   └── attrs: {date: "2010-03-15", duration_hours: 7.8, quality: 0.91}
│   │   │   └── ...
│   │   └── clinical_outcomes
│   │       └── attrs: {ahi: 15.2, severity: "mild", predominant_type: "obstructive"}
│   └── ...
│
├── /statistics
│   ├── ahi_distribution: float32[n_bins]
│   ├── bin_edges: float32[n_bins + 1]
│   ├── severity_counts: {normal: 3245, mild: 1829, moderate: 987, severe: 753}
│   └── attrs: {mean_ahi: 8.7, median_ahi: 4.2, std_ahi: 12.3}
│
└── /quality_control
    ├── excluded_subjects: string[n_excluded]  # Low quality or missing data
    ├── exclusion_reasons: string[n_excluded]
    └── attrs: {inclusion_criteria: "quality >= 0.7, duration >= 4 hours"}
```

#### Cross-Dataset Harmonization

When combining MESA, CMIDSS, and YSYW:

- All signals resampled to common rates (PPG: 25 Hz, SpO2: 1 Hz, Accel: 25 Hz)
- Unified annotation format (start/end indices, event types)
- Consistent signal units and preprocessing pipelines
- Dataset source tracked in metadata for stratified splitting

---

## Part II: SQLite Database Schema

### Entity Relationship Diagram

```
DeviceProfile
    |
    | (1:N)
    v
SleepSession <──(1:1)──> SessionSummary
    |
    | (1:N)
    v
ApneaEvent / HypopneaEvent
    |
    | (1:1, optional)
    v
SensorReadings

DeviceProfile ──(1:N)──> TrendSummary
```

---

### 1. SleepSession

Represents a single continuous monitoring period (typically 6-10 hours of sleep).

#### Schema

```sql
CREATE TABLE sleep_session (
    session_id TEXT PRIMARY KEY,           -- UUID
    device_id TEXT NOT NULL,               -- Foreign key to device_profile
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,                    -- NULL if session ongoing
    total_duration_minutes INTEGER,        -- Derived from end_time - start_time
    quality_score REAL NOT NULL CHECK(quality_score BETWEEN 0.0 AND 1.0),
    firmware_version TEXT NOT NULL,
    model_version TEXT NOT NULL,           -- ML model version (e.g., "v1.0")
    hdf5_path TEXT,                        -- Path to session HDF5 file
    state TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(state IN ('ACTIVE', 'COMPLETED', 'ABORTED')),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (device_id) REFERENCES device_profile(device_id),
    CHECK (end_time IS NULL OR end_time > start_time)
);

CREATE INDEX idx_sleep_session_start_time ON sleep_session(start_time);
CREATE INDEX idx_sleep_session_device_id ON sleep_session(device_id);
CREATE INDEX idx_sleep_session_state ON sleep_session(state);
```

#### Fields

- `session_id` (UUID, primary key): Unique identifier for the sleep session
- `device_id` (string, foreign key): Reference to DeviceProfile
- `start_time` (timestamp): When monitoring began
- `end_time` (timestamp, nullable): When monitoring ended (null if ongoing)
- `total_duration_minutes` (integer): Total session length in minutes
- `quality_score` (float, 0.0-1.0): Overall signal quality (0=poor, 1=excellent)
- `firmware_version` (string): Firmware version running during session
- `model_version` (string): ML model version used for inference (e.g., "v1.0")
- `hdf5_path` (string, nullable): Path to corresponding HDF5 file (if saved)
- `state` (enum): Session state - `ACTIVE`, `COMPLETED`, `ABORTED`
- `created_at` (timestamp): Record creation timestamp
- `updated_at` (timestamp): Last update timestamp

#### Validation Rules

- `end_time` must be after `start_time` if not null
- `total_duration_minutes` must match `(end_time - start_time)` when session completes
- `quality_score` must be between 0.0 and 1.0
- Sessions with `quality_score < 0.3` should trigger user alert for poor sensor contact
- Only one session per device can be in `ACTIVE` state at a time

#### State Transitions

```
[CREATED] -> ACTIVE -> COMPLETED
              |
              v
           ABORTED (if user ends early or device error)
```

---

### 2. ApneaEvent

Represents a detected breathing pause ≥10 seconds.

#### Schema

```sql
CREATE TABLE apnea_event (
    event_id TEXT PRIMARY KEY,             -- UUID
    session_id TEXT NOT NULL,              -- Foreign key to sleep_session
    timestamp TIMESTAMP NOT NULL,          -- Event start time
    duration_seconds REAL NOT NULL CHECK(duration_seconds >= 10.0),
    event_type TEXT NOT NULL CHECK(event_type IN ('OBSTRUCTIVE', 'CENTRAL', 'MIXED', 'UNKNOWN')),
    confidence_score REAL NOT NULL CHECK(confidence_score BETWEEN 0.0 AND 1.0),
    severity TEXT NOT NULL CHECK(severity IN ('MILD', 'MODERATE', 'SEVERE')),
    spo2_nadir INTEGER CHECK(spo2_nadir BETWEEN 0 AND 100),
    spo2_baseline INTEGER CHECK(spo2_baseline BETWEEN 0 AND 100),
    spo2_drop INTEGER GENERATED ALWAYS AS (spo2_baseline - spo2_nadir) VIRTUAL,
    respiratory_effort_detected BOOLEAN,
    hdf5_window_index INTEGER,             -- Index in HDF5 file if signal saved
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (session_id) REFERENCES sleep_session(session_id) ON DELETE CASCADE,
    CHECK (spo2_nadir IS NULL OR spo2_baseline IS NULL OR spo2_nadir <= spo2_baseline)
);

CREATE INDEX idx_apnea_event_session_id ON apnea_event(session_id);
CREATE INDEX idx_apnea_event_timestamp ON apnea_event(timestamp);
CREATE INDEX idx_apnea_event_type ON apnea_event(event_type);
CREATE INDEX idx_apnea_event_severity ON apnea_event(severity);
```

#### Fields

- `event_id` (UUID, primary key): Unique identifier for the event
- `session_id` (UUID, foreign key): Reference to parent SleepSession
- `timestamp` (timestamp): When the event started
- `duration_seconds` (float, ≥10.0): Length of breathing pause
- `event_type` (enum): Classification of apnea type
  - `OBSTRUCTIVE`: Airflow stops but breathing effort continues
  - `CENTRAL`: Both airflow and breathing effort stop
  - `MIXED`: Starts as central, transitions to obstructive
  - `UNKNOWN`: Unable to classify with confidence
- `confidence_score` (float, 0.0-1.0): Model's confidence in classification
- `severity` (enum): Clinical severity classification
  - `MILD`: 10-15 seconds
  - `MODERATE`: 15-30 seconds
  - `SEVERE`: >30 seconds
- `spo2_nadir` (integer, 0-100, nullable): Lowest SpO2 during event (percentage)
- `spo2_baseline` (integer, 0-100, nullable): SpO2 before event (percentage)
- `spo2_drop` (computed): `spo2_baseline - spo2_nadir` (desaturation magnitude)
- `respiratory_effort_detected` (boolean): Whether breathing effort was present
- `hdf5_window_index` (integer, nullable): Index in HDF5 file if detailed signal saved
- `created_at` (timestamp): Record creation timestamp

#### Validation Rules

- `duration_seconds` must be ≥10.0 (clinical definition of apnea)
- `confidence_score` must be between 0.0 and 1.0
- Events with `confidence_score < 0.7` should be flagged for manual review
- `spo2_nadir` must be ≤ `spo2_baseline` if both present
- `respiratory_effort_detected = true` indicates obstructive or mixed type
- `respiratory_effort_detected = false` indicates central type
- Severity classification rules:
  - `MILD`: 10 ≤ duration < 15 seconds
  - `MODERATE`: 15 ≤ duration < 30 seconds
  - `SEVERE`: duration ≥ 30 seconds

---

### 3. HypopneaEvent

Represents a detected reduction in airflow (not complete pause).

#### Schema

```sql
CREATE TABLE hypopnea_event (
    event_id TEXT PRIMARY KEY,             -- UUID
    session_id TEXT NOT NULL,              -- Foreign key to sleep_session
    timestamp TIMESTAMP NOT NULL,          -- Event start time
    duration_seconds REAL NOT NULL CHECK(duration_seconds >= 10.0),
    reduction_percentage INTEGER NOT NULL CHECK(reduction_percentage BETWEEN 30 AND 90),
    confidence_score REAL NOT NULL CHECK(confidence_score BETWEEN 0.0 AND 1.0),
    spo2_drop INTEGER CHECK(spo2_drop BETWEEN 0 AND 100),
    arousal_detected BOOLEAN DEFAULT FALSE,
    clinically_significant BOOLEAN GENERATED ALWAYS AS (
        spo2_drop >= 3 OR arousal_detected
    ) VIRTUAL,
    hdf5_window_index INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (session_id) REFERENCES sleep_session(session_id) ON DELETE CASCADE
);

CREATE INDEX idx_hypopnea_event_session_id ON hypopnea_event(session_id);
CREATE INDEX idx_hypopnea_event_timestamp ON hypopnea_event(timestamp);
```

#### Fields

- `event_id` (UUID, primary key): Unique identifier for the event
- `session_id` (UUID, foreign key): Reference to parent SleepSession
- `timestamp` (timestamp): When the event started
- `duration_seconds` (float, ≥10.0): Length of reduced airflow period
- `reduction_percentage` (integer, 30-90): Percentage reduction in airflow
- `confidence_score` (float, 0.0-1.0): Model's confidence in detection
- `spo2_drop` (integer, 0-100, nullable): Oxygen desaturation magnitude (percentage points)
- `arousal_detected` (boolean): Whether EEG arousal was detected (future enhancement)
- `clinically_significant` (computed): `spo2_drop >= 3 OR arousal_detected`
- `hdf5_window_index` (integer, nullable): Index in HDF5 file if signal saved
- `created_at` (timestamp): Record creation timestamp

#### Validation Rules

- `duration_seconds` must be ≥10.0 (clinical definition)
- `reduction_percentage` must be between 30 and 90
  - Below 30%: normal breathing variance
  - 100%: complete apnea (use ApneaEvent instead)
- Hypopnea is clinically significant if `spo2_drop ≥ 3%` or `arousal_detected = true`

---

### 4. SessionSummary

Aggregated statistics for a completed sleep session.

#### Schema

```sql
CREATE TABLE session_summary (
    summary_id TEXT PRIMARY KEY,           -- UUID
    session_id TEXT UNIQUE NOT NULL,       -- Foreign key to sleep_session
    total_apnea_events INTEGER NOT NULL DEFAULT 0,
    total_hypopnea_events INTEGER NOT NULL DEFAULT 0,
    ahi_score REAL NOT NULL,               -- Events per hour
    severity_classification TEXT NOT NULL CHECK(
        severity_classification IN ('NORMAL', 'MILD', 'MODERATE', 'SEVERE')
    ),
    obstructive_count INTEGER NOT NULL DEFAULT 0,
    central_count INTEGER NOT NULL DEFAULT 0,
    mixed_count INTEGER NOT NULL DEFAULT 0,
    unknown_count INTEGER NOT NULL DEFAULT 0,
    mean_event_duration REAL,              -- Average duration (seconds)
    longest_event_duration REAL,           -- Max duration (seconds)
    mean_spo2_nadir REAL CHECK(mean_spo2_nadir BETWEEN 0 AND 100),
    total_spo2_drops_ge3 INTEGER,          -- Count of desaturations ≥3%
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (session_id) REFERENCES sleep_session(session_id) ON DELETE CASCADE,
    CHECK (
        total_apnea_events = obstructive_count + central_count + mixed_count + unknown_count
    )
);

CREATE INDEX idx_session_summary_session_id ON session_summary(session_id);
CREATE INDEX idx_session_summary_ahi_score ON session_summary(ahi_score);
CREATE INDEX idx_session_summary_severity ON session_summary(severity_classification);
```

#### Fields

- `summary_id` (UUID, primary key): Unique identifier for the summary
- `session_id` (UUID, foreign key, unique): Reference to parent SleepSession
- `total_apnea_events` (integer): Count of apnea events
- `total_hypopnea_events` (integer): Count of hypopnea events
- `ahi_score` (float): Apnea-Hypopnea Index (events per hour)
- `severity_classification` (enum): Clinical severity
  - `NORMAL`: AHI < 5
  - `MILD`: 5 ≤ AHI < 15
  - `MODERATE`: 15 ≤ AHI < 30
  - `SEVERE`: AHI ≥ 30
- `obstructive_count` (integer): Count of obstructive apnea events
- `central_count` (integer): Count of central apnea events
- `mixed_count` (integer): Count of mixed apnea events
- `unknown_count` (integer): Count of unclassified apnea events
- `mean_event_duration` (float): Average duration of all events (seconds)
- `longest_event_duration` (float): Duration of longest event (seconds)
- `mean_spo2_nadir` (float, nullable): Average lowest SpO2 across events
- `total_spo2_drops_ge3` (integer): Count of desaturations ≥3% (clinical significance threshold)
- `created_at` (timestamp): Record creation timestamp

#### Validation Rules

- `total_apnea_events` must equal `obstructive_count + central_count + mixed_count + unknown_count`
- `ahi_score` calculation: `(total_apnea_events + total_hypopnea_events) / (total_duration_minutes / 60.0)`
- `severity_classification` must match AHI score ranges:
  - `NORMAL`: AHI < 5
  - `MILD`: 5 ≤ AHI < 15
  - `MODERATE`: 15 ≤ AHI < 30
  - `SEVERE`: AHI ≥ 30
- Can only be created when session state is `COMPLETED`
- Event counts must match actual event count in database

#### Triggers

```sql
-- Auto-calculate AHI score and severity on insert/update
CREATE TRIGGER calculate_ahi_score
AFTER INSERT ON session_summary
FOR EACH ROW
BEGIN
    UPDATE session_summary
    SET ahi_score = (
        SELECT (NEW.total_apnea_events + NEW.total_hypopnea_events) * 60.0 / ss.total_duration_minutes
        FROM sleep_session ss
        WHERE ss.session_id = NEW.session_id
    ),
    severity_classification = (
        CASE
            WHEN ahi_score < 5 THEN 'NORMAL'
            WHEN ahi_score < 15 THEN 'MILD'
            WHEN ahi_score < 30 THEN 'MODERATE'
            ELSE 'SEVERE'
        END
    )
    WHERE summary_id = NEW.summary_id;
END;
```

---

### 5. DeviceProfile

Configuration and specifications for the edge device.

#### Schema

```sql
CREATE TABLE device_profile (
    device_id TEXT PRIMARY KEY,
    device_name TEXT NOT NULL,
    hardware_platform TEXT NOT NULL,       -- e.g., "Apollo4_Plus", "nRF5340"
    sensor_types TEXT NOT NULL,            -- JSON array: ["PPG", "SpO2", "Accelerometer"]
    sampling_rates TEXT NOT NULL,          -- JSON object: {"PPG": 25, "SpO2": 1, "Accelerometer": 25}
    battery_capacity_mah INTEGER NOT NULL CHECK(battery_capacity_mah > 0),
    storage_capacity_mb INTEGER NOT NULL CHECK(storage_capacity_mb > 0),
    firmware_version TEXT NOT NULL,
    model_version TEXT NOT NULL,           -- Currently deployed model
    last_calibration TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_device_profile_hardware ON device_profile(hardware_platform);
```

#### Fields

- `device_id` (string, primary key): Unique device identifier
- `device_name` (string): User-friendly device name
- `hardware_platform` (string): Hardware identifier (e.g., "Apollo4_Plus", "nRF5340")
- `sensor_types` (JSON array): List of sensor types (e.g., ["PPG", "SpO2", "Accelerometer"])
- `sampling_rates` (JSON object): Sampling rate for each sensor (e.g., {"PPG": 25, "SpO2": 1})
- `battery_capacity_mah` (integer): Battery capacity in milliamp-hours
- `storage_capacity_mb` (integer): Available storage in megabytes
- `firmware_version` (string): Current firmware version
- `model_version` (string): Currently deployed ML model version
- `last_calibration` (timestamp, nullable): Last sensor calibration date
- `created_at` (timestamp): Record creation timestamp
- `updated_at` (timestamp): Last update timestamp

#### Validation Rules

- `sampling_rates` must contain entries for all sensors listed in `sensor_types`
- `battery_capacity_mah` must be > 0
- Firmware should alert if `(current_time - last_calibration) > 90 days`

---

### 6. TrendSummary

Aggregated statistics over a time period (weekly/monthly).

#### Schema

```sql
CREATE TABLE trend_summary (
    trend_id TEXT PRIMARY KEY,
    device_id TEXT NOT NULL,               -- Foreign key to device_profile
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    period_type TEXT NOT NULL CHECK(period_type IN ('WEEKLY', 'MONTHLY')),
    total_sessions INTEGER NOT NULL,
    mean_ahi REAL NOT NULL,
    median_ahi REAL NOT NULL,
    std_dev_ahi REAL NOT NULL,
    best_ahi REAL NOT NULL,                -- Lowest (best) AHI
    worst_ahi REAL NOT NULL,               -- Highest (worst) AHI
    best_night_date DATE,
    worst_night_date DATE,
    total_events INTEGER NOT NULL,
    trend_direction TEXT CHECK(trend_direction IN ('IMPROVING', 'STABLE', 'WORSENING')),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (device_id) REFERENCES device_profile(device_id),
    CHECK (period_end > period_start),
    CHECK (best_ahi <= median_ahi),
    CHECK (median_ahi <= worst_ahi)
);

CREATE INDEX idx_trend_summary_device_id ON trend_summary(device_id);
CREATE INDEX idx_trend_summary_period_start ON trend_summary(period_start);
```

#### Fields

- `trend_id` (UUID, primary key): Unique identifier for the trend summary
- `device_id` (string, foreign key): Reference to DeviceProfile
- `period_start` (date): Start date of aggregation period
- `period_end` (date): End date of aggregation period
- `period_type` (enum): Aggregation granularity (`WEEKLY`, `MONTHLY`)
- `total_sessions` (integer): Number of sleep sessions in period
- `mean_ahi` (float): Average AHI across all sessions
- `median_ahi` (float): Median AHI across all sessions
- `std_dev_ahi` (float): Standard deviation of AHI scores
- `best_ahi` (float): Lowest (best) AHI in period
- `worst_ahi` (float): Highest (worst) AHI in period
- `best_night_date` (date): Date of best night (lowest AHI)
- `worst_night_date` (date): Date of worst night (highest AHI)
- `total_events` (integer): Total apnea + hypopnea events across period
- `trend_direction` (enum): Trend indicator (`IMPROVING`, `STABLE`, `WORSENING`)
- `created_at` (timestamp): Record creation timestamp

#### Validation Rules

- `period_end` must be after `period_start`
- `total_sessions` must match count of sessions in date range
- `best_ahi` ≤ `median_ahi` ≤ `worst_ahi`
- Trend direction calculation:
  - `IMPROVING`: Current period AHI < previous period AHI by ≥10%
  - `WORSENING`: Current period AHI > previous period AHI by ≥10%
  - `STABLE`: Change within ±10%

---

### 7. SensorReadings (Optional)

Raw or processed sensor data associated with an event (stored only for low-quality events).

#### Schema

```sql
CREATE TABLE sensor_readings (
    reading_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,                -- Foreign key to apnea_event or hypopnea_event
    event_table TEXT NOT NULL CHECK(event_table IN ('apnea_event', 'hypopnea_event')),
    sensor_type TEXT NOT NULL CHECK(sensor_type IN ('PPG', 'SpO2', 'ACCELEROMETER', 'AUDIO')),
    data_blob BLOB NOT NULL,               -- Compressed time-series data
    sampling_rate INTEGER NOT NULL,
    duration_seconds REAL NOT NULL,
    compression_method TEXT DEFAULT 'gzip',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CHECK (LENGTH(data_blob) <= 1048576)   -- 1MB max
);

CREATE INDEX idx_sensor_readings_event_id ON sensor_readings(event_id);
```

#### Fields

- `reading_id` (UUID, primary key): Unique identifier
- `event_id` (UUID, foreign key): Reference to ApneaEvent or HypopneaEvent
- `event_table` (enum): Which table the event belongs to
- `sensor_type` (enum): `PPG`, `SpO2`, `ACCELEROMETER`, `AUDIO`
- `data_blob` (binary): Compressed time-series data
- `sampling_rate` (integer): Samples per second
- `duration_seconds` (float): Length of captured data
- `compression_method` (string): Compression algorithm (default: "gzip")
- `created_at` (timestamp): Record creation timestamp

#### Validation Rules

- `data_blob` size must not exceed 1MB per event (compression required)
- Only stored for events with `confidence_score < 0.7` or quality issues
- Automatically purged after 30 days (storage management)

---

## Part III: Configuration Files

### 1. Experiment Configuration (YAML)

SleepKit uses YAML configuration files for reproducible experiments.

#### Training Configuration (`configs/apnea/train.yaml`)

```yaml
# Experiment metadata
name: "apnea_detection_multimodal_v1"
version: "1.0"
description: "Multimodal apnea detection with PPG+SpO2+Accelerometer"
project: "sleep_apnea_edge"
random_seed: 42

# Dataset configuration
dataset:
  name: "mesa"
  path: "/storage/datasets/mesa.h5"
  feature_extractor: "multimodal_fusion"
  window_size_seconds: 30
  hop_size_seconds: 15

  # Data splits
  splits:
    train: 0.7
    val: 0.15
    test: 0.15
    stratify_by: "ahi_score"  # Ensure balanced severity distribution

  # Augmentation
  augmentation:
    enabled: true
    methods:
      - type: "time_shift"
        max_shift_seconds: 5
      - type: "signal_scaling"
        scale_range: [0.8, 1.2]
      - type: "gaussian_noise"
        snr_db: 20

# Model architecture
model:
  architecture: "tcn_classifier"
  params:
    num_classes: 3  # [normal, apnea, hypopnea]
    input_shape: [null, 46]  # [sequence_length, n_features]

    # TCN layers
    tcn:
      num_blocks: 4
      filters: [32, 64, 128, 256]
      kernel_size: 3
      dilation_rates: [1, 2, 4, 8]
      dropout: 0.2

    # Classification head
    dense_layers: [128, 64]
    activation: "relu"
    output_activation: "softmax"

# Training hyperparameters
training:
  batch_size: 64
  epochs: 100
  optimizer:
    type: "adam"
    learning_rate: 0.001
    beta_1: 0.9
    beta_2: 0.999

  loss:
    type: "categorical_crossentropy"
    class_weights:  # Handle class imbalance
      normal: 1.0
      apnea: 2.5
      hypopnea: 2.0

  callbacks:
    - type: "early_stopping"
      monitor: "val_loss"
      patience: 10
      restore_best_weights: true

    - type: "reduce_lr_on_plateau"
      monitor: "val_loss"
      factor: 0.5
      patience: 5
      min_lr: 0.00001

    - type: "model_checkpoint"
      filepath: "/storage/models/finetuned/multimodal_lightweight/checkpoint_{epoch:02d}_{val_accuracy:.4f}.keras"
      monitor: "val_accuracy"
      save_best_only: true

# Evaluation metrics
evaluation:
  metrics:
    - "accuracy"
    - "precision"
    - "recall"
    - "f1_score"
    - "auc_roc"

  per_class_metrics: true
  confusion_matrix: true

# Edge deployment settings
deployment:
  target_device: "apollo4_plus"
  quantization:
    enabled: true
    method: "int8"
    representative_dataset_size: 1000

  optimization:
    pruning: false
    clustering: false
```

#### Evaluation Configuration (`configs/apnea/evaluate.yaml`)

```yaml
name: "apnea_detection_evaluation_mesa_test"
version: "1.0"

# Model to evaluate
model:
  path: "/storage/models/finetuned/multimodal_lightweight.keras"
  version: "1.0"

# Test dataset
dataset:
  name: "mesa"
  path: "/storage/datasets/mesa.h5"
  split: "test"
  feature_extractor: "multimodal_fusion"

# Evaluation settings
evaluation:
  batch_size: 128

  metrics:
    - name: "accuracy"
      threshold: 0.9  # Minimum acceptable

    - name: "precision"
      per_class: true
      threshold: 0.85

    - name: "recall"
      per_class: true
      threshold: 0.90  # Prioritize sensitivity

    - name: "f1_score"
      per_class: true
      threshold: 0.87

    - name: "auc_roc"
      per_class: true

  # Clinical metrics
  clinical:
    - name: "ahi_accuracy"
      max_error_events_per_hour: 2.0

    - name: "severe_apnea_fnr"  # False negative rate for severe events
      max_fnr: 0.05  # ≤5%

  # Generate reports
  reports:
    - type: "confusion_matrix"
      normalize: true

    - type: "classification_report"
      output_format: "json"

    - type: "roc_curve"
      per_class: true

    - type: "precision_recall_curve"
      per_class: true

# Output
output:
  directory: "/storage/reports/evaluation_mesa_test_v1"
  save_predictions: true
  save_probabilities: true
```

#### Export Configuration (`configs/apnea/export.yaml`)

```yaml
name: "apnea_detection_tflite_export"
version: "1.0"

# Source model
model:
  path: "/storage/models/finetuned/multimodal_lightweight.keras"
  version: "1.0"

# TFLite export settings
export:
  output_path: "/storage/models/exported/multimodal_lightweight.tflite"

  # Quantization
  quantization:
    enabled: true
    method: "int8"  # int8, float16, or dynamic

    # Representative dataset for calibration
    representative_data:
      path: "/storage/features/multimodal/fusion_features.h5"
      num_samples: 1000
      sample_method: "random"

  # Optimization
  optimizations:
    - "DEFAULT"  # Basic optimizations
    - "OPTIMIZE_FOR_SIZE"
    - "OPTIMIZE_FOR_LATENCY"

  # Input/output settings
  input_shape: [1, null, 46]  # [batch, sequence, features]
  output_shape: [1, 3]  # [batch, num_classes]

# Metadata embedding
metadata:
  model_name: "SleepApnea Detection - Multimodal Lightweight"
  version: "1.0"
  author: "SleepKit Team"
  description: "Edge-optimized apnea detection using PPG, SpO2, and accelerometer"

  # Model performance (from evaluation)
  performance:
    accuracy: 0.923
    precision: 0.891
    recall: 0.945
    f1_score: 0.917
    ahi_accuracy_error: 1.8  # events/hour

  # Input requirements
  input:
    feature_types: ["ppg", "spo2", "accelerometer"]
    sampling_rates: {"ppg": 25, "spo2": 1, "accelerometer": 25}
    window_size_seconds: 30
    hop_size_seconds: 15
    preprocessing: "multimodal_fusion_v1"

  # Output interpretation
  output:
    classes: ["normal", "apnea", "hypopnea"]
    confidence_threshold: 0.7
    post_processing: "smooth_predictions_3_windows"

# Validation
validation:
  run_inference_test: true
  compare_keras_vs_tflite: true
  max_prediction_difference: 0.01  # Max deviation between Keras and TFLite

  test_data:
    path: "/storage/features/multimodal/fusion_features.h5"
    num_samples: 100
```

---

### 2. Model Metadata (JSON)

Each trained model has an accompanying JSON metadata file.

#### Structure (`models/finetuned/multimodal_lightweight.json`)

```json
{
  "model_metadata": {
    "name": "multimodal_lightweight",
    "version": "1.0",
    "created_at": "2025-11-04T10:30:00Z",
    "framework": "keras",
    "framework_version": "3.0.5",
    "author": "SleepKit Team",
    "description": "Lightweight multimodal apnea detector optimized for edge deployment"
  },

  "architecture": {
    "type": "tcn_classifier",
    "num_parameters": 487321,
    "num_layers": 15,
    "input_shape": [null, 46],
    "output_shape": [3],
    "layers": [
      {"type": "InputLayer", "shape": [null, 46]},
      {"type": "TCNBlock", "filters": 32, "kernel_size": 3, "dilation_rate": 1},
      {"type": "TCNBlock", "filters": 64, "kernel_size": 3, "dilation_rate": 2},
      {"type": "TCNBlock", "filters": 128, "kernel_size": 3, "dilation_rate": 4},
      {"type": "TCNBlock", "filters": 256, "kernel_size": 3, "dilation_rate": 8},
      {"type": "GlobalAveragePooling1D"},
      {"type": "Dense", "units": 128, "activation": "relu"},
      {"type": "Dropout", "rate": 0.2},
      {"type": "Dense", "units": 64, "activation": "relu"},
      {"type": "Dense", "units": 3, "activation": "softmax"}
    ]
  },

  "training": {
    "dataset": {
      "name": "mesa",
      "version": "2018",
      "num_subjects": 6814,
      "num_samples": 487392,
      "train_samples": 341374,
      "val_samples": 73108,
      "test_samples": 72910,
      "class_distribution": {
        "normal": 0.72,
        "apnea": 0.18,
        "hypopnea": 0.10
      }
    },

    "hyperparameters": {
      "batch_size": 64,
      "epochs_trained": 87,
      "optimizer": "adam",
      "learning_rate": 0.001,
      "beta_1": 0.9,
      "beta_2": 0.999,
      "loss": "categorical_crossentropy",
      "class_weights": {"normal": 1.0, "apnea": 2.5, "hypopnea": 2.0}
    },

    "training_time_hours": 3.2,
    "hardware": "NVIDIA RTX 4090",
    "random_seed": 42
  },

  "performance": {
    "validation": {
      "accuracy": 0.918,
      "precision": {
        "macro": 0.887,
        "normal": 0.945,
        "apnea": 0.862,
        "hypopnea": 0.854
      },
      "recall": {
        "macro": 0.921,
        "normal": 0.952,
        "apnea": 0.918,
        "hypopnea": 0.893
      },
      "f1_score": {
        "macro": 0.903,
        "normal": 0.948,
        "apnea": 0.889,
        "hypopnea": 0.873
      },
      "auc_roc": {
        "macro": 0.972,
        "normal": 0.985,
        "apnea": 0.967,
        "hypopnea": 0.964
      },
      "confusion_matrix": [
        [52543, 1234, 331],
        [892, 12453, 201],
        [412, 634, 6408]
      ]
    },

    "test": {
      "accuracy": 0.923,
      "precision": {"macro": 0.891},
      "recall": {"macro": 0.945},
      "f1_score": {"macro": 0.917},
      "ahi_accuracy_error_mean": 1.8,
      "ahi_accuracy_error_std": 1.2,
      "severe_apnea_fnr": 0.042
    },

    "clinical_validation": {
      "ahi_correlation": 0.94,
      "sensitivity_apnea": 0.945,
      "specificity_apnea": 0.897,
      "ppv": 0.891,
      "npv": 0.948
    }
  },

  "deployment": {
    "tflite_version": {
      "path": "/storage/models/exported/multimodal_lightweight.tflite",
      "size_bytes": 1847293,
      "quantization": "int8",
      "input_dtype": "int8",
      "output_dtype": "int8"
    },

    "edge_performance": {
      "target_device": "Apollo4 Plus",
      "inference_latency_ms": 87,
      "memory_usage_kb": 1843,
      "power_consumption_mw": 12.3,
      "energy_per_inference_mj": 1.07
    },

    "accuracy_degradation": {
      "keras_vs_tflite": 0.008,
      "meets_threshold": true
    }
  },

  "feature_requirements": {
    "feature_extractor": "multimodal_fusion_v1",
    "input_signals": ["ppg", "spo2", "accelerometer"],
    "sampling_rates": {"ppg": 25, "spo2": 1, "accelerometer": 25},
    "window_size_seconds": 30,
    "hop_size_seconds": 15,
    "preprocessing_pipeline": [
      "bandpass_filter_ppg_0.5_8hz",
      "normalize_spo2_0_100",
      "compute_accel_magnitude",
      "extract_ppg_features_20",
      "extract_spo2_features_8",
      "extract_accel_features_12",
      "compute_cross_correlations_6"
    ],
    "num_features": 46
  },

  "inference": {
    "output_classes": ["normal", "apnea", "hypopnea"],
    "confidence_threshold": 0.7,
    "post_processing": {
      "smoothing": "median_filter_3_windows",
      "min_event_duration_seconds": 10,
      "merge_consecutive_events": true
    }
  },

  "versioning": {
    "model_version": "1.0",
    "feature_extractor_version": "1.0",
    "sleepkit_version": "0.9.0",
    "compatible_firmware": ["v2.0", "v2.1", "v2.2"]
  },

  "changelog": [
    {
      "version": "1.0",
      "date": "2025-11-04",
      "changes": [
        "Initial production release",
        "Trained on MESA dataset (6,814 subjects)",
        "Achieves 92.3% test accuracy",
        "Optimized for Apollo4 Plus deployment"
      ]
    }
  ]
}
```

---

## Part IV: Data Retention and Archival

### Retention Policy

| Data Type | Retention Period | Archival Strategy |
|-----------|------------------|-------------------|
| Sleep Sessions (SQLite) | 90 days | Delete or archive to external storage |
| Apnea/Hypopnea Events (SQLite) | 90 days | Retained with parent session |
| Session Summaries (SQLite) | Indefinite | Small footprint, never deleted |
| Trend Summaries (SQLite) | Indefinite | Aggregated statistics, never deleted |
| Sensor Readings (SQLite) | 30 days | Automatically purged (debugging only) |
| Signal HDF5 files | 7 days | Deleted after feature extraction |
| Feature HDF5 files | Until model retrained | Can be regenerated from signals |
| Trained models | Indefinite | Versioned, old versions archived |
| Configuration files | Indefinite | Version controlled |

### Storage Management Workflow

#### 1. Session Lifecycle

```
[Session Active]
    ↓
[Real-time inference] → Events logged to SQLite
    ↓
[Session Complete] → Generate SessionSummary
    ↓
[Optional: Save HDF5] → Save raw signals if quality_score < 0.7
    ↓ (7 days)
[Delete HDF5] → Keep only SQLite metadata
    ↓ (90 days)
[Archive/Delete Session] → Keep SessionSummary and TrendSummary
```

#### 2. Storage Capacity Triggers

When device approaches storage capacity (>80% full):

1. **Delete sensor readings** older than 30 days
2. **Delete HDF5 signal files** older than 7 days
3. **Archive old sessions** (>90 days):
   - Delete ApneaEvent and HypopneaEvent records
   - Keep SessionSummary
4. **Generate trend summaries** before archiving to preserve insights
5. **Alert user** to sync data to external storage if needed

#### 3. Archival Format

Archived sessions exported as compressed JSON:

```json
{
  "archive_version": "1.0",
  "archive_date": "2025-11-04T10:00:00Z",
  "sessions": [
    {
      "session_id": "uuid",
      "start_time": "2025-08-01T22:30:00Z",
      "summary": {
        "ahi_score": 12.3,
        "severity": "MILD",
        "total_events": 91
      }
    }
  ]
}
```

### Estimated Storage Requirements

| Component | Size per Unit | 90-Day Total |
|-----------|---------------|--------------|
| Sleep Session (metadata) | 1 KB | ~90 KB |
| Apnea Event (avg 20/night) | 0.5 KB | ~900 KB |
| Hypopnea Event (avg 30/night) | 0.4 KB | ~1.1 MB |
| Session Summary | 1 KB | ~90 KB |
| Trend Summaries (12 weeks) | 2 KB | ~24 KB |
| Sensor Readings (if enabled) | 50 MB/night | **Disabled by default** |
| **Total (without raw signals)** | ~100 KB/night | **~9 MB** |

Additional storage for models and features:

- TFLite models: ~2 MB (1-3 variants)
- Feature extractors: ~5 MB
- Configuration files: <1 MB

**Total device storage requirement**: ~15-20 MB for 90 days of operation.

---

## Part V: Data Validation and Integrity

### Validation Layers

#### 1. Schema-Level Validation (SQLite)

- Primary key constraints (UUID uniqueness)
- Foreign key constraints (referential integrity)
- Check constraints (range validation, enum enforcement)
- Unique constraints (one summary per session)
- Not null constraints (required fields)

#### 2. Application-Level Validation

Performed by edge firmware before database insert:

```python
def validate_apnea_event(event):
    """Validate apnea event before insertion."""
    errors = []

    # Duration validation
    if event.duration_seconds < 10.0:
        errors.append("Apnea duration must be >= 10 seconds")

    # Confidence validation
    if not 0.0 <= event.confidence_score <= 1.0:
        errors.append("Confidence score must be in [0.0, 1.0]")

    # SpO2 consistency
    if event.spo2_nadir and event.spo2_baseline:
        if event.spo2_nadir > event.spo2_baseline:
            errors.append("SpO2 nadir cannot exceed baseline")

    # Type-specific validation
    if event.event_type == "OBSTRUCTIVE":
        if not event.respiratory_effort_detected:
            errors.append("Obstructive apnea must have respiratory effort")
    elif event.event_type == "CENTRAL":
        if event.respiratory_effort_detected:
            errors.append("Central apnea cannot have respiratory effort")

    # Severity classification
    severity_map = {
        (10, 15): "MILD",
        (15, 30): "MODERATE",
        (30, float('inf')): "SEVERE"
    }
    expected_severity = next(
        (sev for (low, high), sev in severity_map.items()
         if low <= event.duration_seconds < high),
        None
    )
    if event.severity != expected_severity:
        errors.append(f"Severity mismatch: expected {expected_severity}, got {event.severity}")

    return errors
```

#### 3. Cross-Entity Validation

```python
def validate_session_summary(summary, session):
    """Validate session summary consistency."""
    errors = []

    # AHI calculation
    expected_ahi = (
        (summary.total_apnea_events + summary.total_hypopnea_events)
        * 60.0 / session.total_duration_minutes
    )
    if abs(summary.ahi_score - expected_ahi) > 0.01:
        errors.append(f"AHI mismatch: expected {expected_ahi}, got {summary.ahi_score}")

    # Severity classification
    if summary.ahi_score < 5 and summary.severity_classification != "NORMAL":
        errors.append("Severity should be NORMAL for AHI < 5")
    elif 5 <= summary.ahi_score < 15 and summary.severity_classification != "MILD":
        errors.append("Severity should be MILD for 5 <= AHI < 15")
    elif 15 <= summary.ahi_score < 30 and summary.severity_classification != "MODERATE":
        errors.append("Severity should be MODERATE for 15 <= AHI < 30")
    elif summary.ahi_score >= 30 and summary.severity_classification != "SEVERE":
        errors.append("Severity should be SEVERE for AHI >= 30")

    # Event counts
    apnea_sum = (
        summary.obstructive_count + summary.central_count +
        summary.mixed_count + summary.unknown_count
    )
    if apnea_sum != summary.total_apnea_events:
        errors.append(f"Apnea type counts don't sum to total: {apnea_sum} != {summary.total_apnea_events}")

    return errors
```

#### 4. HDF5 Data Validation

```python
def validate_hdf5_session(h5_file):
    """Validate HDF5 signal file structure."""
    errors = []

    with h5py.File(h5_file, 'r') as f:
        # Check required groups
        required_groups = ['/metadata', '/signals', '/preprocessed']
        for group in required_groups:
            if group not in f:
                errors.append(f"Missing required group: {group}")

        # Validate signal datasets
        for signal_name in f['/signals'].keys():
            signal_group = f['/signals'][signal_name]

            # Check required datasets
            if 'data' not in signal_group:
                errors.append(f"Missing 'data' dataset in {signal_name}")
            if 'timestamps' not in signal_group:
                errors.append(f"Missing 'timestamps' dataset in {signal_name}")

            # Check shape consistency
            if 'data' in signal_group and 'timestamps' in signal_group:
                data_shape = signal_group['data'].shape
                ts_shape = signal_group['timestamps'].shape
                if data_shape[0] != ts_shape[0]:
                    errors.append(
                        f"{signal_name}: data shape {data_shape} doesn't match timestamps {ts_shape}"
                    )

            # Check sampling rate attribute
            if 'sampling_rate' not in signal_group.attrs:
                errors.append(f"Missing 'sampling_rate' attribute in {signal_name}")

        # Validate annotations
        if '/annotations/apnea_events' in f:
            apnea_group = f['/annotations/apnea_events']
            start_len = len(apnea_group['start_indices'])
            end_len = len(apnea_group['end_indices'])
            type_len = len(apnea_group['event_types'])

            if not (start_len == end_len == type_len):
                errors.append(
                    f"Annotation length mismatch: {start_len} starts, {end_len} ends, {type_len} types"
                )

    return errors
```

### Integrity Checks

#### Daily Integrity Scan

Run during device idle time:

```sql
-- Check for orphaned events
SELECT COUNT(*) FROM apnea_event
WHERE session_id NOT IN (SELECT session_id FROM sleep_session);

-- Check for sessions without summaries (should be only active sessions)
SELECT COUNT(*) FROM sleep_session
WHERE state = 'COMPLETED'
  AND session_id NOT IN (SELECT session_id FROM session_summary);

-- Check for invalid AHI calculations
SELECT ss.session_id, ss.ahi_score,
       (ae.count + he.count) * 60.0 / s.total_duration_minutes AS calculated_ahi
FROM session_summary ss
JOIN sleep_session s ON ss.session_id = s.session_id
JOIN (SELECT session_id, COUNT(*) AS count FROM apnea_event GROUP BY session_id) ae
  ON ss.session_id = ae.session_id
JOIN (SELECT session_id, COUNT(*) AS count FROM hypopnea_event GROUP BY session_id) he
  ON ss.session_id = he.session_id
WHERE ABS(ss.ahi_score - calculated_ahi) > 0.1;
```

---

## Part VI: Export Formats

### Medical Reporting Format (JSON)

For healthcare provider review (FR-013):

```json
{
  "export_version": "1.0",
  "export_date": "2025-11-04T08:00:00Z",
  "device": {
    "device_id": "device-12345",
    "device_name": "SleepMonitor Pro",
    "hardware_platform": "Apollo4_Plus",
    "firmware_version": "v2.1",
    "model_version": "multimodal_lightweight_v1.0"
  },

  "session": {
    "session_id": "uuid-session-001",
    "start_time": "2025-11-03T22:30:00Z",
    "end_time": "2025-11-04T06:45:00Z",
    "total_duration_minutes": 495,
    "total_duration_hours": 8.25,
    "quality_score": 0.87
  },

  "summary": {
    "ahi_score": 18.5,
    "severity": "MODERATE",
    "total_apnea_events": 42,
    "total_hypopnea_events": 112,
    "total_events": 154,

    "event_breakdown": {
      "obstructive": 38,
      "central": 4,
      "mixed": 0,
      "unknown": 0
    },

    "event_statistics": {
      "mean_event_duration_seconds": 16.3,
      "longest_event_duration_seconds": 38.2,
      "mean_spo2_nadir": 88.4,
      "total_spo2_drops_ge3": 127
    },

    "hourly_distribution": [
      {"hour": 0, "events": 18, "ahi": 18.0},
      {"hour": 1, "events": 21, "ahi": 21.0},
      {"hour": 2, "events": 19, "ahi": 19.0},
      {"hour": 3, "events": 23, "ahi": 23.0},
      {"hour": 4, "events": 20, "ahi": 20.0},
      {"hour": 5, "events": 17, "ahi": 17.0},
      {"hour": 6, "events": 14, "ahi": 14.0},
      {"hour": 7, "events": 12, "ahi": 12.0},
      {"hour": 8, "events": 10, "ahi": 40.0}
    ]
  },

  "events": [
    {
      "event_number": 1,
      "timestamp": "2025-11-03T23:15:32Z",
      "time_from_sleep_start": "00:45:32",
      "type": "APNEA",
      "classification": "OBSTRUCTIVE",
      "severity": "MODERATE",
      "duration_seconds": 24.5,
      "confidence": 0.94,
      "spo2_baseline": 96,
      "spo2_nadir": 90,
      "spo2_drop": 6,
      "respiratory_effort_detected": true
    },
    {
      "event_number": 2,
      "timestamp": "2025-11-03T23:22:18Z",
      "time_from_sleep_start": "00:52:18",
      "type": "HYPOPNEA",
      "reduction_percentage": 45,
      "duration_seconds": 18.2,
      "confidence": 0.88,
      "spo2_drop": 4,
      "clinically_significant": true
    }
  ],

  "clinical_notes": {
    "predominant_apnea_type": "obstructive",
    "obstructive_percentage": 90.5,
    "central_percentage": 9.5,
    "rem_vs_nrem": "not_available",
    "positional_dependence": "not_available"
  },

  "disclaimer": "This report is generated by an automated sleep monitoring device. It is intended to assist healthcare providers in the diagnosis and management of sleep apnea. This report does not replace a formal polysomnography study or clinical diagnosis by a qualified sleep medicine specialist."
}
```

### CSV Export (Simplified)

For data analysis and spreadsheet import:

```csv
session_id,date,duration_hours,ahi_score,severity,total_apnea,total_hypopnea,obstructive,central,mixed,quality_score
uuid-001,2025-11-03,8.25,18.5,MODERATE,42,112,38,4,0,0.87
uuid-002,2025-11-02,7.80,12.3,MILD,28,68,25,3,0,0.91
uuid-003,2025-11-01,8.10,22.7,MODERATE,51,133,47,4,0,0.85
```

---

## Summary

This comprehensive data model defines:

1. **HDF5 structures** for signals, features, and training datasets (PhysioKit compatible)
2. **SQLite schema** for session metadata, events, summaries, and device profiles
3. **Configuration files** (YAML) for reproducible experiments
4. **Model metadata** (JSON) for deployment and versioning
5. **Retention policies** optimized for edge device storage constraints (90 days)
6. **Validation rules** at schema, application, and cross-entity levels
7. **Export formats** for medical reporting and data analysis

The design balances:
- **Edge constraints**: Minimal storage footprint (~15-20 MB for 90 days)
- **Clinical accuracy**: Comprehensive event tracking and AHI calculations
- **Interoperability**: PhysioKit-compatible HDF5, standard JSON/CSV exports
- **Reproducibility**: Full experiment tracking with configuration files
- **Data integrity**: Multi-layer validation and integrity checks

All data structures align with SleepKit conventions while adding custom extensions for edge deployment, clinical reporting, and longitudinal trend analysis.
