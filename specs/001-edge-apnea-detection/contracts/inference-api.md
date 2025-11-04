# Inference API Contract

**Feature**: `001-edge-apnea-detection` | **Version**: 1.0 | **Date**: 2025-11-04

## Overview

This document defines the internal API contracts for the edge-based sleep apnea detection inference pipeline. All interfaces operate locally on the device (no network communication required per FR-003).

---

## 1. Preprocessing Pipeline

### `preprocess_sensor_data()`

Filters and extracts features from raw sensor readings.

**Input**:
```python
{
  "ppg_signal": np.ndarray,        # PPG waveform, shape: (n_samples,), dtype: float32
  "spo2_signal": np.ndarray,       # SpO2 measurements, shape: (n_samples,), dtype: float32
  "accelerometer": {
    "x": np.ndarray,               # X-axis, shape: (n_samples,), dtype: float32
    "y": np.ndarray,               # Y-axis
    "z": np.ndarray                # Z-axis
  },
  "sampling_rate": int,            # Samples per second (e.g., 25 Hz)
  "window_size_seconds": int       # Analysis window duration (e.g., 30)
}
```

**Output**:
```python
{
  "features": np.ndarray,          # Feature vector, shape: (n_features,), dtype: float32
  "quality_score": float,          # Signal quality, range: [0.0, 1.0]
  "metadata": {
    "timestamp": str,              # ISO 8601 timestamp
    "window_start": int,           # Sample index of window start
    "window_end": int,             # Sample index of window end
    "sensors_active": List[str]    # List of sensors with valid data
  }
}
```

**Error Conditions**:
- `ValueError`: Invalid sampling rate (must be > 0)
- `ValueError`: Signal length < window_size_seconds * sampling_rate
- `ValueError`: Array shape mismatch between signals
- `QualityError`: `quality_score < 0.3` (insufficient signal quality for inference)

**Constraints**:
- Processing latency: <20ms per window on Nordic nRF5340
- Memory usage: <512 KB for intermediate buffers

---

## 2. Apnea Event Detection

### `detect_apnea_events()`

Runs ML inference to detect apnea events from preprocessed features.

**Input**:
```python
{
  "features": np.ndarray,          # Feature vector from preprocessing, shape: (n_features,)
  "model": Model,                  # TFLite Micro model handle
  "threshold": float               # Detection threshold, default: 0.7
}
```

**Output**:
```python
{
  "event_detected": bool,          # True if apnea event detected
  "confidence": float,             # Model confidence, range: [0.0, 1.0]
  "duration_seconds": float,       # Estimated event duration (if detected)
  "severity": str,                 # "MILD" | "MODERATE" | "SEVERE" | null
  "inference_time_ms": float       # Model inference latency
}
```

**Error Conditions**:
- `ModelError`: Model inference failed (invalid input shape, memory allocation failure)
- `ValueError`: `threshold` not in range [0.0, 1.0]
- `TimeoutError`: Inference exceeded 100ms latency constraint (SC-ML-004)

**Constraints**:
- Inference latency: <100ms per window (SC-ML-004)
- Model size: <50MB (SC-ML-006)
- Accuracy: ≥95% sensitivity, ≤10% false positive rate (SC-002, SC-003)

---

## 3. Apnea Event Classification

### `classify_apnea_type()`

Classifies detected apnea event as obstructive, central, or mixed.

**Input**:
```python
{
  "features": np.ndarray,          # Feature vector from preprocessing
  "respiratory_effort": np.ndarray, # Accelerometer-derived effort signal
  "model": Model,                  # TFLite Micro classification model
  "threshold": float               # Classification threshold, default: 0.7
}
```

**Output**:
```python
{
  "event_type": str,               # "OBSTRUCTIVE" | "CENTRAL" | "MIXED" | "UNKNOWN"
  "confidence": float,             # Model confidence, range: [0.0, 1.0]
  "respiratory_effort_detected": bool, # True if breathing effort present
  "inference_time_ms": float       # Model inference latency
}
```

**Error Conditions**:
- `ModelError`: Classification model inference failed
- `ValueError`: `threshold` not in range [0.0, 1.0]
- `TimeoutError`: Inference exceeded 100ms latency constraint

**Constraints**:
- Inference latency: <100ms per event
- Accuracy: ≥90% classification accuracy (SC-ML-007)

---

## 4. Session Management

### `start_session()`

Initializes a new sleep monitoring session.

**Input**:
```python
{
  "device_id": str,                # Unique device identifier
  "firmware_version": str,         # Current firmware version
  "model_version": str             # ML model version
}
```

**Output**:
```python
{
  "session_id": str,               # UUID for the session
  "start_time": str,               # ISO 8601 timestamp
  "status": str                    # "ACTIVE"
}
```

**Error Conditions**:
- `StorageError`: Insufficient storage space for new session
- `StateError`: Previous session not properly closed

---

### `end_session()`

Completes a sleep monitoring session and generates summary.

**Input**:
```python
{
  "session_id": str                # UUID of active session
}
```

**Output**:
```python
{
  "session_id": str,
  "end_time": str,                 # ISO 8601 timestamp
  "total_duration_minutes": int,
  "summary": {
    "ahi_score": float,
    "total_events": int,
    "severity_classification": str # "NORMAL" | "MILD" | "MODERATE" | "SEVERE"
  },
  "status": str                    # "COMPLETED"
}
```

**Error Conditions**:
- `NotFoundError`: Session ID not found
- `StateError`: Session already completed
- `ValidationError`: Session duration < 2 hours (insufficient for AHI calculation)

---

## 5. Event Logging

### `log_apnea_event()`

Stores a detected apnea event to local database.

**Input**:
```python
{
  "session_id": str,
  "timestamp": str,                # ISO 8601 timestamp
  "duration_seconds": float,
  "event_type": str,               # "OBSTRUCTIVE" | "CENTRAL" | "MIXED" | "UNKNOWN"
  "confidence_score": float,
  "severity": str,                 # "MILD" | "MODERATE" | "SEVERE"
  "spo2_nadir": int,               # Optional, 0-100
  "spo2_baseline": int,            # Optional, 0-100
  "respiratory_effort_detected": bool
}
```

**Output**:
```python
{
  "event_id": str,                 # UUID for the logged event
  "status": str                    # "LOGGED"
}
```

**Error Conditions**:
- `StorageError`: Database write failed (disk full, corruption)
- `ValidationError`: Invalid field values (e.g., `duration_seconds < 10.0`)
- `NotFoundError`: Session ID not found

**Constraints**:
- Write latency: <10ms to avoid blocking real-time inference

---

### `log_hypopnea_event()`

Stores a detected hypopnea event to local database.

**Input**:
```python
{
  "session_id": str,
  "timestamp": str,
  "duration_seconds": float,
  "reduction_percentage": int,     # 30-90
  "confidence_score": float,
  "spo2_drop": int,                # Optional, 0-100
  "arousal_detected": bool         # Default: false
}
```

**Output**:
```python
{
  "event_id": str,
  "status": str                    # "LOGGED"
}
```

**Error Conditions**:
- Same as `log_apnea_event()`
- `ValidationError`: `reduction_percentage` not in range [30, 90]

---

## 6. Data Export

### `export_session()`

Exports sleep session data in medical reporting format (FR-013).

**Input**:
```python
{
  "session_id": str,
  "format": str,                   # "JSON" | "CSV" | "PDF" (future)
  "include_raw_events": bool,      # Default: true
  "include_sensor_data": bool      # Default: false (large files)
}
```

**Output**:
```python
{
  "export_file": bytes,            # Exported data blob
  "file_size_bytes": int,
  "checksum": str,                 # SHA-256 hash for integrity
  "format": str                    # Confirmed format
}
```

**Error Conditions**:
- `NotFoundError`: Session ID not found
- `StateError`: Session not yet completed
- `StorageError`: Insufficient space for temporary export file

**Constraints**:
- Export latency: <5 seconds for typical session (SC-004)

---

## 7. Quality Assessment

### `assess_signal_quality()`

Evaluates real-time signal quality for user feedback.

**Input**:
```python
{
  "ppg_signal": np.ndarray,
  "spo2_signal": np.ndarray,
  "accelerometer": dict,           # {x, y, z} arrays
  "sampling_rate": int
}
```

**Output**:
```python
{
  "quality_score": float,          # Overall quality, range: [0.0, 1.0]
  "issues": List[str],             # List of detected issues
  "recommendations": List[str]     # User actions to improve quality
}
```

**Example Issues**:
- `"PPG_SATURATION"`: PPG signal saturated (sensor too tight)
- `"LOW_SIGNAL"`: Weak signal (poor sensor contact)
- `"EXCESSIVE_MOTION"`: High motion artifacts
- `"SPO2_UNRELIABLE"`: SpO2 readings inconsistent

**Example Recommendations**:
- `"Adjust device position"`
- `"Ensure sensor is clean"`
- `"Reduce movement"`

**Error Conditions**:
- `ValueError`: Invalid input signals

**Constraints**:
- Assessment latency: <50ms (real-time feedback)

---

## 8. Model Management

### `load_model()`

Loads a TFLite model from device storage into memory.

**Input**:
```python
{
  "model_path": str,               # Path to .tflite file
  "model_type": str                # "DETECTOR" | "CLASSIFIER"
}
```

**Output**:
```python
{
  "model": Model,                  # Opaque model handle
  "version": str,                  # Model version from metadata
  "input_shape": tuple,            # Expected input shape
  "output_shape": tuple,           # Output shape
  "memory_usage_kb": int           # Model memory footprint
}
```

**Error Conditions**:
- `FileNotFoundError`: Model file not found
- `ValidationError`: Model file corrupted or incompatible format
- `MemoryError`: Insufficient memory to load model (exceeds 50MB constraint)

---

### `validate_model()`

Validates model accuracy on test dataset.

**Input**:
```python
{
  "model": Model,
  "test_dataset": Dataset,         # Labeled test data
  "metrics": List[str]             # ["accuracy", "precision", "recall", "f1"]
}
```

**Output**:
```python
{
  "accuracy": float,               # Overall accuracy
  "precision": float,              # Per-class average
  "recall": float,
  "f1": float,
  "per_class_metrics": dict,       # Breakdown by class
  "passes_threshold": bool         # True if accuracy ≥ 0.90 (SC-ML-001)
}
```

**Error Conditions**:
- `ValueError`: Test dataset empty or invalid format
- `ValidationError`: Model fails ≥90% accuracy threshold

---

## Error Handling Standards

All API functions follow consistent error handling:

1. **Return error codes/exceptions** (no silent failures)
2. **Log errors** with timestamp, function name, and context
3. **Preserve device state** (roll back partial transactions on failure)
4. **User notifications** for critical errors (storage full, sensor failure)

**Error Severity Levels**:
- `CRITICAL`: Device cannot continue monitoring (e.g., storage full) → alert user immediately
- `ERROR`: Inference failed for current window → skip window, continue monitoring
- `WARNING`: Quality issue detected → notify user, continue monitoring
- `INFO`: Normal operational events → log only

---

## Performance Guarantees

All APIs must meet these constraints on Nordic nRF5340 hardware:

| Function | Max Latency | Max Memory | Notes |
|----------|-------------|------------|-------|
| `preprocess_sensor_data()` | 20ms | 512 KB | Per 30-second window |
| `detect_apnea_events()` | 100ms | 10 MB | Includes model inference |
| `classify_apnea_type()` | 100ms | 10 MB | Includes model inference |
| `log_apnea_event()` | 10ms | 1 KB | Non-blocking writes |
| `assess_signal_quality()` | 50ms | 256 KB | Real-time feedback |
| `export_session()` | 5s | 5 MB | One-time per session |

**Power Budget**:
- Active inference: <50 mA average current
- Idle monitoring: <5 mA average current
- Total session power: <400 mAh for 8-hour night (enables 2-3 night battery life on 1000 mAh battery)

---

## Versioning

API version follows semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes to input/output contracts
- **MINOR**: New optional parameters or fields
- **PATCH**: Bug fixes, no contract changes

Current version: **1.0.0**

Future compatibility: Models must include API version compatibility metadata to prevent runtime errors.
