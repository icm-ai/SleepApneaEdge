# Data Model: Edge-Based Sleep Apnea Detection

**Feature**: `001-edge-apnea-detection` | **Date**: 2025-11-04 | **Phase**: 1 (Design)

## Overview

This document defines the data structures, relationships, and validation rules for the edge-based sleep apnea detection system. All data is stored locally on the device using SQLite with a 90-day retention policy.

---

## Core Entities

### 1. SleepSession

Represents a single continuous monitoring period (typically 6-10 hours of sleep).

**Fields**:
- `session_id` (UUID, primary key): Unique identifier for the sleep session
- `start_time` (timestamp): When monitoring began
- `end_time` (timestamp, nullable): When monitoring ended (null if ongoing)
- `total_duration_minutes` (integer): Total session length in minutes
- `quality_score` (float, 0.0-1.0): Overall signal quality (0=poor, 1=excellent)
- `device_id` (string): Identifier of the wearable device used
- `firmware_version` (string): Firmware version running during session
- `model_version` (string): ML model version used for inference (e.g., "v1.0")
- `created_at` (timestamp): Record creation timestamp

**Relationships**:
- One-to-many with ApneaEvent
- One-to-many with HypopneaEvent
- One-to-one with SessionSummary

**Validation Rules**:
- `end_time` must be after `start_time` if not null
- `total_duration_minutes` must match `(end_time - start_time)` when session completes
- `quality_score` must be between 0.0 and 1.0
- Sessions with `quality_score < 0.3` should trigger user alert for poor sensor contact

**State Transitions**:
```
[Created] -> [Active] -> [Completed]
              |
              v
          [Aborted] (if user ends session early or device error)
```

---

### 2. ApneaEvent

Represents a detected breathing pause ≥10 seconds.

**Fields**:
- `event_id` (UUID, primary key): Unique identifier for the event
- `session_id` (UUID, foreign key): Reference to parent SleepSession
- `timestamp` (timestamp): When the event started
- `duration_seconds` (float): Length of breathing pause
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
- `respiratory_effort_detected` (boolean): Whether breathing effort was present
- `created_at` (timestamp): Record creation timestamp

**Relationships**:
- Many-to-one with SleepSession
- One-to-one with SensorReadings (optional detailed sensor data)

**Validation Rules**:
- `duration_seconds` must be ≥10.0 (definition of apnea)
- `confidence_score` must be between 0.0 and 1.0
- Events with `confidence_score < 0.7` should be flagged for manual review
- `spo2_nadir` must be ≤ `spo2_baseline` if both present
- `respiratory_effort_detected = true` indicates obstructive or mixed type
- `respiratory_effort_detected = false` indicates central type

**Derived Fields**:
- `spo2_drop`: `spo2_baseline - spo2_nadir` (desaturation magnitude)

---

### 3. HypopneaEvent

Represents a detected reduction in airflow (not complete pause).

**Fields**:
- `event_id` (UUID, primary key): Unique identifier for the event
- `session_id` (UUID, foreign key): Reference to parent SleepSession
- `timestamp` (timestamp): When the event started
- `duration_seconds` (float): Length of reduced airflow period
- `reduction_percentage` (integer, 30-90): Percentage reduction in airflow (clinical definition: ≥30%)
- `confidence_score` (float, 0.0-1.0): Model's confidence in detection
- `spo2_drop` (integer, 0-100, nullable): Oxygen desaturation magnitude (percentage points)
- `arousal_detected` (boolean): Whether EEG arousal was detected (future enhancement)
- `created_at` (timestamp): Record creation timestamp

**Relationships**:
- Many-to-one with SleepSession

**Validation Rules**:
- `duration_seconds` must be ≥10.0 (clinical definition)
- `reduction_percentage` must be between 30 and 90 (below 30% = normal variance, 100% = apnea)
- `confidence_score` must be between 0.0 and 1.0
- Hypopnea classified as clinically significant if `spo2_drop ≥ 3` or `arousal_detected = true`

---

### 4. SessionSummary

Aggregated statistics for a completed sleep session.

**Fields**:
- `summary_id` (UUID, primary key): Unique identifier for the summary
- `session_id` (UUID, foreign key, unique): Reference to parent SleepSession
- `total_apnea_events` (integer): Count of apnea events
- `total_hypopnea_events` (integer): Count of hypopnea events
- `ahi_score` (float): Apnea-Hypopnea Index (events per hour)
- `severity_classification` (enum): Clinical severity
  - `NORMAL`: AHI < 5
  - `MILD`: AHI 5-15
  - `MODERATE`: AHI 15-30
  - `SEVERE`: AHI > 30
- `obstructive_count` (integer): Count of obstructive apnea events
- `central_count` (integer): Count of central apnea events
- `mixed_count` (integer): Count of mixed apnea events
- `mean_event_duration` (float): Average duration of all events (seconds)
- `longest_event_duration` (float): Duration of longest event (seconds)
- `mean_spo2_nadir` (float, nullable): Average lowest SpO2 across events
- `total_spo2_drops_gt3` (integer): Count of desaturations ≥3% (clinical significance threshold)
- `created_at` (timestamp): Record creation timestamp

**Relationships**:
- One-to-one with SleepSession

**Validation Rules**:
- `total_apnea_events + total_hypopnea_events` must match actual event count in database
- `ahi_score` calculation: `(total_apnea_events + total_hypopnea_events) / (total_duration_minutes / 60.0)`
- `severity_classification` must match AHI score ranges
- Can only be created when session state is `Completed`

**Derived Calculations**:
- **AHI Score**: `(apnea_count + hypopnea_count) / sleep_hours`
- **Obstructive Percentage**: `(obstructive_count / total_apnea_events) * 100`
- **Central Percentage**: `(central_count / total_apnea_events) * 100`

---

### 5. DeviceProfile

Configuration and specifications for the edge device.

**Fields**:
- `device_id` (string, primary key): Unique device identifier
- `device_name` (string): User-friendly device name
- `hardware_platform` (string): Hardware identifier (e.g., "nRF5340", "MAX32664")
- `sensor_types` (JSON array): List of sensor types (e.g., ["PPG", "SpO2", "Accelerometer"])
- `sampling_rates` (JSON object): Sampling rate for each sensor (e.g., {"PPG": 25, "SpO2": 1})
- `battery_capacity_mah` (integer): Battery capacity in milliamp-hours
- `storage_capacity_mb` (integer): Available storage in megabytes
- `firmware_version` (string): Current firmware version
- `model_version` (string): Currently deployed ML model version
- `last_calibration` (timestamp, nullable): Last sensor calibration date
- `created_at` (timestamp): Record creation timestamp
- `updated_at` (timestamp): Last update timestamp

**Validation Rules**:
- `sampling_rates` must contain entries for all sensors listed in `sensor_types`
- `battery_capacity_mah` must be > 0
- Firmware should alert if `(current_time - last_calibration) > 90 days`

---

### 6. TrendSummary

Aggregated statistics over a time period (weekly/monthly).

**Fields**:
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

**Relationships**:
- Many-to-one with DeviceProfile
- References multiple SleepSession records (via date range)

**Validation Rules**:
- `period_end` must be after `period_start`
- `total_sessions` must match count of sessions in date range
- `best_ahi` ≤ `median_ahi` ≤ `worst_ahi`
- Trend direction calculation:
  - `IMPROVING`: Current period AHI < previous period AHI by ≥10%
  - `WORSENING`: Current period AHI > previous period AHI by ≥10%
  - `STABLE`: Change within ±10%

---

## Supporting Entities

### 7. SensorReadings (Optional, for detailed analysis)

Raw or processed sensor data associated with an event (stored only if `quality_score < 0.7` for debugging).

**Fields**:
- `reading_id` (UUID, primary key)
- `event_id` (UUID, foreign key): Reference to ApneaEvent or HypopneaEvent
- `sensor_type` (enum): `PPG`, `SpO2`, `ACCELEROMETER`
- `data_blob` (binary): Compressed time-series data
- `sampling_rate` (integer): Samples per second
- `duration_seconds` (float): Length of captured data
- `created_at` (timestamp)

**Validation Rules**:
- `data_blob` size must not exceed 1MB per event (compression required)
- Only stored for events with low confidence or quality issues
- Automatically purged after 30 days (storage management)

---

## Storage Management

### Retention Policy
- **Sleep Sessions**: 90 nights (FR-005)
- **Events**: Retained with parent session (90 nights)
- **Summaries**: Indefinite (small data footprint)
- **Sensor Readings**: 30 days maximum (debugging only)
- **Trend Summaries**: Indefinite (aggregated statistics)

### Archival Strategy
When device approaches storage capacity:
1. Delete sensor readings older than 30 days
2. Archive sessions older than 90 days (keep summaries, delete raw events)
3. Generate trend summaries before archiving to preserve longitudinal insights

### Estimated Storage Requirements
- Single session: ~100 KB (metadata + events)
- 90 nights: ~9 MB
- Sensor readings (if enabled): ~50 MB/night → disabled by default
- Total footprint: 10-15 MB for 90 days (within device constraints)

---

## Export Format

For medical reporting (FR-013), sessions can be exported in JSON format:

```json
{
  "export_version": "1.0",
  "device_id": "device-uuid",
  "session": {
    "session_id": "session-uuid",
    "start_time": "2025-11-03T22:30:00Z",
    "end_time": "2025-11-04T06:45:00Z",
    "total_duration_minutes": 495,
    "quality_score": 0.87
  },
  "summary": {
    "ahi_score": 18.5,
    "severity": "MODERATE",
    "total_apnea_events": 42,
    "total_hypopnea_events": 112,
    "event_breakdown": {
      "obstructive": 38,
      "central": 4,
      "mixed": 0
    }
  },
  "events": [
    {
      "timestamp": "2025-11-03T23:15:32Z",
      "type": "APNEA",
      "classification": "OBSTRUCTIVE",
      "duration_seconds": 24.5,
      "confidence": 0.94,
      "spo2_drop": 6
    }
  ]
}
```

---

## Indexing Strategy

For optimal query performance on SQLite:

- **Primary Keys**: UUID for all entities
- **Indexes**:
  - `sleep_session(start_time)` - Time-range queries
  - `apnea_event(session_id, timestamp)` - Event retrieval
  - `hypopnea_event(session_id, timestamp)` - Event retrieval
  - `session_summary(ahi_score)` - Trend analysis
  - `trend_summary(device_id, period_start)` - Historical trends

---

## Validation Summary

All data entities enforce:
- Type safety (enforced by SQLite schema + application layer validation)
- Range constraints (e.g., `quality_score` between 0.0-1.0)
- Referential integrity (foreign key constraints)
- Timestamp consistency (end times after start times)
- Clinical accuracy (AHI calculations match event counts)

Validation errors trigger device alerts and prevent session summary generation until resolved.
