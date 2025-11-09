# Data Export API Contract

## Overview

This document defines the API contract for exporting sleep apnea detection session data. The export API supports multiple formats (JSON, CSV) and provides medical reporting capabilities compliant with FR-013 requirements, including AHI calculation and event summaries.

## Export Formats

### Supported Formats

| Format | Use Case | File Size | Features |
|--------|----------|-----------|----------|
| JSON | Programmatic access, clinical systems | Medium | Full data, nested structures, metadata |
| CSV | Spreadsheet analysis, data science | Small | Tabular events, simple structure |
| Medical Report (JSON) | Clinical documentation | Medium | Standardized medical format, AHI metrics |

## Core Export API

### Session Export Interface

```python
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    MEDICAL_REPORT = "medical_report"

class CompressionType(Enum):
    """Compression options for export files."""
    NONE = "none"
    GZIP = "gzip"
    ZIP = "zip"

@dataclass
class ExportConfig:
    """Configuration for data export."""
    format: ExportFormat
    include_raw_events: bool = True
    include_sensor_data: bool = False
    include_metadata: bool = True
    compression: CompressionType = CompressionType.NONE
    anonymize: bool = False

class SessionExporter:
    """Session data exporter for sleep apnea detection results."""

    def __init__(self, session_id: str, database_path: Path):
        """
        Initialize session exporter.

        Args:
            session_id: Unique session identifier
            database_path: Path to session database

        Raises:
            ValueError: If session_id is invalid
            FileNotFoundError: If database does not exist
        """
        self.session_id = session_id
        self.database_path = database_path

    def export_session(
        self,
        output_path: Path,
        config: ExportConfig
    ) -> Dict[str, Any]:
        """
        Export complete session data to file.

        Args:
            output_path: Output file path
            config: Export configuration

        Returns:
            Dictionary containing export metadata:
            {
                'file_path': str,
                'file_size_bytes': int,
                'checksum_sha256': str,
                'format': str,
                'export_timestamp': str,
                'session_summary': Dict[str, Any]
            }

        Raises:
            ValueError: If session not found or incomplete
            IOError: If file write fails
            RuntimeError: If export processing fails

        Performance:
            - Export time: <5 seconds for typical 8-hour session
            - Memory usage: <100MB during export
        """
        pass

    def export_events_only(
        self,
        output_path: Path,
        config: ExportConfig
    ) -> Dict[str, Any]:
        """
        Export only detected apnea/hypopnea events.

        Args:
            output_path: Output file path
            config: Export configuration

        Returns:
            Export metadata dictionary

        Performance:
            - Export time: <1 second
            - Typical file size: <100KB for 8-hour session
        """
        pass

    def export_sensor_data(
        self,
        output_path: Path,
        modalities: Optional[List[str]] = None,
        downsample_factor: int = 1,
        compression: CompressionType = CompressionType.GZIP
    ) -> Dict[str, Any]:
        """
        Export raw sensor data from session.

        Args:
            output_path: Output file path (HDF5 format)
            modalities: List of modalities to export
                       ['ppg', 'spo2', 'accel', 'audio', 'respiratory']
                       If None, export all available
            downsample_factor: Downsampling factor to reduce file size
            compression: Compression type for HDF5

        Returns:
            Export metadata dictionary

        Raises:
            ValueError: If modalities list contains invalid values
            IOError: If file write fails

        Performance:
            - Export time: <30 seconds for full 8-hour session
            - File size: 100MB-1GB depending on modalities and compression
        """
        pass

    def generate_medical_report(
        self,
        output_path: Path,
        include_physician_notes: bool = False
    ) -> Dict[str, Any]:
        """
        Generate standardized medical report (FR-013).

        Args:
            output_path: Output file path (JSON format)
            include_physician_notes: Include section for physician annotations

        Returns:
            Export metadata dictionary

        Report includes:
            - Patient demographics (anonymized if configured)
            - Session metadata (date, duration)
            - AHI calculation and severity classification
            - Event summary (counts by type)
            - SpO2 statistics (mean, nadir, desaturation events)
            - Sleep position analysis
            - Quality metrics

        Performance:
            - Generation time: <2 seconds
        """
        pass
```

## Export Data Schemas

### 1. JSON Format

```python
# Complete session export schema
JSON_SCHEMA = {
    "session_id": str,
    "device_id": str,
    "firmware_version": str,
    "model_version": str,
    "session_metadata": {
        "start_time": str,  # ISO 8601
        "end_time": str,    # ISO 8601
        "duration_seconds": int,
        "time_zone": str
    },
    "patient_info": {
        "patient_id": Optional[str],  # Anonymized if configured
        "age": Optional[int],
        "sex": Optional[str],
        "bmi": Optional[float]
    },
    "summary_metrics": {
        "ahi": float,
        "ahi_classification": str,  # "normal", "mild", "moderate", "severe"
        "total_apnea_events": int,
        "total_hypopnea_events": int,
        "apnea_breakdown": {
            "obstructive": int,
            "central": int,
            "mixed": int
        },
        "spo2_statistics": {
            "mean_spo2": float,
            "min_spo2": int,
            "desaturation_events": int,
            "odi_3": float,  # Oxygen Desaturation Index (≥3%)
            "odi_4": float   # Oxygen Desaturation Index (≥4%)
        },
        "heart_rate_statistics": {
            "mean_hr_bpm": float,
            "min_hr_bpm": int,
            "max_hr_bpm": int,
            "hrv_rmssd_ms": float
        },
        "sleep_position": {
            "supine_percentage": float,
            "lateral_percentage": float,
            "prone_percentage": float
        }
    },
    "events": List[{
        "event_id": str,
        "timestamp": str,  # ISO 8601
        "event_type": str,  # "apnea" or "hypopnea"
        "apnea_type": Optional[str],  # "obstructive", "central", "mixed"
        "duration_seconds": float,
        "confidence": float,
        "severity": str,  # "mild", "moderate", "severe"
        "spo2_baseline": Optional[int],
        "spo2_nadir": Optional[int],
        "spo2_drop": Optional[int],
        "heart_rate_change": Optional[float],
        "position": Optional[str],  # "supine", "lateral", "prone"
        "arousal_detected": bool
    }],
    "quality_metrics": {
        "overall_quality_score": float,  # 0.0-1.0
        "signal_quality_per_hour": List[float],
        "data_loss_percentage": float,
        "motion_artifact_percentage": float
    },
    "export_metadata": {
        "export_timestamp": str,
        "export_version": str,
        "checksum_sha256": str
    }
}
```

**Example JSON Export**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "device_id": "DEVICE-001",
  "firmware_version": "1.0.0",
  "model_version": "multimodal_v2.1",
  "session_metadata": {
    "start_time": "2025-01-15T22:30:00Z",
    "end_time": "2025-01-16T06:45:00Z",
    "duration_seconds": 29700,
    "time_zone": "America/New_York"
  },
  "summary_metrics": {
    "ahi": 18.5,
    "ahi_classification": "moderate",
    "total_apnea_events": 87,
    "total_hypopnea_events": 66,
    "apnea_breakdown": {
      "obstructive": 72,
      "central": 12,
      "mixed": 3
    },
    "spo2_statistics": {
      "mean_spo2": 94.2,
      "min_spo2": 82,
      "desaturation_events": 95,
      "odi_3": 11.5,
      "odi_4": 8.2
    }
  },
  "events": [
    {
      "event_id": "evt-001",
      "timestamp": "2025-01-15T23:15:30Z",
      "event_type": "apnea",
      "apnea_type": "obstructive",
      "duration_seconds": 15.3,
      "confidence": 0.92,
      "severity": "moderate",
      "spo2_baseline": 95,
      "spo2_nadir": 88,
      "spo2_drop": 7,
      "position": "supine",
      "arousal_detected": true
    }
  ]
}
```

### 2. CSV Format

```python
# CSV schema for events export
CSV_COLUMNS = [
    "event_id",
    "timestamp",
    "event_type",
    "apnea_type",
    "duration_seconds",
    "confidence",
    "severity",
    "spo2_baseline",
    "spo2_nadir",
    "spo2_drop",
    "heart_rate_baseline",
    "heart_rate_change",
    "position",
    "arousal_detected"
]
```

**Example CSV Export**:
```csv
event_id,timestamp,event_type,apnea_type,duration_seconds,confidence,severity,spo2_baseline,spo2_nadir,spo2_drop,position,arousal_detected
evt-001,2025-01-15T23:15:30Z,apnea,obstructive,15.3,0.92,moderate,95,88,7,supine,true
evt-002,2025-01-15T23:22:45Z,hypopnea,,12.7,0.87,mild,94,91,3,supine,false
evt-003,2025-01-15T23:31:20Z,apnea,central,18.1,0.95,severe,93,79,14,lateral,true
```

### 3. Medical Report Format

```python
# Medical report schema (JSON)
MEDICAL_REPORT_SCHEMA = {
    "report_id": str,
    "report_date": str,
    "patient_information": {
        "patient_id": str,
        "demographics": {
            "age": int,
            "sex": str,
            "bmi": float,
            "neck_circumference_cm": Optional[float]
        },
        "medical_history": {
            "hypertension": Optional[bool],
            "diabetes": Optional[bool],
            "cardiovascular_disease": Optional[bool]
        }
    },
    "study_information": {
        "study_date": str,
        "study_duration_hours": float,
        "total_sleep_time_hours": float,
        "sleep_efficiency": float,
        "device_type": str,
        "recording_quality": str  # "excellent", "good", "fair", "poor"
    },
    "respiratory_events": {
        "ahi": float,
        "ahi_classification": str,
        "rdi": float,  # Respiratory Disturbance Index
        "total_apnea_count": int,
        "total_hypopnea_count": int,
        "apnea_index": float,  # Apneas per hour
        "hypopnea_index": float,  # Hypopneas per hour
        "event_distribution": {
            "obstructive_apnea": int,
            "central_apnea": int,
            "mixed_apnea": int,
            "hypopnea": int
        },
        "event_duration": {
            "mean_seconds": float,
            "max_seconds": float,
            "min_seconds": float
        },
        "positional_ahi": {
            "supine_ahi": float,
            "lateral_ahi": float,
            "prone_ahi": Optional[float]
        }
    },
    "oxygen_saturation": {
        "baseline_spo2": float,
        "mean_spo2": float,
        "minimum_spo2": int,
        "odi_3_percent": float,
        "odi_4_percent": float,
        "time_below_90_percent": float,  # Percentage of time
        "desaturation_profile": {
            "mean_desaturation": float,
            "max_desaturation": int
        }
    },
    "cardiovascular_metrics": {
        "mean_heart_rate_bpm": float,
        "heart_rate_variability": {
            "rmssd_ms": float,
            "sdnn_ms": float,
            "lf_hf_ratio": float
        }
    },
    "sleep_architecture": {
        "sleep_position_distribution": {
            "supine_percentage": float,
            "lateral_percentage": float,
            "prone_percentage": float
        },
        "movement_index": float,  # Movements per hour
        "arousal_index": float    # Arousals per hour
    },
    "clinical_interpretation": {
        "severity_assessment": str,
        "predominant_event_type": str,
        "positional_dependency": bool,
        "recommendations": List[str]
    },
    "physician_notes": Optional[str],
    "report_generated_by": str,  # "Automated System"
    "report_version": str
}
```

## AHI Calculation

### AHI Computation Function

```python
def compute_ahi(
    apnea_events: List[Dict[str, Any]],
    hypopnea_events: List[Dict[str, Any]],
    total_sleep_time_hours: float
) -> Dict[str, Any]:
    """
    Compute Apnea-Hypopnea Index and related metrics.

    Args:
        apnea_events: List of detected apnea events
        hypopnea_events: List of detected hypopnea events
        total_sleep_time_hours: Total sleep duration in hours

    Returns:
        Dictionary containing:
        {
            'ahi': float,  # (apnea_count + hypopnea_count) / sleep_hours
            'apnea_index': float,  # apnea_count / sleep_hours
            'hypopnea_index': float,  # hypopnea_count / sleep_hours
            'rdi': float,  # Respiratory Disturbance Index
            'classification': str,  # AHI severity classification
            'event_breakdown': Dict[str, int],
            'positional_ahi': Dict[str, float]
        }

    AHI Classification:
        - Normal: AHI < 5
        - Mild OSA: 5 ≤ AHI < 15
        - Moderate OSA: 15 ≤ AHI < 30
        - Severe OSA: AHI ≥ 30

    Raises:
        ValueError: If total_sleep_time_hours <= 0
        ValueError: If events contain invalid data
    """
    pass

def compute_odi(
    spo2_data: npt.NDArray,
    sample_rate: float,
    threshold: int = 3
) -> Dict[str, Any]:
    """
    Compute Oxygen Desaturation Index.

    Args:
        spo2_data: SpO2 time series data
        sample_rate: Sample rate in Hz
        threshold: Desaturation threshold in percentage points (3 or 4)

    Returns:
        Dictionary containing:
        {
            'odi': float,  # Desaturations per hour
            'desaturation_count': int,
            'mean_desaturation': float,
            'max_desaturation': int,
            'time_below_90': float  # Percentage of time SpO2 < 90%
        }

    ODI Definition:
        Number of times SpO2 drops by ≥threshold% from baseline per hour
    """
    pass
```

## Event Summary Generation

### Summary Functions

```python
def generate_event_summary(
    events: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate statistical summary of detected events.

    Args:
        events: List of apnea/hypopnea events

    Returns:
        Dictionary containing:
        {
            'total_events': int,
            'events_by_type': Dict[str, int],
            'events_by_severity': Dict[str, int],
            'duration_statistics': {
                'mean': float,
                'median': float,
                'std': float,
                'min': float,
                'max': float
            },
            'temporal_distribution': Dict[str, int],  # Events per hour
            'confidence_statistics': {
                'mean': float,
                'high_confidence_count': int  # confidence > 0.8
            }
        }
    """
    pass

def generate_spo2_summary(
    spo2_data: npt.NDArray,
    sample_rate: float
) -> Dict[str, Any]:
    """
    Generate SpO2 statistical summary.

    Args:
        spo2_data: SpO2 time series
        sample_rate: Sample rate in Hz

    Returns:
        Dictionary containing:
        {
            'mean_spo2': float,
            'median_spo2': float,
            'min_spo2': int,
            'std_spo2': float,
            'time_below_90': float,  # Percentage
            'time_below_88': float,  # Percentage
            'time_below_85': float,  # Percentage
            'desaturation_events': int,
            'odi_3': float,
            'odi_4': float
        }
    """
    pass

def generate_position_analysis(
    position_data: List[Dict[str, Any]],
    events: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze sleep position distribution and positional AHI.

    Args:
        position_data: Time-stamped position data
        events: Apnea/hypopnea events with position information

    Returns:
        Dictionary containing:
        {
            'position_distribution': {
                'supine_percentage': float,
                'lateral_percentage': float,
                'prone_percentage': float
            },
            'positional_ahi': {
                'supine': float,
                'lateral': float,
                'prone': float
            },
            'positional_dependency': bool,  # True if supine AHI > 2× lateral AHI
            'predominant_position': str
        }
    """
    pass
```

## Anonymization

### Data Anonymization Functions

```python
def anonymize_session_data(
    session_data: Dict[str, Any],
    anonymization_level: str = 'standard'
) -> Dict[str, Any]:
    """
    Anonymize patient-identifiable information.

    Args:
        session_data: Complete session data dictionary
        anonymization_level: Level of anonymization
                            - 'minimal': Remove direct identifiers only
                            - 'standard': Remove identifiers and quasi-identifiers
                            - 'maximum': Remove all potentially identifying info

    Returns:
        Anonymized session data dictionary

    Anonymization actions:
        - Replace patient_id with hashed ID
        - Remove device_id (replace with device_type)
        - Round timestamps to hour precision
        - Remove location/timezone information (maximum level)
        - Generalize age to ranges (maximum level)

    Raises:
        ValueError: If anonymization_level is invalid
    """
    pass

def generate_anonymized_patient_id(
    original_id: str,
    salt: Optional[str] = None
) -> str:
    """
    Generate anonymized patient identifier.

    Args:
        original_id: Original patient identifier
        salt: Optional salt for hashing (use consistent salt for linking)

    Returns:
        Anonymized patient ID (SHA-256 hash, truncated)

    Format: "ANON-" + first 16 chars of hex digest
    Example: "ANON-a3f5b2c4d8e6f1a9"
    """
    pass
```

## Compression and Packaging

### Compression Functions

```python
def compress_export(
    file_path: Path,
    compression_type: CompressionType = CompressionType.GZIP
) -> Path:
    """
    Compress exported data file.

    Args:
        file_path: Path to uncompressed export file
        compression_type: Type of compression to apply

    Returns:
        Path to compressed file

    Compression ratios (typical):
        - JSON: 70-80% size reduction with gzip
        - CSV: 60-70% size reduction with gzip
        - HDF5 sensor data: 30-50% reduction with gzip

    Performance:
        - Compression time: <2 seconds for 10MB file
    """
    pass

def create_export_package(
    session_id: str,
    export_files: List[Path],
    output_path: Path,
    include_manifest: bool = True
) -> Dict[str, Any]:
    """
    Create packaged export with multiple files.

    Args:
        session_id: Session identifier
        export_files: List of files to include in package
        output_path: Output package path (ZIP archive)
        include_manifest: Whether to include manifest.json

    Returns:
        Package metadata:
        {
            'package_path': str,
            'package_size_bytes': int,
            'files_included': List[str],
            'checksum_sha256': str
        }

    Package structure:
        - manifest.json (metadata, file checksums)
        - session_summary.json
        - events.csv
        - medical_report.json (optional)
        - sensor_data.h5 (optional)
    """
    pass
```

## Validation

### Export Validation Functions

```python
def validate_export(
    file_path: Path,
    expected_format: ExportFormat
) -> Dict[str, Any]:
    """
    Validate exported file integrity and format compliance.

    Args:
        file_path: Path to exported file
        expected_format: Expected file format

    Returns:
        Validation results:
        {
            'valid': bool,
            'format_correct': bool,
            'schema_valid': bool,
            'checksum_verified': bool,
            'issues': List[str],
            'warnings': List[str]
        }

    Validation checks:
        - File exists and readable
        - Format matches expected
        - JSON schema validation (for JSON exports)
        - Required fields present
        - Data types correct
        - Value ranges valid
        - Checksum verification
    """
    pass

def verify_medical_report_compliance(
    report_path: Path
) -> Dict[str, Any]:
    """
    Verify medical report compliance with FR-013 requirements.

    Args:
        report_path: Path to medical report JSON file

    Returns:
        Compliance check results:
        {
            'compliant': bool,
            'required_fields_present': bool,
            'ahi_calculation_valid': bool,
            'format_correct': bool,
            'missing_fields': List[str],
            'compliance_issues': List[str]
        }

    FR-013 Requirements:
        - AHI calculation and classification
        - Event summary by type
        - SpO2 statistics
        - Study metadata
        - Device/model version information
    """
    pass
```

## Usage Examples

### Example 1: Export Complete Session

```python
from export import SessionExporter, ExportConfig, ExportFormat

# Initialize exporter
exporter = SessionExporter(
    session_id="550e8400-e29b-41d4-a716-446655440000",
    database_path=Path("./data/sessions.db")
)

# Configure export
config = ExportConfig(
    format=ExportFormat.JSON,
    include_raw_events=True,
    include_sensor_data=False,
    include_metadata=True,
    anonymize=True
)

# Export session
result = exporter.export_session(
    output_path=Path("./exports/session_export.json"),
    config=config
)

print(f"Export completed: {result['file_path']}")
print(f"File size: {result['file_size_bytes'] / 1024:.1f} KB")
print(f"AHI: {result['session_summary']['ahi']:.1f}")
```

### Example 2: Generate Medical Report

```python
# Generate standardized medical report
report_result = exporter.generate_medical_report(
    output_path=Path("./reports/medical_report.json"),
    include_physician_notes=True
)

print(f"Medical report generated: {report_result['file_path']}")

# Validate compliance
from export import verify_medical_report_compliance

compliance = verify_medical_report_compliance(
    report_path=Path("./reports/medical_report.json")
)

if compliance['compliant']:
    print("Report meets FR-013 requirements")
else:
    print(f"Compliance issues: {compliance['compliance_issues']}")
```

### Example 3: Export Events to CSV

```python
# Export events only in CSV format
config = ExportConfig(
    format=ExportFormat.CSV,
    include_raw_events=True,
    include_sensor_data=False
)

result = exporter.export_events_only(
    output_path=Path("./exports/events.csv"),
    config=config
)

print(f"Exported {result['session_summary']['total_events']} events")
```

### Example 4: Create Complete Export Package

```python
from export import create_export_package

# Export multiple formats
summary_result = exporter.export_session(
    output_path=Path("./temp/summary.json"),
    config=ExportConfig(format=ExportFormat.JSON)
)

events_result = exporter.export_events_only(
    output_path=Path("./temp/events.csv"),
    config=ExportConfig(format=ExportFormat.CSV)
)

report_result = exporter.generate_medical_report(
    output_path=Path("./temp/medical_report.json")
)

# Package all exports
package_result = create_export_package(
    session_id="550e8400-e29b-41d4-a716-446655440000",
    export_files=[
        Path("./temp/summary.json"),
        Path("./temp/events.csv"),
        Path("./temp/medical_report.json")
    ],
    output_path=Path("./exports/session_package.zip"),
    include_manifest=True
)

print(f"Package created: {package_result['package_path']}")
print(f"Package size: {package_result['package_size_bytes'] / 1024:.1f} KB")
```

### Example 5: Export with Sensor Data

```python
# Export raw sensor data (large file)
sensor_result = exporter.export_sensor_data(
    output_path=Path("./exports/sensor_data.h5"),
    modalities=['ppg', 'spo2', 'accel'],  # Exclude audio
    downsample_factor=2,  # Reduce file size by 50%
    compression=CompressionType.GZIP
)

print(f"Sensor data exported: {sensor_result['file_size_bytes'] / (1024**2):.1f} MB")
```

## Performance Characteristics

### Export Performance

| Operation | Typical Time | File Size | Notes |
|-----------|-------------|-----------|-------|
| JSON export (events only) | <1s | 50-200 KB | 8-hour session |
| CSV export (events only) | <0.5s | 20-100 KB | 8-hour session |
| Medical report generation | <2s | 30-50 KB | Includes calculations |
| Complete session export | <5s | 100-500 KB | Without sensor data |
| Sensor data export | <30s | 100MB-1GB | Depends on modalities |
| Package creation | <10s | Varies | Multiple formats |

### Memory Usage

- JSON export: <50MB working memory
- CSV export: <20MB working memory
- Sensor data export: <200MB working memory
- Package creation: <100MB working memory

## Error Handling

```python
class ExportError(RuntimeError):
    """Base class for export errors."""
    pass

class SessionNotFoundError(ExportError):
    """Raised when session ID is not found."""
    pass

class IncompleteSessionError(ExportError):
    """Raised when session is incomplete or invalid."""
    pass

class ExportFormatError(ExportError):
    """Raised when export format is invalid or unsupported."""
    pass

class FileWriteError(ExportError):
    """Raised when file write operation fails."""
    pass
```

## Version Compatibility

- **Python Version**: >=3.9
- **Pandas Version**: >=1.5.0 (for CSV export)
- **H5Py Version**: >=3.8.0 (for HDF5 sensor data)
- **JSON Schema**: Draft 7

## References

- FR-013: Medical Reporting Format Requirements
- AASM Manual for Scoring Sleep: https://aasm.org/clinical-resources/scoring-manual/
- HL7 FHIR ObservationDefinition: https://www.hl7.org/fhir/observationdefinition.html
