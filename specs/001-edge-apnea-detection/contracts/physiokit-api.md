# PhysioKit Signal Processing API Contract

## Overview

This document defines the API contract for PhysioKit signal processing operations used in sleep apnea detection. PhysioKit provides preprocessing, feature extraction, and quality assessment for physiological signals including PPG, SpO2, accelerometer, audio, and respiratory data.

## Signal Types

### Supported Signal Modalities

| Signal Type | Sample Rate | Data Type | Channels | Use Case |
|-------------|-------------|-----------|----------|----------|
| PPG | 25-100 Hz | float32 | 1-3 | Heart rate, HRV analysis |
| SpO2 | 1-25 Hz | float32 | 1 | Oxygen saturation |
| Accelerometer | 25-100 Hz | float32 | 3 | Motion, sleep position |
| Audio | 8000-16000 Hz | float32 | 1-2 | Respiratory events |
| Respiratory | 10-50 Hz | float32 | 1 | Breathing patterns |

## Core API

### Signal Class

```python
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from enum import Enum

class SignalType(Enum):
    """Enumeration of supported signal types."""
    PPG = "ppg"
    SPO2 = "spo2"
    ACCELEROMETER = "accel"
    AUDIO = "audio"
    RESPIRATORY = "respiratory"

@dataclass
class SignalMetadata:
    """Metadata for physiological signals."""
    signal_type: SignalType
    sample_rate: float
    duration_seconds: float
    num_channels: int
    recording_start_time: Optional[str] = None
    device_id: Optional[str] = None
    quality_score: Optional[float] = None

class Signal:
    """Container for physiological signal data."""

    def __init__(
        self,
        data: npt.NDArray[np.float32],
        sample_rate: float,
        signal_type: SignalType,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize signal container.

        Args:
            data: Signal samples (shape: [num_samples, num_channels] or [num_samples])
            sample_rate: Sampling frequency in Hz
            signal_type: Type of physiological signal
            metadata: Optional metadata dictionary

        Raises:
            ValueError: If data shape is invalid or sample_rate <= 0
            TypeError: If data is not numpy array
        """
        self.data = data
        self.sample_rate = sample_rate
        self.signal_type = signal_type
        self.metadata = metadata or {}

    @property
    def duration(self) -> float:
        """Return signal duration in seconds."""
        return len(self.data) / self.sample_rate

    @property
    def num_samples(self) -> int:
        """Return number of samples."""
        return len(self.data)

    @property
    def num_channels(self) -> int:
        """Return number of channels."""
        return self.data.shape[1] if self.data.ndim > 1 else 1
```

## Preprocessing API

### Filtering Functions

```python
def bandpass_filter(
    signal: Signal,
    low_freq: float,
    high_freq: float,
    order: int = 4,
    filter_type: str = 'butterworth'
) -> Signal:
    """
    Apply bandpass filter to signal.

    Args:
        signal: Input signal
        low_freq: Lower cutoff frequency in Hz
        high_freq: Upper cutoff frequency in Hz
        order: Filter order (default: 4)
        filter_type: Filter type ('butterworth', 'chebyshev', 'bessel')

    Returns:
        Filtered signal with same shape as input

    Raises:
        ValueError: If frequency bounds are invalid
        ValueError: If low_freq >= high_freq
        ValueError: If frequencies exceed Nyquist limit

    Performance:
        - Time complexity: O(n × order)
        - Memory usage: O(n) where n = num_samples
        - Processing time: <10ms for 1 minute of data @ 100Hz
    """
    pass

def notch_filter(
    signal: Signal,
    freq: float,
    quality_factor: float = 30.0
) -> Signal:
    """
    Apply notch filter to remove specific frequency component.

    Args:
        signal: Input signal
        freq: Frequency to remove in Hz (e.g., 50/60 Hz powerline)
        quality_factor: Q factor (bandwidth = freq/Q)

    Returns:
        Filtered signal

    Raises:
        ValueError: If freq exceeds Nyquist limit

    Performance:
        - Processing time: <5ms for 1 minute of data
    """
    pass

def normalize(
    signal: Signal,
    method: str = 'zscore',
    axis: Optional[int] = None
) -> Signal:
    """
    Normalize signal values.

    Args:
        signal: Input signal
        method: Normalization method
                - 'zscore': (x - mean) / std
                - 'minmax': (x - min) / (max - min)
                - 'robust': (x - median) / IQR
        axis: Axis for normalization (None = global, 0 = per-channel)

    Returns:
        Normalized signal

    Raises:
        ValueError: If method is unsupported

    Performance:
        - Processing time: <2ms for 1 minute of data
    """
    pass
```

### Resampling Functions

```python
def resample(
    signal: Signal,
    target_rate: float,
    method: str = 'polyphase'
) -> Signal:
    """
    Resample signal to target sample rate.

    Args:
        signal: Input signal
        target_rate: Target sampling rate in Hz
        method: Resampling method
                - 'polyphase': Polyphase filtering (high quality)
                - 'linear': Linear interpolation (fast)
                - 'cubic': Cubic spline interpolation

    Returns:
        Resampled signal at target_rate

    Raises:
        ValueError: If target_rate <= 0
        ValueError: If method is unsupported

    Performance:
        - Time complexity: O(n × m) where m = resampling ratio
        - Processing time: <20ms for 1 minute of data
        - Preserves signal characteristics within 1% error
    """
    pass

def downsample(
    signal: Signal,
    factor: int,
    antialias: bool = True
) -> Signal:
    """
    Downsample signal by integer factor.

    Args:
        signal: Input signal
        factor: Downsampling factor (e.g., 4 = reduce by 4x)
        antialias: Apply anti-aliasing filter before downsampling

    Returns:
        Downsampled signal

    Raises:
        ValueError: If factor < 1

    Performance:
        - Processing time: <10ms for 1 minute of data
    """
    pass
```

### Artifact Removal

```python
def remove_motion_artifacts(
    signal: Signal,
    accelerometer_data: Optional[Signal] = None,
    threshold: float = 0.5
) -> Tuple[Signal, npt.NDArray[np.bool_]]:
    """
    Remove motion artifacts from signal.

    Args:
        signal: Input signal (PPG, SpO2, etc.)
        accelerometer_data: Corresponding accelerometer data (optional)
        threshold: Motion detection threshold (g-force)

    Returns:
        Tuple of (cleaned_signal, artifact_mask) where artifact_mask
        indicates detected artifact regions (True = artifact)

    Raises:
        ValueError: If signals have mismatched lengths

    Performance:
        - Processing time: <50ms for 1 minute of data
        - Detection accuracy: >90% for typical motion artifacts
    """
    pass

def remove_baseline_wander(
    signal: Signal,
    window_size: float = 5.0
) -> Signal:
    """
    Remove baseline wander from signal.

    Args:
        signal: Input signal
        window_size: Moving average window size in seconds

    Returns:
        Signal with baseline removed

    Performance:
        - Processing time: <15ms for 1 minute of data
    """
    pass
```

## Feature Extraction API

### Time Domain Features

```python
def extract_time_domain_features(
    signal: Signal,
    features: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Extract time-domain features from signal.

    Args:
        signal: Input signal
        features: List of feature names to extract
                  Available: ['mean', 'std', 'min', 'max', 'rms',
                             'peak_to_peak', 'zero_crossing_rate',
                             'kurtosis', 'skewness']
                  If None, extract all features

    Returns:
        Dictionary mapping feature names to values

    Raises:
        ValueError: If signal is empty or contains invalid values

    Performance:
        - Processing time: <5ms for 1 minute of data
        - Memory usage: O(1) constant

    Example:
        >>> features = extract_time_domain_features(ppg_signal)
        >>> print(features)
        {
            'mean': 512.3,
            'std': 45.2,
            'rms': 515.7,
            'peak_to_peak': 250.1,
            'zero_crossing_rate': 1.2,
            'kurtosis': 2.8,
            'skewness': -0.3
        }
    """
    pass
```

### Frequency Domain Features

```python
def extract_frequency_features(
    signal: Signal,
    freq_bands: Optional[Dict[str, Tuple[float, float]]] = None,
    window_size: float = 30.0,
    overlap: float = 0.5
) -> Dict[str, Any]:
    """
    Extract frequency-domain features using FFT/Welch method.

    Args:
        signal: Input signal
        freq_bands: Dictionary of frequency bands
                    Default for PPG: {
                        'vlf': (0.0, 0.04),    # Very low frequency
                        'lf': (0.04, 0.15),    # Low frequency
                        'hf': (0.15, 0.4)      # High frequency
                    }
        window_size: Window size in seconds for Welch method
        overlap: Window overlap ratio (0.0-1.0)

    Returns:
        Dictionary containing:
        {
            'power_spectrum': np.ndarray,  # Full power spectrum
            'frequencies': np.ndarray,      # Frequency bins
            'band_power': Dict[str, float], # Power in each band
            'peak_frequency': float,        # Dominant frequency
            'spectral_entropy': float,      # Signal complexity
            'band_power_ratio': Dict[str, float]  # Normalized ratios
        }

    Raises:
        ValueError: If window_size > signal duration
        ValueError: If overlap not in [0, 1]

    Performance:
        - Processing time: <30ms for 1 minute of data
        - Frequency resolution: sample_rate / window_samples

    Example:
        >>> features = extract_frequency_features(ppg_signal)
        >>> print(features['band_power'])
        {'vlf': 123.4, 'lf': 456.7, 'hf': 234.5}
    """
    pass

def compute_spectrogram(
    signal: Signal,
    window_size: float = 2.0,
    overlap: float = 0.75,
    freq_range: Optional[Tuple[float, float]] = None
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Compute time-frequency spectrogram.

    Args:
        signal: Input signal
        window_size: Window size in seconds
        overlap: Window overlap ratio
        freq_range: Frequency range to include (min_freq, max_freq)

    Returns:
        Tuple of (times, frequencies, spectrogram)
        - times: Time bins (shape: [num_time_bins])
        - frequencies: Frequency bins (shape: [num_freq_bins])
        - spectrogram: Power values (shape: [num_freq_bins, num_time_bins])

    Performance:
        - Processing time: <100ms for 1 minute of data
        - Memory usage: O(num_freq_bins × num_time_bins)
    """
    pass
```

### PPG-Specific Features

```python
def extract_ppg_features(
    ppg_signal: Signal,
    detect_peaks: bool = True
) -> Dict[str, Any]:
    """
    Extract PPG-specific features including heart rate and HRV.

    Args:
        ppg_signal: PPG signal data
        detect_peaks: Whether to perform peak detection

    Returns:
        Dictionary containing:
        {
            'heart_rate_bpm': float,           # Average heart rate
            'heart_rate_std': float,           # HR variability
            'ibi_mean_ms': float,              # Inter-beat interval
            'ibi_std_ms': float,               # IBI variability
            'rmssd': float,                    # Root mean square of successive differences
            'sdnn': float,                     # Standard deviation of NN intervals
            'pnn50': float,                    # % of intervals >50ms different
            'lf_hf_ratio': float,              # LF/HF power ratio
            'peak_indices': np.ndarray,        # Peak locations
            'peak_amplitudes': np.ndarray,     # Peak amplitudes
            'perfusion_index': float           # Signal strength indicator
        }

    Raises:
        ValueError: If signal type is not PPG
        RuntimeError: If peak detection fails

    Performance:
        - Processing time: <50ms for 1 minute of data
        - Peak detection accuracy: >95% for clean signals
    """
    pass

def estimate_heart_rate(
    ppg_signal: Signal,
    window_size: float = 10.0,
    method: str = 'autocorrelation'
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Estimate instantaneous heart rate from PPG.

    Args:
        ppg_signal: PPG signal data
        window_size: Window size in seconds for HR estimation
        method: Estimation method
                - 'autocorrelation': Autocorrelation-based
                - 'peak_detection': Peak-to-peak intervals
                - 'fft': Frequency-domain estimation

    Returns:
        Tuple of (times, heart_rates) arrays
        - times: Time points in seconds
        - heart_rates: HR values in BPM

    Raises:
        ValueError: If method is unsupported

    Performance:
        - Processing time: <30ms for 1 minute of data
        - Temporal resolution: window_size / 2
        - Accuracy: ±2 BPM for clean signals
    """
    pass
```

### Audio Features for Apnea Detection

```python
def extract_audio_features(
    audio_signal: Signal,
    frame_length: float = 0.025,
    hop_length: float = 0.010,
    feature_types: Optional[List[str]] = None
) -> Dict[str, npt.NDArray]:
    """
    Extract audio features for respiratory event detection.

    Args:
        audio_signal: Audio signal data
        frame_length: Frame length in seconds (default: 25ms)
        hop_length: Hop length in seconds (default: 10ms)
        feature_types: List of features to extract
                      Available: ['mel_spectrogram', 'mfcc', 'spectral_centroid',
                                 'spectral_rolloff', 'zero_crossing_rate',
                                 'rms_energy', 'pitch']
                      If None, extract all features

    Returns:
        Dictionary mapping feature names to 2D arrays (shape: [num_features, num_frames])

    Raises:
        ValueError: If signal type is not AUDIO
        ValueError: If frame_length or hop_length are invalid

    Performance:
        - Processing time: <200ms for 1 minute of audio @ 16kHz
        - Memory usage: ~5MB for 1 minute of features

    Example:
        >>> features = extract_audio_features(audio_signal, feature_types=['mfcc', 'mel_spectrogram'])
        >>> print(features['mfcc'].shape)
        (13, 6000)  # 13 MFCC coefficients, 6000 frames for 1 minute
    """
    pass

def detect_respiratory_events(
    audio_signal: Signal,
    sensitivity: float = 0.5
) -> Dict[str, Any]:
    """
    Detect respiratory events (snoring, apnea, hypopnea) from audio.

    Args:
        audio_signal: Audio signal data
        sensitivity: Detection sensitivity (0.0-1.0, higher = more sensitive)

    Returns:
        Dictionary containing:
        {
            'events': List[Dict],  # List of detected events
            'event_count': int,    # Total number of events
            'snoring_percentage': float,  # % of time spent snoring
            'silence_percentage': float,  # % of time in silence (potential apnea)
            'breathing_regularity': float  # Breathing pattern regularity score
        }
        Each event in 'events' list contains:
        {
            'start_time': float,   # Event start in seconds
            'end_time': float,     # Event end in seconds
            'event_type': str,     # 'snoring', 'apnea', 'hypopnea'
            'confidence': float    # Detection confidence (0.0-1.0)
        }

    Performance:
        - Processing time: <300ms for 1 minute of audio
        - Detection accuracy: >85% compared to manual annotation
    """
    pass
```

## Quality Assessment API

```python
def assess_signal_quality(
    signal: Signal,
    quality_metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Assess physiological signal quality.

    Args:
        signal: Input signal
        quality_metrics: List of metrics to compute
                        Available: ['snr', 'saturation_ratio', 'artifact_ratio',
                                   'signal_stability', 'perfusion_quality']
                        If None, compute all applicable metrics

    Returns:
        Dictionary containing quality scores (0.0-1.0 for each metric)
        {
            'snr': float,              # Signal-to-noise ratio (normalized)
            'saturation_ratio': float, # % of saturated samples
            'artifact_ratio': float,   # % of samples with artifacts
            'signal_stability': float, # Temporal stability score
            'perfusion_quality': float,# PPG perfusion quality (PPG only)
            'overall_quality': float   # Composite quality score
        }

    Raises:
        ValueError: If signal is empty

    Performance:
        - Processing time: <20ms for 1 minute of data

    Quality Interpretation:
        - 0.9-1.0: Excellent quality
        - 0.7-0.9: Good quality
        - 0.5-0.7: Fair quality (usable with caution)
        - 0.0-0.5: Poor quality (should be excluded)
    """
    pass

def detect_signal_artifacts(
    signal: Signal,
    artifact_types: Optional[List[str]] = None
) -> Dict[str, npt.NDArray[np.bool_]]:
    """
    Detect various types of signal artifacts.

    Args:
        signal: Input signal
        artifact_types: List of artifact types to detect
                       Available: ['saturation', 'motion', 'disconnection',
                                  'powerline', 'baseline_drift']
                       If None, detect all types

    Returns:
        Dictionary mapping artifact types to boolean masks
        (True = artifact detected at that sample)

    Performance:
        - Processing time: <30ms for 1 minute of data
        - Detection sensitivity: 85-95% depending on artifact type
    """
    pass
```

## Batch Processing API

```python
def process_signal_batch(
    signals: List[Signal],
    processing_pipeline: List[Dict[str, Any]],
    n_jobs: int = -1
) -> List[Signal]:
    """
    Process multiple signals with same pipeline in parallel.

    Args:
        signals: List of input signals
        processing_pipeline: List of processing operations
                            Example:
                            [
                                {'function': 'bandpass_filter', 'low_freq': 0.5, 'high_freq': 5.0},
                                {'function': 'normalize', 'method': 'zscore'},
                                {'function': 'resample', 'target_rate': 50.0}
                            ]
        n_jobs: Number of parallel jobs (-1 = use all cores)

    Returns:
        List of processed signals

    Raises:
        ValueError: If pipeline contains invalid operations

    Performance:
        - Processing time: ~1/n_jobs of sequential processing
        - Memory usage: O(n_signals × signal_size)
    """
    pass

def extract_features_batch(
    signals: List[Signal],
    feature_extractors: List[str],
    n_jobs: int = -1
) -> npt.NDArray[np.float32]:
    """
    Extract features from multiple signals in parallel.

    Args:
        signals: List of input signals
        feature_extractors: List of feature extraction functions
        n_jobs: Number of parallel jobs (-1 = use all cores)

    Returns:
        Feature matrix (shape: [num_signals, num_features])

    Performance:
        - Processing time: ~1/n_jobs of sequential processing
        - Memory efficient with chunked processing
    """
    pass
```

## Configuration and Utilities

```python
class PreprocessingConfig:
    """Configuration for signal preprocessing pipeline."""

    @dataclass
    class FilterConfig:
        """Filter configuration."""
        type: str  # 'bandpass', 'notch', 'lowpass', 'highpass'
        low_freq: Optional[float] = None
        high_freq: Optional[float] = None
        order: int = 4

    @dataclass
    class ResamplingConfig:
        """Resampling configuration."""
        target_rate: float
        method: str = 'polyphase'

    @dataclass
    class NormalizationConfig:
        """Normalization configuration."""
        method: str = 'zscore'
        axis: Optional[int] = None

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize preprocessing configuration.

        Config format:
        {
            'filters': [
                {'type': 'bandpass', 'low_freq': 0.5, 'high_freq': 5.0, 'order': 4},
                {'type': 'notch', 'freq': 60.0}
            ],
            'resampling': {'target_rate': 50.0, 'method': 'polyphase'},
            'normalization': {'method': 'zscore'},
            'artifact_removal': {
                'motion': {'threshold': 0.5},
                'baseline': {'window_size': 5.0}
            }
        }
        """
        pass

def load_signal(
    file_path: str,
    signal_type: SignalType,
    file_format: str = 'auto'
) -> Signal:
    """
    Load signal from file.

    Args:
        file_path: Path to signal file
        signal_type: Type of signal
        file_format: File format ('csv', 'hdf5', 'npy', 'auto')

    Returns:
        Signal object

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is unsupported

    Supported formats:
        - CSV: timestamp, value columns
        - HDF5: structured datasets with metadata
        - NPY: numpy array format
        - WAV: audio signals (automatically detected)
    """
    pass

def save_signal(
    signal: Signal,
    file_path: str,
    file_format: str = 'auto',
    compress: bool = True
) -> None:
    """
    Save signal to file.

    Args:
        signal: Signal object to save
        file_path: Output file path
        file_format: File format ('csv', 'hdf5', 'npy', 'auto')
        compress: Whether to compress data

    Raises:
        ValueError: If file format is unsupported
    """
    pass
```

## Usage Examples

### Example 1: Basic Signal Preprocessing

```python
from physiokit import Signal, SignalType, bandpass_filter, normalize, resample

# Load PPG signal
ppg = Signal(
    data=raw_ppg_data,
    sample_rate=100.0,
    signal_type=SignalType.PPG
)

# Apply bandpass filter (0.5-5 Hz for PPG)
ppg_filtered = bandpass_filter(ppg, low_freq=0.5, high_freq=5.0, order=4)

# Normalize
ppg_normalized = normalize(ppg_filtered, method='zscore')

# Resample to 50 Hz for model input
ppg_resampled = resample(ppg_normalized, target_rate=50.0, method='polyphase')

print(f"Original: {ppg.num_samples} samples @ {ppg.sample_rate} Hz")
print(f"Processed: {ppg_resampled.num_samples} samples @ {ppg_resampled.sample_rate} Hz")
```

### Example 2: Feature Extraction

```python
from physiokit import extract_ppg_features, extract_frequency_features

# Extract PPG-specific features
ppg_features = extract_ppg_features(ppg_signal, detect_peaks=True)
print(f"Heart Rate: {ppg_features['heart_rate_bpm']:.1f} BPM")
print(f"HRV RMSSD: {ppg_features['rmssd']:.1f} ms")
print(f"LF/HF Ratio: {ppg_features['lf_hf_ratio']:.2f}")

# Extract frequency-domain features
freq_features = extract_frequency_features(ppg_signal, window_size=30.0)
print(f"Band Powers: {freq_features['band_power']}")
print(f"Peak Frequency: {freq_features['peak_frequency']:.3f} Hz")
```

### Example 3: Audio Processing for Apnea Detection

```python
from physiokit import Signal, SignalType, extract_audio_features, detect_respiratory_events

# Load audio signal
audio = Signal(
    data=audio_data,
    sample_rate=16000.0,
    signal_type=SignalType.AUDIO
)

# Extract audio features
features = extract_audio_features(
    audio,
    frame_length=0.025,
    hop_length=0.010,
    feature_types=['mel_spectrogram', 'mfcc']
)

# Detect respiratory events
events = detect_respiratory_events(audio, sensitivity=0.7)
print(f"Detected {events['event_count']} respiratory events")
print(f"Snoring: {events['snoring_percentage']:.1f}% of recording")

for event in events['events']:
    print(f"{event['event_type']}: {event['start_time']:.1f}s - {event['end_time']:.1f}s "
          f"(confidence: {event['confidence']:.2f})")
```

### Example 4: Quality Assessment

```python
from physiokit import assess_signal_quality, detect_signal_artifacts

# Assess signal quality
quality = assess_signal_quality(ppg_signal)
print(f"Overall Quality: {quality['overall_quality']:.2f}")
print(f"SNR: {quality['snr']:.2f}")
print(f"Artifact Ratio: {quality['artifact_ratio']*100:.1f}%")

# Detect specific artifacts
artifacts = detect_signal_artifacts(ppg_signal, artifact_types=['motion', 'saturation'])
motion_percentage = artifacts['motion'].mean() * 100
print(f"Motion Artifacts: {motion_percentage:.1f}% of signal")

# Only use signal if quality is sufficient
if quality['overall_quality'] >= 0.7:
    print("Signal quality is good, proceeding with analysis")
else:
    print("Signal quality is poor, consider re-recording")
```

### Example 5: Batch Processing

```python
from physiokit import process_signal_batch, extract_features_batch

# Define preprocessing pipeline
pipeline = [
    {'function': 'bandpass_filter', 'low_freq': 0.5, 'high_freq': 5.0},
    {'function': 'normalize', 'method': 'zscore'},
    {'function': 'resample', 'target_rate': 50.0}
]

# Process multiple signals in parallel
processed_signals = process_signal_batch(
    signals=[signal1, signal2, signal3],
    processing_pipeline=pipeline,
    n_jobs=-1  # Use all CPU cores
)

# Extract features from all signals
feature_matrix = extract_features_batch(
    signals=processed_signals,
    feature_extractors=['extract_ppg_features', 'extract_time_domain_features'],
    n_jobs=-1
)

print(f"Feature matrix shape: {feature_matrix.shape}")
```

## Performance Characteristics

### Processing Latency

| Operation | 1-min data | 5-min data | Memory |
|-----------|-----------|-----------|---------|
| Bandpass Filter | <10ms | <50ms | O(n) |
| Resampling | <20ms | <100ms | O(n) |
| Feature Extraction | <50ms | <200ms | O(1) |
| Quality Assessment | <20ms | <80ms | O(n) |
| Audio Features | <200ms | <1s | O(n) |

### Accuracy Guarantees

- **Heart Rate Estimation**: ±2 BPM for signals with quality >0.7
- **Peak Detection**: >95% sensitivity for clean PPG signals
- **Respiratory Event Detection**: >85% accuracy compared to manual annotation
- **Quality Assessment**: >90% agreement with expert evaluation

## Error Codes

| Code | Error Type | Description |
|------|-----------|-------------|
| P001 | ValueError | Invalid signal data (NaN, Inf, or empty) |
| P002 | ValueError | Invalid sample rate (<=0) |
| P003 | ValueError | Invalid frequency range for filter |
| P004 | ValueError | Mismatched signal lengths |
| P005 | RuntimeError | Peak detection failed |
| P006 | RuntimeError | Feature extraction failed |
| P007 | FileNotFoundError | Signal file not found |
| P008 | ValueError | Unsupported file format |

## Version Compatibility

- **PhysioKit Version**: >=1.0.0
- **NumPy Version**: >=1.24.0
- **SciPy Version**: >=1.10.0
- **Librosa Version**: >=0.10.0 (for audio processing)
- **Python Version**: >=3.9

## References

- PhysioKit Documentation: https://github.com/AmbiqAI/physiokit
- Signal Processing Algorithms: scipy.signal
- Audio Feature Extraction: librosa
