# Model Inference API Contract

## Overview

This document defines the API contract for TFLite model inference on edge devices for sleep apnea detection. The inference API provides real-time detection with strict performance constraints: <100ms latency and <2MB model size.

## Model Specifications

### Model Variants

| Model Type | Size | Latency | Accuracy | Use Case |
|------------|------|---------|----------|----------|
| Acoustic Apnea | <1.5 MB | <80ms | >85% | Audio-only detection |
| Multimodal Lightweight | <2.0 MB | <100ms | >88% | PPG+SpO2+Accel+Audio |
| Clinical Research | <5.0 MB | <200ms | >92% | High-accuracy validation |

### Input Requirements

All models accept fixed-size input tensors with the following characteristics:

- **Data Type**: float32 (quantized to int8 internally)
- **Normalization**: z-score normalized (mean=0, std=1)
- **Window Size**: 30-second or 60-second sliding windows
- **Sample Rate**: Model-specific (typically 25-50 Hz for physiological, 16kHz for audio)

## Core Inference API

### TFLite Interpreter Interface

```python
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class ModelType(Enum):
    """Enumeration of supported model types."""
    ACOUSTIC_APNEA = "acoustic_apnea"
    MULTIMODAL_LIGHTWEIGHT = "multimodal_lightweight"
    CLINICAL_RESEARCH = "clinical_research"

@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    model_path: Path
    num_threads: int = 4
    use_gpu: bool = False
    use_nnapi: bool = False  # Android Neural Networks API
    use_xnnpack: bool = True  # Optimized CPU inference
    allow_fp16: bool = True   # Half-precision on compatible hardware

@dataclass
class TensorSpec:
    """Specification for input/output tensors."""
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    quantization: Optional[Dict[str, Any]] = None  # scale, zero_point

class TFLiteInference:
    """TensorFlow Lite inference interface for sleep apnea detection."""

    def __init__(self, config: InferenceConfig):
        """
        Initialize TFLite interpreter.

        Args:
            config: Inference configuration

        Raises:
            FileNotFoundError: If model file does not exist
            RuntimeError: If interpreter initialization fails
            ValueError: If model format is invalid

        Performance:
            - Initialization time: <500ms
            - Memory overhead: <10MB
        """
        self.config = config
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._model_metadata = None

    def load_model(self) -> None:
        """
        Load TFLite model and allocate tensors.

        Raises:
            RuntimeError: If model loading fails
            MemoryError: If insufficient memory for tensor allocation

        Performance:
            - Load time: <200ms
            - Memory allocation: Based on model size + working memory
        """
        pass

    def get_input_specs(self) -> List[TensorSpec]:
        """
        Get input tensor specifications.

        Returns:
            List of input tensor specifications

        Example:
            >>> specs = inference.get_input_specs()
            >>> for spec in specs:
            ...     print(f"{spec.name}: {spec.shape} ({spec.dtype})")
            ppg_input: (1, 3000, 1) (float32)
            spo2_input: (1, 3000, 1) (float32)
            accel_input: (1, 3000, 3) (float32)
        """
        pass

    def get_output_specs(self) -> List[TensorSpec]:
        """
        Get output tensor specifications.

        Returns:
            List of output tensor specifications

        Example:
            >>> specs = inference.get_output_specs()
            >>> for spec in specs:
            ...     print(f"{spec.name}: {spec.shape} ({spec.dtype})")
            apnea_probability: (1, 2) (float32)
            event_type: (1, 4) (float32)
        """
        pass

    def predict(
        self,
        inputs: Dict[str, npt.NDArray],
        return_all_outputs: bool = False
    ) -> Dict[str, npt.NDArray]:
        """
        Run inference on input data.

        Args:
            inputs: Dictionary mapping input names to numpy arrays
                   Must match input tensor specifications
            return_all_outputs: Whether to return all output tensors
                               (False = only main prediction)

        Returns:
            Dictionary mapping output names to prediction arrays

        Raises:
            ValueError: If input shape/dtype mismatch
            RuntimeError: If inference fails

        Performance:
            - Latency: <100ms (model-dependent)
            - Throughput: >10 inferences/second
            - Memory: <50MB working memory

        Example:
            >>> inputs = {
            ...     'ppg_input': ppg_data,      # shape: (1, 3000, 1)
            ...     'spo2_input': spo2_data,    # shape: (1, 3000, 1)
            ...     'accel_input': accel_data   # shape: (1, 3000, 3)
            ... }
            >>> outputs = inference.predict(inputs)
            >>> apnea_prob = outputs['apnea_probability'][0]
            >>> print(f"Apnea probability: {apnea_prob[1]:.3f}")
        """
        pass

    def predict_batch(
        self,
        inputs_batch: List[Dict[str, npt.NDArray]],
        batch_size: int = 8
    ) -> List[Dict[str, npt.NDArray]]:
        """
        Run inference on batch of inputs.

        Args:
            inputs_batch: List of input dictionaries
            batch_size: Maximum batch size for processing

        Returns:
            List of output dictionaries

        Performance:
            - Processes in chunks of batch_size
            - Total time: ~(num_samples / batch_size) × inference_time
        """
        pass

    def benchmark(
        self,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark model inference performance.

        Args:
            num_runs: Number of inference runs for benchmarking
            warmup_runs: Number of warmup runs (excluded from stats)

        Returns:
            Dictionary containing performance metrics:
            {
                'mean_latency_ms': float,
                'std_latency_ms': float,
                'min_latency_ms': float,
                'max_latency_ms': float,
                'p50_latency_ms': float,
                'p95_latency_ms': float,
                'p99_latency_ms': float,
                'throughput_per_second': float
            }

        Performance:
            - Benchmark time: ~(warmup_runs + num_runs) × inference_time
        """
        pass

    def get_model_metadata(self) -> Dict[str, Any]:
        """
        Get model metadata and configuration.

        Returns:
            Dictionary containing:
            {
                'model_type': str,
                'version': str,
                'creation_date': str,
                'input_tensors': List[Dict],
                'output_tensors': List[Dict],
                'model_size_bytes': int,
                'quantization_scheme': str,
                'training_metrics': Dict[str, float],
                'target_performance': Dict[str, float]
            }
        """
        pass
```

## Input Tensor Specifications

### 1. Acoustic Apnea Detection Model

```python
INPUT_SPEC_ACOUSTIC = {
    'audio_input': TensorSpec(
        name='audio_input',
        shape=(1, 480000, 1),  # 30 seconds @ 16kHz
        dtype=np.float32,
        quantization={
            'scale': 0.003921568859368563,  # 1/255
            'zero_point': -128
        }
    )
}

OUTPUT_SPEC_ACOUSTIC = {
    'apnea_probability': TensorSpec(
        name='apnea_probability',
        shape=(1, 2),  # [normal, apnea]
        dtype=np.float32,
        quantization=None  # Dequantized output
    ),
    'confidence_score': TensorSpec(
        name='confidence_score',
        shape=(1, 1),
        dtype=np.float32,
        quantization=None
    )
}
```

**Input Preprocessing**:
```python
def preprocess_audio_input(
    audio_data: npt.NDArray,
    sample_rate: int = 16000,
    window_size: float = 30.0
) -> npt.NDArray:
    """
    Preprocess audio data for acoustic apnea model.

    Args:
        audio_data: Raw audio samples
        sample_rate: Audio sample rate (Hz)
        window_size: Window size in seconds

    Returns:
        Preprocessed audio tensor (shape: [1, num_samples, 1])

    Processing steps:
        1. Resample to 16kHz if needed
        2. Extract 30-second window
        3. Apply bandpass filter (50-2000 Hz)
        4. Z-score normalization
        5. Reshape to model input format
    """
    pass
```

### 2. Multimodal Lightweight Model

```python
INPUT_SPEC_MULTIMODAL = {
    'ppg_input': TensorSpec(
        name='ppg_input',
        shape=(1, 1500, 1),  # 30 seconds @ 50Hz
        dtype=np.float32,
        quantization={'scale': 0.007843137718737125, 'zero_point': 0}
    ),
    'spo2_input': TensorSpec(
        name='spo2_input',
        shape=(1, 1500, 1),  # 30 seconds @ 50Hz
        dtype=np.float32,
        quantization={'scale': 0.007843137718737125, 'zero_point': 0}
    ),
    'accel_input': TensorSpec(
        name='accel_input',
        shape=(1, 1500, 3),  # 30 seconds @ 50Hz, 3 axes
        dtype=np.float32,
        quantization={'scale': 0.007843137718737125, 'zero_point': 0}
    ),
    'audio_features': TensorSpec(
        name='audio_features',
        shape=(1, 128, 128),  # Mel-spectrogram
        dtype=np.float32,
        quantization={'scale': 0.003921568859368563, 'zero_point': -128}
    )
}

OUTPUT_SPEC_MULTIMODAL = {
    'apnea_probability': TensorSpec(
        name='apnea_probability',
        shape=(1, 2),  # [normal, apnea]
        dtype=np.float32
    ),
    'event_type': TensorSpec(
        name='event_type',
        shape=(1, 4),  # [none, obstructive, central, mixed]
        dtype=np.float32
    ),
    'severity_score': TensorSpec(
        name='severity_score',
        shape=(1, 1),  # 0-1 continuous severity
        dtype=np.float32
    )
}
```

**Input Preprocessing**:
```python
def preprocess_multimodal_inputs(
    ppg_data: npt.NDArray,
    spo2_data: npt.NDArray,
    accel_data: npt.NDArray,
    audio_data: npt.NDArray,
    sample_rate: int = 50,
    audio_sample_rate: int = 16000
) -> Dict[str, npt.NDArray]:
    """
    Preprocess multimodal inputs for lightweight model.

    Args:
        ppg_data: PPG signal
        spo2_data: SpO2 signal
        accel_data: Accelerometer data (3-axis)
        audio_data: Audio signal
        sample_rate: Physiological signal sample rate
        audio_sample_rate: Audio sample rate

    Returns:
        Dictionary of preprocessed inputs ready for inference

    Processing steps:
        1. Resample all signals to target rate (50 Hz)
        2. Extract synchronized 30-second windows
        3. Apply signal-specific filtering
        4. Z-score normalization per modality
        5. Compute mel-spectrogram from audio
        6. Reshape to model input format
    """
    pass
```

### 3. Clinical Research Model

```python
INPUT_SPEC_CLINICAL = {
    'ppg_input': TensorSpec(
        name='ppg_input',
        shape=(1, 6000, 1),  # 120 seconds @ 50Hz
        dtype=np.float32,
        quantization={'scale': 0.007843137718737125, 'zero_point': 0}
    ),
    'spo2_input': TensorSpec(
        name='spo2_input',
        shape=(1, 6000, 1),
        dtype=np.float32,
        quantization={'scale': 0.007843137718737125, 'zero_point': 0}
    ),
    'accel_input': TensorSpec(
        name='accel_input',
        shape=(1, 6000, 3),
        dtype=np.float32,
        quantization={'scale': 0.007843137718737125, 'zero_point': 0}
    ),
    'audio_input': TensorSpec(
        name='audio_input',
        shape=(1, 1920000, 1),  # 120 seconds @ 16kHz
        dtype=np.float32,
        quantization={'scale': 0.003921568859368563, 'zero_point': -128}
    ),
    'respiratory_input': TensorSpec(
        name='respiratory_input',
        shape=(1, 6000, 1),
        dtype=np.float32,
        quantization={'scale': 0.007843137718737125, 'zero_point': 0}
    )
}

OUTPUT_SPEC_CLINICAL = {
    'apnea_probability': TensorSpec(
        name='apnea_probability',
        shape=(1, 2),
        dtype=np.float32
    ),
    'event_type': TensorSpec(
        name='event_type',
        shape=(1, 5),  # [none, obstructive, central, mixed, hypopnea]
        dtype=np.float32
    ),
    'ahi_estimate': TensorSpec(
        name='ahi_estimate',
        shape=(1, 1),  # Estimated AHI
        dtype=np.float32
    ),
    'attention_weights': TensorSpec(
        name='attention_weights',
        shape=(1, 6000, 5),  # Temporal attention for each modality
        dtype=np.float32
    )
}
```

## Output Interpretation

### Classification Outputs

```python
def interpret_predictions(
    outputs: Dict[str, npt.NDArray],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Interpret model predictions into actionable results.

    Args:
        outputs: Raw model outputs from inference
        threshold: Classification threshold for apnea detection

    Returns:
        Dictionary containing:
        {
            'has_apnea': bool,
            'apnea_probability': float,
            'event_type': str,  # 'none', 'obstructive', 'central', 'mixed', 'hypopnea'
            'event_type_probabilities': Dict[str, float],
            'confidence': float,
            'severity': str  # 'none', 'mild', 'moderate', 'severe'
        }

    Example:
        >>> results = interpret_predictions(outputs, threshold=0.5)
        >>> if results['has_apnea']:
        ...     print(f"Apnea detected: {results['event_type']}")
        ...     print(f"Probability: {results['apnea_probability']:.2f}")
        ...     print(f"Severity: {results['severity']}")
    """
    pass

def compute_ahi_from_predictions(
    predictions: List[Dict[str, Any]],
    total_sleep_time_hours: float
) -> Dict[str, Any]:
    """
    Compute Apnea-Hypopnea Index from sliding window predictions.

    Args:
        predictions: List of prediction results from interpret_predictions()
        total_sleep_time_hours: Total sleep duration in hours

    Returns:
        Dictionary containing:
        {
            'ahi': float,  # Events per hour
            'total_events': int,
            'obstructive_count': int,
            'central_count': int,
            'mixed_count': int,
            'hypopnea_count': int,
            'severity_category': str  # 'normal', 'mild', 'moderate', 'severe'
        }

    AHI Severity Categories:
        - Normal: AHI < 5
        - Mild: 5 ≤ AHI < 15
        - Moderate: 15 ≤ AHI < 30
        - Severe: AHI ≥ 30
    """
    pass
```

## Streaming Inference

### Real-Time Processing

```python
class StreamingInference:
    """Real-time streaming inference for continuous monitoring."""

    def __init__(
        self,
        model_path: Path,
        window_size: float = 30.0,
        stride: float = 10.0,
        buffer_size: int = 10
    ):
        """
        Initialize streaming inference.

        Args:
            model_path: Path to TFLite model
            window_size: Analysis window size in seconds
            stride: Window stride in seconds (overlap = window_size - stride)
            buffer_size: Number of windows to buffer

        Performance:
            - Latency: stride + inference_time
            - Memory: buffer_size × window_memory
        """
        pass

    def add_samples(
        self,
        samples: Dict[str, npt.NDArray]
    ) -> Optional[Dict[str, Any]]:
        """
        Add new samples to processing buffer.

        Args:
            samples: Dictionary of new samples for each modality

        Returns:
            Prediction result if window is complete, None otherwise

        Example:
            >>> stream = StreamingInference(model_path, window_size=30.0, stride=10.0)
            >>> while recording:
            ...     new_samples = get_sensor_data(duration=0.1)  # 100ms chunks
            ...     result = stream.add_samples(new_samples)
            ...     if result is not None:
            ...         print(f"Apnea probability: {result['apnea_probability']:.3f}")
        """
        pass

    def get_buffered_predictions(self) -> List[Dict[str, Any]]:
        """
        Get all buffered predictions.

        Returns:
            List of recent predictions (up to buffer_size)
        """
        pass

    def reset(self) -> None:
        """Reset streaming buffer and state."""
        pass
```

## Performance Optimization

### Hardware Acceleration

```python
def configure_hardware_acceleration(
    use_gpu: bool = False,
    use_nnapi: bool = False,
    use_xnnpack: bool = True,
    num_threads: int = 4
) -> InferenceConfig:
    """
    Configure hardware acceleration for optimal performance.

    Args:
        use_gpu: Enable GPU acceleration (if available)
        use_nnapi: Enable Android NNAPI (Android only)
        use_xnnpack: Enable XNNPACK CPU optimization
        num_threads: Number of CPU threads

    Returns:
        Optimized inference configuration

    Performance impact:
        - GPU: 2-5x speedup on compatible hardware
        - NNAPI: 1.5-3x speedup on modern Android devices
        - XNNPACK: 1.2-2x speedup on ARM CPUs
        - Multi-threading: ~1.5x speedup with 4 threads

    Platform recommendations:
        - iOS (iPhone 8+): use_xnnpack=True, num_threads=4
        - Android (Snapdragon 8xx): use_nnapi=True, use_xnnpack=True
        - Raspberry Pi 4: use_xnnpack=True, num_threads=4
        - Desktop (x86): use_xnnpack=True, num_threads=8
    """
    pass
```

### Memory Optimization

```python
def optimize_memory_usage(
    inference: TFLiteInference,
    enable_memory_arena: bool = True,
    arena_size_mb: int = 20
) -> None:
    """
    Optimize memory usage for edge deployment.

    Args:
        inference: TFLite inference instance
        enable_memory_arena: Use pre-allocated memory arena
        arena_size_mb: Memory arena size in MB

    Memory breakdown:
        - Model: 1-5 MB (depends on model type)
        - Input tensors: 0.5-2 MB
        - Working memory: 5-15 MB
        - Output tensors: <1 MB
        - Total typical: 10-25 MB
    """
    pass
```

## Error Handling

```python
class InferenceError(RuntimeError):
    """Base class for inference errors."""
    pass

class ModelLoadError(InferenceError):
    """Raised when model cannot be loaded."""
    pass

class InputShapeError(InferenceError):
    """Raised when input shape doesn't match expected."""
    pass

class InferenceTimeoutError(InferenceError):
    """Raised when inference exceeds time limit."""
    pass

def validate_input_tensor(
    tensor: npt.NDArray,
    spec: TensorSpec
) -> None:
    """
    Validate input tensor against specification.

    Args:
        tensor: Input tensor to validate
        spec: Expected tensor specification

    Raises:
        InputShapeError: If shape doesn't match
        ValueError: If dtype doesn't match
        ValueError: If tensor contains invalid values (NaN, Inf)
    """
    pass
```

## Usage Examples

### Example 1: Single Inference

```python
from inference import TFLiteInference, InferenceConfig
from pathlib import Path

# Initialize inference
config = InferenceConfig(
    model_path=Path('./models/multimodal_lightweight.tflite'),
    num_threads=4,
    use_xnnpack=True
)

inference = TFLiteInference(config)
inference.load_model()

# Prepare inputs
inputs = {
    'ppg_input': ppg_window,      # (1, 1500, 1)
    'spo2_input': spo2_window,    # (1, 1500, 1)
    'accel_input': accel_window,  # (1, 1500, 3)
    'audio_features': mel_spec    # (1, 128, 128)
}

# Run inference
outputs = inference.predict(inputs)

# Interpret results
results = interpret_predictions(outputs, threshold=0.5)
print(f"Apnea detected: {results['has_apnea']}")
print(f"Event type: {results['event_type']}")
print(f"Confidence: {results['confidence']:.2f}")
```

### Example 2: Batch Processing

```python
# Process multiple windows
windows = load_signal_windows('./data/night_recording.h5')

# Batch inference
batch_results = []
for i in range(0, len(windows), 8):
    batch = windows[i:i+8]
    outputs_batch = inference.predict_batch(batch, batch_size=8)
    batch_results.extend(outputs_batch)

# Compute AHI
predictions = [interpret_predictions(out) for out in batch_results]
ahi_results = compute_ahi_from_predictions(
    predictions,
    total_sleep_time_hours=7.5
)

print(f"AHI: {ahi_results['ahi']:.1f} events/hour")
print(f"Severity: {ahi_results['severity_category']}")
print(f"Total events: {ahi_results['total_events']}")
```

### Example 3: Real-Time Streaming

```python
from inference import StreamingInference

# Initialize streaming
stream = StreamingInference(
    model_path=Path('./models/acoustic_apnea.tflite'),
    window_size=30.0,
    stride=10.0,
    buffer_size=10
)

# Process real-time audio stream
import sounddevice as sd

def audio_callback(indata, frames, time, status):
    """Callback for real-time audio processing."""
    audio_chunk = indata[:, 0]  # Mono audio

    result = stream.add_samples({'audio_input': audio_chunk})

    if result is not None:
        if result['has_apnea']:
            print(f"Warning: Apnea detected! Probability: {result['apnea_probability']:.2f}")
        else:
            print(f"Normal breathing (confidence: {result['confidence']:.2f})")

# Start real-time monitoring
with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
    print("Monitoring started. Press Ctrl+C to stop.")
    sd.sleep(3600000)  # Monitor for 1 hour
```

### Example 4: Benchmarking

```python
# Benchmark model performance
benchmark_results = inference.benchmark(num_runs=100, warmup_runs=10)

print(f"Mean latency: {benchmark_results['mean_latency_ms']:.2f} ms")
print(f"P95 latency: {benchmark_results['p95_latency_ms']:.2f} ms")
print(f"P99 latency: {benchmark_results['p99_latency_ms']:.2f} ms")
print(f"Throughput: {benchmark_results['throughput_per_second']:.1f} inferences/sec")

# Verify performance requirements
assert benchmark_results['p95_latency_ms'] < 100, "Latency requirement not met"
print("Performance requirements satisfied")
```

## Performance Guarantees

### Latency Targets

| Model Type | Target | Typical | Max |
|------------|--------|---------|-----|
| Acoustic | <80ms | 60ms | 100ms |
| Multimodal Lightweight | <100ms | 75ms | 120ms |
| Clinical Research | <200ms | 150ms | 250ms |

### Memory Constraints

| Component | Memory Usage |
|-----------|-------------|
| Model file | 1.5-2.0 MB |
| Input buffers | 0.5-1.0 MB |
| Working memory | 5-10 MB |
| Output buffers | 0.1-0.5 MB |
| **Total** | **<15 MB** |

### Accuracy Targets

| Metric | Acoustic | Multimodal | Clinical |
|--------|----------|------------|----------|
| Sensitivity | >85% | >88% | >92% |
| Specificity | >82% | >86% | >90% |
| AUC-ROC | >0.87 | >0.90 | >0.94 |

## Version Compatibility

- **TensorFlow Lite Version**: >=2.13.0
- **Python Version**: >=3.9 (for development/benchmarking)
- **NumPy Version**: >=1.24.0
- **Platform Support**: Linux, macOS, Windows, iOS, Android

## References

- TensorFlow Lite Documentation: https://www.tensorflow.org/lite
- Model Optimization Guide: https://www.tensorflow.org/model_optimization
- TFLite Benchmark Tool: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark
