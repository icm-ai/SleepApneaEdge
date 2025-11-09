# SleepKit Task API Contract

## Overview

This document defines the API contract for custom SleepKit tasks implementing the BYOT (Bring-Your-Own-Task) pattern for sleep apnea detection. Custom tasks extend the base `sleepkit.Task` interface to provide acoustic apnea detection, multimodal lightweight models, and clinical research models.

## Base Task Interface

### Class: `sleepkit.Task`

All custom tasks must inherit from the base `Task` class and implement the required methods.

```python
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np
import numpy.typing as npt

class Task:
    """Base class for SleepKit tasks."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize task with configuration.

        Args:
            config: Task configuration dictionary loaded from YAML

        Raises:
            ValueError: If configuration is invalid
            KeyError: If required configuration keys are missing
        """
        pass

    def train(
        self,
        dataset_path: Path,
        output_path: Path,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model on provided dataset.

        Args:
            dataset_path: Path to training dataset (HDF5 format)
            output_path: Path to save trained model and checkpoints
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Initial learning rate
            **kwargs: Additional training parameters

        Returns:
            Dictionary containing training metrics:
            {
                'final_loss': float,
                'final_accuracy': float,
                'best_epoch': int,
                'training_time_seconds': float,
                'model_path': str,
                'history': Dict[str, List[float]]
            }

        Raises:
            FileNotFoundError: If dataset_path does not exist
            ValueError: If hyperparameters are invalid
            RuntimeError: If training fails

        Performance:
            - Training time: <24 hours for standard datasets
            - Memory usage: <16GB RAM during training
        """
        pass

    def evaluate(
        self,
        dataset_path: Path,
        model_path: Optional[Path] = None,
        batch_size: int = 32,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate model on test dataset.

        Args:
            dataset_path: Path to evaluation dataset (HDF5 format)
            model_path: Path to trained model (uses last trained if None)
            batch_size: Evaluation batch size
            **kwargs: Additional evaluation parameters

        Returns:
            Dictionary containing evaluation metrics:
            {
                'accuracy': float,
                'precision': float,
                'recall': float,
                'f1_score': float,
                'auc_roc': float,
                'confusion_matrix': List[List[int]],
                'per_class_metrics': Dict[str, Dict[str, float]],
                'inference_time_ms': float
            }

        Raises:
            FileNotFoundError: If dataset_path or model_path does not exist
            ValueError: If model is not compatible with task

        Performance:
            - Evaluation time: <5 minutes for standard test sets
            - Memory usage: <8GB RAM during evaluation
        """
        pass

    def export(
        self,
        model_path: Path,
        output_path: Path,
        target_format: str = 'tflite',
        quantization: str = 'int8',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Export model to edge deployment format.

        Args:
            model_path: Path to trained Keras/TensorFlow model
            output_path: Path to save exported model
            target_format: Export format ('tflite', 'onnx', 'neuralspot')
            quantization: Quantization scheme ('int8', 'float16', 'none')
            **kwargs: Format-specific export parameters

        Returns:
            Dictionary containing export metadata:
            {
                'output_file': str,
                'model_size_bytes': int,
                'quantization_applied': str,
                'input_shape': List[int],
                'output_shape': List[int],
                'estimated_latency_ms': float,
                'estimated_memory_kb': int
            }

        Raises:
            FileNotFoundError: If model_path does not exist
            ValueError: If target_format or quantization is unsupported
            RuntimeError: If export conversion fails

        Performance:
            - Export time: <5 minutes
            - Output model size: <2MB (with int8 quantization)
            - Target inference latency: <100ms
        """
        pass

    def visualize(
        self,
        data_path: Path,
        predictions_path: Optional[Path] = None,
        output_path: Path = Path('./visualizations'),
        plot_type: str = 'all',
        **kwargs
    ) -> List[Path]:
        """
        Generate visualization plots for data and predictions.

        Args:
            data_path: Path to data file (HDF5 format)
            predictions_path: Path to predictions file (optional)
            output_path: Directory to save visualization plots
            plot_type: Type of plots ('confusion_matrix', 'roc_curve',
                      'time_series', 'feature_importance', 'all')
            **kwargs: Plot-specific parameters

        Returns:
            List of paths to generated visualization files

        Raises:
            FileNotFoundError: If data_path does not exist
            ValueError: If plot_type is unsupported

        Performance:
            - Visualization time: <2 minutes for standard datasets
        """
        pass
```

## Custom Task Implementations

### 1. Acoustic Apnea Detection Task

```python
class AcousticApneaTask(Task):
    """Task for detecting apnea events from audio signals."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize acoustic apnea detection task.

        Configuration schema:
        {
            'model_architecture': 'tcn' | 'unet' | 'resnet',
            'input_channels': int (default: 1),
            'sample_rate': int (default: 16000),
            'window_size_seconds': float (default: 30.0),
            'num_classes': int (default: 2),  # apnea/normal
            'feature_extraction': {
                'type': 'mel_spectrogram' | 'mfcc',
                'n_mels': int (default: 128),
                'n_fft': int (default: 2048),
                'hop_length': int (default: 512)
            }
        }
        """
        super().__init__(config)
```

### 2. Multimodal Lightweight Task

```python
class MultimodalLightweightTask(Task):
    """Lightweight multimodal task for edge deployment."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multimodal lightweight task.

        Configuration schema:
        {
            'model_architecture': 'mobilenet' | 'efficientnet',
            'input_modalities': ['ppg', 'spo2', 'accel', 'audio'],
            'fusion_strategy': 'early' | 'late' | 'hybrid',
            'compression_ratio': float (default: 0.25),
            'target_size_mb': float (default: 1.5),
            'target_latency_ms': float (default: 80)
        }
        """
        super().__init__(config)
```

### 3. Clinical Research Task

```python
class ClinicalResearchTask(Task):
    """High-accuracy task for clinical validation."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize clinical research task.

        Configuration schema:
        {
            'model_architecture': 'transformer' | 'attention_unet',
            'input_modalities': ['ppg', 'spo2', 'accel', 'audio', 'respiratory'],
            'temporal_context_seconds': float (default: 300),
            'event_types': ['obstructive', 'central', 'mixed', 'hypopnea'],
            'clinical_metrics': ['ahi', 'odi', 'arousal_index'],
            'ensemble_models': int (default: 5)
        }
        """
        super().__init__(config)
```

## Configuration File Format

### YAML Configuration Schema

```yaml
# Task configuration example: acoustic_apnea.yaml

task: acoustic_apnea  # Task identifier

# Dataset configuration
dataset:
  path: /data/sleep_apnea_audio
  format: hdf5
  train_split: 0.7
  validation_split: 0.15
  test_split: 0.15
  preprocessing:
    normalize: true
    augmentation:
      - type: time_shift
        probability: 0.3
      - type: pitch_shift
        probability: 0.2

# Model configuration
model:
  architecture: tcn
  input_shape: [1920000]  # 2 minutes @ 16kHz
  num_classes: 2
  layers:
    - type: conv1d
      filters: 64
      kernel_size: 7
      activation: relu
    - type: tcn_block
      filters: 128
      kernel_size: 3
      dilation_rates: [1, 2, 4, 8]
    - type: global_avg_pool
    - type: dense
      units: 64
      activation: relu
    - type: dropout
      rate: 0.3
    - type: dense
      units: 2
      activation: softmax

# Training configuration
training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: adam
  loss: categorical_crossentropy
  metrics: [accuracy, precision, recall, auc]
  callbacks:
    - type: early_stopping
      patience: 10
      monitor: val_loss
    - type: reduce_lr
      patience: 5
      factor: 0.5
    - type: model_checkpoint
      save_best_only: true

# Export configuration
export:
  format: tflite
  quantization: int8
  optimization:
    - default
  representative_dataset_size: 100
```

## Method Call Examples

### Example 1: Training a Model

```python
from sleepkit.tasks import AcousticApneaTask
from pathlib import Path
import yaml

# Load configuration
with open('acoustic_apnea.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize task
task = AcousticApneaTask(config)

# Train model
results = task.train(
    dataset_path=Path('/data/sleep_apnea_audio/train.h5'),
    output_path=Path('./models/acoustic_apnea'),
    num_epochs=100,
    batch_size=32,
    learning_rate=0.001
)

print(f"Training completed in {results['training_time_seconds']}s")
print(f"Final accuracy: {results['final_accuracy']:.4f}")
print(f"Model saved to: {results['model_path']}")
```

### Example 2: Evaluating a Model

```python
# Evaluate trained model
metrics = task.evaluate(
    dataset_path=Path('/data/sleep_apnea_audio/test.h5'),
    model_path=Path('./models/acoustic_apnea/best_model.h5'),
    batch_size=64
)

print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
print(f"Inference Time: {metrics['inference_time_ms']:.2f}ms")
```

### Example 3: Exporting for Edge Deployment

```python
# Export to TFLite with int8 quantization
export_info = task.export(
    model_path=Path('./models/acoustic_apnea/best_model.h5'),
    output_path=Path('./models/acoustic_apnea/model.tflite'),
    target_format='tflite',
    quantization='int8'
)

print(f"Model size: {export_info['model_size_bytes'] / 1024:.2f} KB")
print(f"Estimated latency: {export_info['estimated_latency_ms']:.2f}ms")
print(f"Input shape: {export_info['input_shape']}")
print(f"Output shape: {export_info['output_shape']}")
```

### Example 4: Generating Visualizations

```python
# Generate all visualization plots
plot_files = task.visualize(
    data_path=Path('/data/sleep_apnea_audio/test.h5'),
    predictions_path=Path('./predictions/test_predictions.npy'),
    output_path=Path('./visualizations'),
    plot_type='all'
)

for plot_file in plot_files:
    print(f"Generated: {plot_file}")
```

## Error Handling

### Exception Types

```python
class TaskConfigurationError(ValueError):
    """Raised when task configuration is invalid."""
    pass

class DatasetLoadError(RuntimeError):
    """Raised when dataset cannot be loaded or parsed."""
    pass

class ModelTrainingError(RuntimeError):
    """Raised when model training fails."""
    pass

class ModelExportError(RuntimeError):
    """Raised when model export/conversion fails."""
    pass

class InferenceError(RuntimeError):
    """Raised when model inference fails."""
    pass
```

### Error Codes

| Code | Exception | Description |
|------|-----------|-------------|
| E001 | TaskConfigurationError | Missing required configuration key |
| E002 | TaskConfigurationError | Invalid configuration value |
| E003 | DatasetLoadError | Dataset file not found |
| E004 | DatasetLoadError | Dataset format incompatible |
| E005 | ModelTrainingError | Training convergence failure |
| E006 | ModelTrainingError | Out of memory during training |
| E007 | ModelExportError | Unsupported export format |
| E008 | ModelExportError | Quantization failed |
| E009 | InferenceError | Model prediction failed |
| E010 | InferenceError | Input shape mismatch |

## Performance Guarantees

### Training Performance

- **Time Complexity**: O(n × e × b) where n=dataset size, e=epochs, b=batch operations
- **Memory Usage**: <16GB RAM for standard datasets
- **GPU Acceleration**: Supported (CUDA, ROCm)
- **Max Training Time**: <24 hours for full dataset

### Evaluation Performance

- **Throughput**: >1000 samples/second (with GPU)
- **Memory Usage**: <8GB RAM
- **Latency**: <5ms per sample (inference only)

### Export Performance

- **Conversion Time**: <5 minutes
- **Output Model Size**: <2MB (int8 quantization)
- **Compression Ratio**: 75-90% size reduction
- **Accuracy Loss**: <2% from quantization

## Validation Rules

### Configuration Validation

```python
def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate task configuration.

    Rules:
    - 'task' key must be present and valid
    - 'model.architecture' must be supported
    - 'model.input_shape' must be list of positive integers
    - 'training.num_epochs' must be positive integer
    - 'training.batch_size' must be power of 2 (recommended)
    - 'export.quantization' must be in ['int8', 'float16', 'none']
    """
    pass
```

### Input Validation

```python
def validate_input_data(
    data: npt.NDArray,
    expected_shape: tuple,
    expected_dtype: np.dtype
) -> None:
    """
    Validate input data tensor.

    Rules:
    - Data shape must match expected_shape
    - Data dtype must match expected_dtype
    - Data must not contain NaN or Inf values
    - Data values must be within valid range
    """
    pass
```

## Extensibility

### Creating Custom Tasks

To create a new custom task:

1. Inherit from `sleepkit.Task`
2. Implement all required methods: `train()`, `evaluate()`, `export()`, `visualize()`
3. Define task-specific configuration schema
4. Register task in SleepKit task registry

```python
from sleepkit import Task, register_task

@register_task('my_custom_task')
class MyCustomTask(Task):
    """Custom task implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Custom initialization

    # Implement required methods...
```

## Version Compatibility

- **SleepKit Version**: >=2.0.0
- **TensorFlow Version**: >=2.13.0, <3.0.0
- **Python Version**: >=3.9, <3.13
- **NumPy Version**: >=1.24.0
- **H5Py Version**: >=3.8.0

## References

- SleepKit Documentation: https://github.com/AmbiqAI/sleepkit
- TensorFlow Lite Guide: https://www.tensorflow.org/lite
- Model Optimization: https://www.tensorflow.org/model_optimization
