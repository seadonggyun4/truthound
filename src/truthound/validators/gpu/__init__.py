"""GPU-accelerated validators using RAPIDS cuDF.

This module provides GPU-accelerated validation capabilities using NVIDIA RAPIDS cuDF.
When GPU is available, validators automatically use CUDA for significant performance gains.

**Requirements:**
- NVIDIA GPU with CUDA support
- cuDF library (part of RAPIDS)
- Optional: cuml for ML-based validators

**Features:**
- Automatic GPU/CPU fallback
- Unified API with Polars validators
- GPU memory management
- Multi-GPU support (future)

**Installation:**

    # Install RAPIDS cuDF (requires CUDA toolkit)
    conda install -c rapidsai -c conda-forge cudf=24.02

**Usage Example:**

    from truthound.validators.gpu import (
        GPUNullValidator,
        GPURangeValidator,
        is_gpu_available,
    )

    # Check GPU availability
    if is_gpu_available():
        validator = GPUNullValidator(column="customer_id")
        issues = validator.validate(df.lazy())  # Uses GPU
    else:
        # Fallback to CPU
        from truthound.validators.completeness import NullValidator
        validator = NullValidator(column="customer_id")
        issues = validator.validate(df.lazy())

**Auto-fallback Example:**

    from truthound.validators.gpu import GPUValidator

    # Automatically uses GPU if available, CPU otherwise
    validator = GPUValidator.create_null_validator(column="customer_id")
    issues = validator.validate(df.lazy())

Validators:
    GPUNullValidator: GPU-accelerated null value detection
    GPURangeValidator: GPU-accelerated range validation
    GPUPatternValidator: GPU-accelerated regex pattern matching
    GPUUniqueValidator: GPU-accelerated uniqueness validation
    GPUStatisticsValidator: GPU-accelerated statistical analysis
"""

from truthound.validators.gpu.base import (
    # GPU detection
    is_gpu_available,
    get_gpu_info,
    GPUConfig,
    # Base classes
    GPUValidator,
    GPUValidatorMixin,
    # Conversion utilities
    polars_to_cudf,
    cudf_to_polars,
)

from truthound.validators.gpu.validators import (
    # Core validators
    GPUNullValidator,
    GPURangeValidator,
    GPUPatternValidator,
    GPUUniqueValidator,
    GPUStatisticsValidator,
    # Factory
    create_gpu_validator,
)

__all__ = [
    # GPU detection
    "is_gpu_available",
    "get_gpu_info",
    "GPUConfig",
    # Base classes
    "GPUValidator",
    "GPUValidatorMixin",
    # Conversion utilities
    "polars_to_cudf",
    "cudf_to_polars",
    # Core validators
    "GPUNullValidator",
    "GPURangeValidator",
    "GPUPatternValidator",
    "GPUUniqueValidator",
    "GPUStatisticsValidator",
    # Factory
    "create_gpu_validator",
]
