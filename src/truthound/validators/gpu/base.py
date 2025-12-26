"""GPU acceleration base module using RAPIDS cuDF.

This module provides the foundational components for GPU-accelerated
data validation using NVIDIA RAPIDS cuDF library.

Key Features:
- Automatic GPU detection and fallback to CPU
- Efficient Polars <-> cuDF conversion
- GPU memory management
- Unified validation interface

Usage:
    from truthound.validators.gpu.base import (
        is_gpu_available,
        GPUValidator,
        polars_to_cudf,
    )

    if is_gpu_available():
        # Use GPU-accelerated validation
        gdf = polars_to_cudf(df.lazy())
        # ... perform GPU operations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
import warnings

import polars as pl

from truthound.types import Severity
from truthound.validators.base import (
    ValidationIssue,
    Validator,
    ValidatorConfig,
    ValidatorLogger,
)

# Conditional imports for GPU support
CUDF_AVAILABLE = False
CUML_AVAILABLE = False
GPU_DEVICE_COUNT = 0

try:
    import cudf
    CUDF_AVAILABLE = True

    # Try to get GPU count
    try:
        import cupy as cp
        GPU_DEVICE_COUNT = cp.cuda.runtime.getDeviceCount()
    except Exception:
        GPU_DEVICE_COUNT = 1  # Assume at least 1 if cudf works

except ImportError:
    cudf = None  # type: ignore

try:
    import cuml
    CUML_AVAILABLE = True
except ImportError:
    cuml = None  # type: ignore


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available.

    Returns:
        True if RAPIDS cuDF is installed and GPU is accessible
    """
    return CUDF_AVAILABLE and GPU_DEVICE_COUNT > 0


def get_gpu_info() -> dict[str, Any]:
    """Get detailed GPU information.

    Returns:
        Dictionary with GPU information including:
        - available: Whether GPU is available
        - device_count: Number of GPUs
        - cudf_version: cuDF version if available
        - cuml_available: Whether cuML is available
        - devices: List of device info
    """
    info: dict[str, Any] = {
        "available": is_gpu_available(),
        "cudf_available": CUDF_AVAILABLE,
        "cuml_available": CUML_AVAILABLE,
        "device_count": GPU_DEVICE_COUNT,
        "cudf_version": None,
        "devices": [],
    }

    if CUDF_AVAILABLE:
        try:
            info["cudf_version"] = cudf.__version__
        except Exception:
            pass

    if is_gpu_available():
        try:
            import cupy as cp
            for i in range(GPU_DEVICE_COUNT):
                with cp.cuda.Device(i):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    info["devices"].append({
                        "id": i,
                        "name": props.get("name", b"Unknown").decode(),
                        "total_memory_gb": props.get("totalGlobalMem", 0) / (1024**3),
                        "compute_capability": f"{props.get('major', 0)}.{props.get('minor', 0)}",
                    })
        except Exception:
            pass

    return info


@dataclass
class GPUConfig:
    """Configuration for GPU-accelerated validation.

    Attributes:
        device_id: GPU device to use (default: 0)
        memory_limit_gb: Maximum GPU memory to use (None = no limit)
        fallback_to_cpu: Whether to fallback to CPU on GPU errors
        batch_size: Batch size for GPU operations
        prefetch: Whether to prefetch data to GPU
    """
    device_id: int = 0
    memory_limit_gb: float | None = None
    fallback_to_cpu: bool = True
    batch_size: int = 1_000_000
    prefetch: bool = True


def polars_to_cudf(
    lf: pl.LazyFrame,
    columns: list[str] | None = None,
) -> "cudf.DataFrame":
    """Convert Polars LazyFrame to cuDF DataFrame.

    This function efficiently transfers data from Polars to GPU memory
    using Arrow as an intermediate format.

    Args:
        lf: Polars LazyFrame to convert
        columns: Optional list of columns to include

    Returns:
        cuDF DataFrame on GPU

    Raises:
        ImportError: If cuDF is not available
        RuntimeError: If GPU transfer fails
    """
    if not CUDF_AVAILABLE:
        raise ImportError(
            "cuDF is not available. Install RAPIDS cuDF for GPU acceleration: "
            "conda install -c rapidsai -c conda-forge cudf"
        )

    # Collect to Polars DataFrame first
    if columns:
        df = lf.select(columns).collect()
    else:
        df = lf.collect()

    # Convert via Arrow for efficiency
    arrow_table = df.to_arrow()

    # Create cuDF DataFrame from Arrow
    return cudf.DataFrame.from_arrow(arrow_table)


def cudf_to_polars(gdf: "cudf.DataFrame") -> pl.DataFrame:
    """Convert cuDF DataFrame back to Polars DataFrame.

    Args:
        gdf: cuDF DataFrame to convert

    Returns:
        Polars DataFrame
    """
    if not CUDF_AVAILABLE:
        raise ImportError("cuDF is not available")

    # Convert via Arrow
    arrow_table = gdf.to_arrow()
    return pl.from_arrow(arrow_table)


class GPUMemoryManager:
    """Manages GPU memory for validation operations.

    Provides memory tracking, limiting, and cleanup utilities.
    """

    def __init__(self, device_id: int = 0, limit_gb: float | None = None):
        self.device_id = device_id
        self.limit_gb = limit_gb
        self._initial_free: float = 0.0

    def get_memory_info(self) -> dict[str, float]:
        """Get current GPU memory usage.

        Returns:
            Dictionary with 'used_gb', 'free_gb', 'total_gb'
        """
        if not is_gpu_available():
            return {"used_gb": 0.0, "free_gb": 0.0, "total_gb": 0.0}

        try:
            import cupy as cp
            with cp.cuda.Device(self.device_id):
                mempool = cp.get_default_memory_pool()
                total = mempool.total_bytes()
                used = mempool.used_bytes()
                free = cp.cuda.runtime.memGetInfo()[0]

                return {
                    "used_gb": used / (1024**3),
                    "free_gb": free / (1024**3),
                    "total_gb": (used + free) / (1024**3),
                }
        except Exception:
            return {"used_gb": 0.0, "free_gb": 0.0, "total_gb": 0.0}

    def check_memory(self, required_gb: float = 0.0) -> bool:
        """Check if sufficient GPU memory is available.

        Args:
            required_gb: Required memory in GB

        Returns:
            True if sufficient memory is available
        """
        info = self.get_memory_info()
        available = info["free_gb"]

        if self.limit_gb is not None:
            available = min(available, self.limit_gb - info["used_gb"])

        return available >= required_gb

    def cleanup(self) -> None:
        """Release cached GPU memory."""
        if not is_gpu_available():
            return

        try:
            import cupy as cp
            with cp.cuda.Device(self.device_id):
                mempool = cp.get_default_memory_pool()
                mempool.free_all_blocks()
        except Exception:
            pass

    def __enter__(self) -> "GPUMemoryManager":
        self._initial_free = self.get_memory_info()["free_gb"]
        return self

    def __exit__(self, *args: Any) -> None:
        self.cleanup()


class GPUValidatorMixin:
    """Mixin providing GPU acceleration capabilities for validators.

    This mixin adds GPU support to any validator, with automatic
    fallback to CPU processing when GPU is unavailable.
    """

    gpu_config: GPUConfig

    def _to_gpu(
        self,
        lf: pl.LazyFrame,
        columns: list[str] | None = None,
    ) -> "cudf.DataFrame":
        """Transfer data to GPU.

        Args:
            lf: Polars LazyFrame
            columns: Optional column subset

        Returns:
            cuDF DataFrame
        """
        return polars_to_cudf(lf, columns)

    def _from_gpu(self, gdf: "cudf.DataFrame") -> pl.DataFrame:
        """Transfer data from GPU.

        Args:
            gdf: cuDF DataFrame

        Returns:
            Polars DataFrame
        """
        return cudf_to_polars(gdf)

    def _gpu_available(self) -> bool:
        """Check if GPU is available for this operation."""
        if not is_gpu_available():
            return False

        if hasattr(self, 'gpu_config') and self.gpu_config:
            manager = GPUMemoryManager(
                self.gpu_config.device_id,
                self.gpu_config.memory_limit_gb,
            )
            return manager.check_memory(0.1)  # At least 100MB available

        return True


class GPUValidator(Validator, GPUValidatorMixin, ABC):
    """Base class for GPU-accelerated validators.

    This class provides:
    - Automatic GPU/CPU selection
    - Efficient data transfer to/from GPU
    - Memory management
    - Fallback handling

    Subclasses should implement:
    - _validate_gpu(): GPU-accelerated validation logic
    - _validate_cpu(): CPU fallback validation logic

    Example:
        class MyGPUValidator(GPUValidator):
            name = "my_gpu_validator"

            def _validate_gpu(self, gdf: cudf.DataFrame) -> list[ValidationIssue]:
                # GPU-accelerated logic using cuDF
                ...

            def _validate_cpu(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
                # CPU fallback using Polars
                ...
    """

    name: str = "gpu_base"
    category: str = "gpu"

    def __init__(
        self,
        gpu_config: GPUConfig | None = None,
        config: ValidatorConfig | None = None,
        **kwargs: Any,
    ):
        """Initialize GPU validator.

        Args:
            gpu_config: GPU-specific configuration
            config: Standard validator configuration
            **kwargs: Additional config options
        """
        super().__init__(config=config, **kwargs)
        self.gpu_config = gpu_config or GPUConfig()
        self._gpu_logger = ValidatorLogger(f"GPU.{self.name}")

    def validate(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Run validation, using GPU if available.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        if self._gpu_available():
            try:
                self._gpu_logger.debug(f"Using GPU for {self.name}")
                return self._validate_with_gpu(lf)
            except Exception as e:
                if self.gpu_config.fallback_to_cpu:
                    self._gpu_logger.warning(
                        f"GPU validation failed, falling back to CPU: {e}"
                    )
                    return self._validate_cpu(lf)
                raise
        else:
            self._gpu_logger.debug(f"GPU not available, using CPU for {self.name}")
            return self._validate_cpu(lf)

    def _validate_with_gpu(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """Execute GPU validation with memory management.

        Args:
            lf: Input LazyFrame

        Returns:
            List of validation issues
        """
        memory_manager = GPUMemoryManager(
            self.gpu_config.device_id,
            self.gpu_config.memory_limit_gb,
        )

        with memory_manager:
            # Get target columns
            columns = self._get_target_columns(lf)

            # Transfer to GPU
            gdf = self._to_gpu(lf, columns)

            # Run GPU validation
            return self._validate_gpu(gdf)

    @abstractmethod
    def _validate_gpu(self, gdf: "cudf.DataFrame") -> list[ValidationIssue]:
        """GPU-accelerated validation logic.

        Implement this method with cuDF operations.

        Args:
            gdf: cuDF DataFrame on GPU

        Returns:
            List of validation issues
        """
        pass

    @abstractmethod
    def _validate_cpu(self, lf: pl.LazyFrame) -> list[ValidationIssue]:
        """CPU fallback validation logic.

        Implement this method with Polars operations.

        Args:
            lf: Polars LazyFrame

        Returns:
            List of validation issues
        """
        pass

    @classmethod
    def is_available(cls) -> bool:
        """Check if GPU validation is available."""
        return is_gpu_available()

    @classmethod
    def create_null_validator(cls, column: str, **kwargs: Any) -> Validator:
        """Factory method to create appropriate null validator.

        Returns GPUNullValidator if GPU available, otherwise NullValidator.
        """
        from truthound.validators.gpu.validators import GPUNullValidator
        from truthound.validators.completeness import NullValidator

        if is_gpu_available():
            return GPUNullValidator(columns=[column], **kwargs)
        return NullValidator(column=column, **kwargs)

    @classmethod
    def create_range_validator(
        cls,
        column: str,
        min_value: float | None = None,
        max_value: float | None = None,
        **kwargs: Any,
    ) -> Validator:
        """Factory method to create appropriate range validator.

        Returns GPURangeValidator if GPU available, otherwise RangeValidator.
        """
        from truthound.validators.gpu.validators import GPURangeValidator
        from truthound.validators.distribution import RangeValidator

        if is_gpu_available():
            return GPURangeValidator(
                column=column,
                min_value=min_value,
                max_value=max_value,
                **kwargs,
            )
        return RangeValidator(
            column=column,
            min_value=min_value,
            max_value=max_value,
            **kwargs,
        )
