"""
Pydantic schemas for the REST API request / response bodies.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Tensor descriptor
# ---------------------------------------------------------------------------

class TensorSpec(BaseModel):
    """
    JSON-safe representation of a GPU tensor.

    Either supply **data** (base64-encoded raw bytes) or a **fill** strategy.
    """
    shape: List[int] = Field(..., description="Tensor shape, e.g. [32, 4096]")
    dtype: str       = Field("bfloat16", description="dtype: float32|float16|bfloat16|fp8|int8|…")
    data:  Optional[str] = Field(None,  description="Base64-encoded raw tensor bytes (little-endian)")
    fill:  Optional[Union[str, float]] = Field(
        "random",
        description="Fill strategy when data is absent: 'zeros'|'ones'|'random'|<float>",
    )

    @field_validator("dtype")
    @classmethod
    def _validate_dtype(cls, v: str) -> str:
        supported = {
            "float32","fp32","float16","fp16","bfloat16","bf16",
            "float8_e4m3fn","fp8","int8","int32","int64",
        }
        if v.lower() not in supported:
            raise ValueError(f"dtype '{v}' not supported. Choose from {sorted(supported)}")
        return v.lower()

    @field_validator("shape")
    @classmethod
    def _validate_shape(cls, v: list) -> list:
        if not v:
            raise ValueError("shape must be non-empty")
        if any(d <= 0 for d in v):
            raise ValueError("all shape dimensions must be > 0")
        return v


# ---------------------------------------------------------------------------
# Kernel run request
# ---------------------------------------------------------------------------

class KernelRunRequest(BaseModel):
    """
    POST /api/v1/kernels/run
    """
    kernel_name: str = Field(
        ...,
        description="Dot-separated kernel identifier, e.g. 'normalization.rmsnorm'",
        examples=["normalization.rmsnorm", "gemm.bf16_gemm", "activations.swiglu"],
    )
    backend: str = Field(
        ...,
        description="Backend implementation: 'triton' | 'tilelang' | 'cuda'",
        examples=["triton", "tilelang"],
    )
    inputs: List[TensorSpec] = Field(
        default_factory=list,
        description="Ordered list of input tensor descriptors",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional scalar keyword arguments forwarded to the kernel",
    )

    @field_validator("backend")
    @classmethod
    def _validate_backend(cls, v: str) -> str:
        allowed = {"triton", "tilelang", "cuda"}
        if v.lower() not in allowed:
            raise ValueError(f"backend '{v}' not recognised. Choose from {sorted(allowed)}")
        return v.lower()


# ---------------------------------------------------------------------------
# Kernel run response
# ---------------------------------------------------------------------------

class TensorOutput(BaseModel):
    shape: List[int]
    dtype: str
    data:  str = Field(..., description="Base64-encoded output tensor bytes")


class KernelRunResponse(BaseModel):
    kernel_name: str
    backend:     str
    elapsed_ms:  float = Field(..., description="Wall-clock kernel execution time in ms")
    device_info: str
    output:      Union[TensorOutput, List[TensorOutput], Any]


# ---------------------------------------------------------------------------
# Discovery responses
# ---------------------------------------------------------------------------

class KernelSummary(BaseModel):
    kernel_name: str
    backend:     str
    description: str


class KernelListResponse(BaseModel):
    count:   int
    kernels: List[KernelSummary]


class KernelDetailResponse(BaseModel):
    kernel_name:   str
    backend:       str
    description:   str
    input_schema:  Dict[str, Any]
    output_schema: Dict[str, Any]
    backends_available: List[str]


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error:   str
    detail:  Optional[str] = None
    kernel_name: Optional[str] = None
    backend:     Optional[str] = None
