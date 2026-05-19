"""
Core Dispatcher
===============
Resolves (kernel_name, backend) -> KernelEntry, deserialises tensor inputs
from JSON-safe representations, dispatches to the GPU kernel, and returns
JSON-safe results with timing metadata.

Tensor wire format (used by the REST API and CLI):
  {
    "shape": [2, 4096],
    "dtype": "bfloat16",
    "data":  "<base64-encoded little-endian bytes>",   # optional
    "fill":  "zeros" | "ones" | "random" | <float>     # when data absent
  }
"""
from __future__ import annotations

import base64
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .registry import KernelEntry, get_registry

__all__ = [
    "KernelDispatcher",
    "DispatchResult",
    "deserialize_tensor",
    "serialize_tensor",
    "summarize_tensor",
]

__all__ = [
    "KernelDispatcher",
    "DispatchResult",
    "deserialize_tensor",
    "serialize_tensor",
    "summarize_tensor",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32":    torch.float32,
    "float16": torch.float16,
    "fp16":    torch.float16,
    "bfloat16":torch.bfloat16,
    "bf16":    torch.bfloat16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "fp8":     torch.float8_e4m3fn,
    "int8":    torch.int8,
    "int32":   torch.int32,
    "int64":   torch.int64,
}


def _resolve_dtype(name: str) -> torch.dtype:
    dt = _DTYPE_MAP.get(name.lower())
    if dt is None:
        raise ValueError(f"Unknown dtype '{name}'. Supported: {list(_DTYPE_MAP)}")
    return dt


# ---------------------------------------------------------------------------
# Tensor de-serialisation
# ---------------------------------------------------------------------------

def deserialize_tensor(spec: Dict[str, Any], device: str = "cuda") -> torch.Tensor:
    """
    Build a torch.Tensor from a JSON-safe dict:
      { "shape": [...], "dtype": "...", "data": "<b64>" }
    or
      { "shape": [...], "dtype": "...", "fill": "random"|"zeros"|"ones"|<float> }
    """
    shape = tuple(spec["shape"])
    dtype = _resolve_dtype(spec.get("dtype", "bfloat16"))

    if "data" in spec:
        import numpy as np
        raw = base64.b64decode(spec["data"])
        np_dtype = {
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.bfloat16: np.float32,   # numpy has no bf16; convert
            torch.int8:  np.int8,
            torch.int32: np.int32,
            torch.int64: np.int64,
        }.get(dtype, np.float32)
        arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
        t = torch.from_numpy(arr.copy())
        if dtype == torch.bfloat16:
            t = t.to(torch.bfloat16)
        return t.to(device)

    fill = spec.get("fill", "random")
    if fill == "zeros":
        return torch.zeros(shape, dtype=dtype, device=device)
    elif fill == "ones":
        return torch.ones(shape, dtype=dtype, device=device)
    elif fill == "random":
        if dtype in (torch.float32, torch.float16, torch.bfloat16):
            return torch.randn(shape, dtype=dtype, device=device)
        return torch.randint(0, 127, shape, dtype=dtype, device=device)
    else:
        return torch.full(shape, float(fill), dtype=dtype, device=device)


def serialize_tensor(t: torch.Tensor) -> Dict[str, Any]:
    """
    Serialize a torch.Tensor to a JSON-safe dict (base64 data).
    """
    import numpy as np
    cpu = t.detach().cpu()
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.to(torch.float32)
    arr = cpu.numpy()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "data":  base64.b64encode(arr.tobytes()).decode(),
    }


def summarize_tensor(t: torch.Tensor) -> Dict[str, Any]:
    """
    Return a human-readable summary of a tensor (no raw bytes).
    Used by the CLI to avoid flooding the terminal with base64 data.
    """
    f = t.detach().cpu().float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "numel": t.numel(),
        "min":   round(f.min().item(), 6),
        "max":   round(f.max().item(), 6),
        "mean":  round(f.mean().item(), 6),
        "std":   round(f.std().item(), 6),
    }


# ---------------------------------------------------------------------------
# Dispatch result
# ---------------------------------------------------------------------------

class DispatchResult:
    def __init__(
        self,
        output: Any,
        kernel_name: str,
        backend: str,
        elapsed_ms: float,
        device_info: str,
    ) -> None:
        self.output       = output         # raw torch.Tensor or list thereof
        self.kernel_name  = kernel_name
        self.backend      = backend
        self.elapsed_ms   = elapsed_ms
        self.device_info  = device_info

    def to_dict(self) -> dict:
        """JSON-serialisable representation."""
        if isinstance(self.output, torch.Tensor):
            out_serial = serialize_tensor(self.output)
        elif isinstance(self.output, (list, tuple)):
            out_serial = [
                serialize_tensor(o) if isinstance(o, torch.Tensor) else o
                for o in self.output
            ]
        else:
            out_serial = self.output

        return {
            "kernel_name": self.kernel_name,
            "backend":     self.backend,
            "elapsed_ms":  round(self.elapsed_ms, 4),
            "device_info": self.device_info,
            "output":      out_serial,
        }


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

class KernelDispatcher:
    """
    Public programmatic API.

    Usage::

        dispatcher = KernelDispatcher()

        # Call with real torch tensors
        x      = torch.randn(32, 4096, dtype=torch.bfloat16, device="cuda")
        weight = torch.ones(4096, dtype=torch.bfloat16, device="cuda")
        result = dispatcher.run(
            kernel_name="normalization.rmsnorm",
            backend="triton",
            args=[x, weight],
            kwargs={"eps": 1e-6},
        )
        print(result.output.shape)   # [32, 4096]

        # Call from JSON-safe spec (e.g. from REST API or CLI)
        result = dispatcher.run_from_spec(
            kernel_name="normalization.rmsnorm",
            backend="triton",
            tensor_specs=[
                {"shape": [32, 4096], "dtype": "bfloat16", "fill": "random"},
                {"shape": [4096],     "dtype": "bfloat16", "fill": "ones"},
            ],
            scalar_params={"eps": 1e-6},
        )
        print(result.to_dict())
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device   = device if torch.cuda.is_available() else "cpu"
        self._registry = get_registry(auto_load=True)
        if self.device == "cpu":
            logger.warning("CUDA not available – running on CPU (results may differ)")

    # ------------------------------------------------------------------ #
    #  Core dispatch                                                       #
    # ------------------------------------------------------------------ #

    def run(
        self,
        kernel_name: str,
        backend: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> DispatchResult:
        """
        Dispatch (kernel_name, backend) with positional *args* and **kwargs*.
        Inputs must be proper torch.Tensors already on the target device.
        """
        entry = self._resolve(kernel_name, backend)
        args   = args   or []
        kwargs = kwargs or {}

        device_info = self._device_info()
        start = time.perf_counter()
        try:
            output = entry.fn(*args, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Kernel '{kernel_name}' [{backend}] raised: {exc}"
            ) from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return DispatchResult(
            output=output,
            kernel_name=entry.kernel_name,
            backend=entry.backend,
            elapsed_ms=elapsed_ms,
            device_info=device_info,
        )

    def run_from_spec(
        self,
        kernel_name: str,
        backend: str,
        tensor_specs: Optional[List[Dict[str, Any]]] = None,
        scalar_params: Optional[Dict[str, Any]] = None,
    ) -> DispatchResult:
        """
        Dispatch from JSON-safe tensor specifications.  Tensors are constructed
        on-device before being forwarded to the kernel.

        Parameters
        ----------
        tensor_specs:
            Ordered list of tensor descriptor dicts (see module docstring).
        scalar_params:
            Extra keyword arguments forwarded verbatim to the kernel.
        """
        tensors = [
            deserialize_tensor(spec, device=self.device)
            for spec in (tensor_specs or [])
        ]
        return self.run(
            kernel_name=kernel_name,
            backend=backend,
            args=tensors,
            kwargs=scalar_params or {},
        )

    # ------------------------------------------------------------------ #
    #  Discovery                                                           #
    # ------------------------------------------------------------------ #

    def list_kernels(self) -> List[dict]:
        """Return all registered (kernel_name, backend) pairs with descriptions."""
        return self._registry.list_kernels()

    def list_backends(self, kernel_name: str) -> List[str]:
        """Return available backends for a specific kernel."""
        return self._registry.list_backends_for(kernel_name)

    def kernel_info(self, kernel_name: str, backend: str) -> dict:
        """Return detailed info for one (kernel, backend) pair."""
        entry = self._resolve(kernel_name, backend)
        return {
            "kernel_name":    entry.kernel_name,
            "backend":        entry.backend,
            "description":    entry.description,
            "input_schema":   entry.input_schema,
            "output_schema":  entry.output_schema,
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _resolve(self, kernel_name: str, backend: str) -> KernelEntry:
        entry = self._registry.get(kernel_name, backend)
        if entry is None:
            available = self._registry.list_backends_for(kernel_name)
            if available:
                raise KeyError(
                    f"Backend '{backend}' not available for kernel '{kernel_name}'. "
                    f"Available backends: {available}"
                )
            raise KeyError(
                f"Unknown kernel '{kernel_name}'. "
                f"Run list_kernels() to see all registered kernels."
            )
        return entry

    def _device_info(self) -> str:
        if not torch.cuda.is_available():
            return "cpu"
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_mb = torch.cuda.get_device_properties(idx).total_memory // (1024 ** 2)
        return f"{name} ({total_mb} MB) [device:{idx}]"
