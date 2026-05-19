"""
Kernel Registry
===============
Central registry mapping (kernel_name, backend) -> callable.

Kernel naming convention:
    <category>.<function>
    e.g. "normalization.rmsnorm", "gemm.bf16_gemm", "activations.swiglu"

Backend values: "triton" | "tilelang" | "cuda"
"""
from __future__ import annotations

import importlib.util
import os
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------

@dataclass
class KernelEntry:
    """Metadata + callable for one (kernel, backend) pair."""
    kernel_name: str          # e.g. "normalization.rmsnorm"
    backend: str              # "triton" | "tilelang" | "cuda"
    fn: Callable              # the Python callable to invoke
    description: str = ""
    input_schema: dict = field(default_factory=dict)   # optional param docs
    output_schema: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry singleton
# ---------------------------------------------------------------------------

class KernelRegistry:
    """
    Thread-safe registry.  Back-ends are loaded lazily on first access so that
    missing optional dependencies (TileLang, etc.) don't break import time.
    """

    def __init__(self) -> None:
        self._store: Dict[Tuple[str, str], KernelEntry] = {}

    # ------------------------------------------------------------------ #
    #  Registration helpers                                                #
    # ------------------------------------------------------------------ #

    def register(
        self,
        kernel_name: str,
        backend: str,
        fn: Callable,
        description: str = "",
        input_schema: Optional[dict] = None,
        output_schema: Optional[dict] = None,
    ) -> None:
        key = (kernel_name.lower(), backend.lower())
        self._store[key] = KernelEntry(
            kernel_name=kernel_name.lower(),
            backend=backend.lower(),
            fn=fn,
            description=description,
            input_schema=input_schema or {},
            output_schema=output_schema or {},
        )
        logger.debug("Registered kernel %s [%s]", kernel_name, backend)

    def get(self, kernel_name: str, backend: str) -> Optional[KernelEntry]:
        return self._store.get((kernel_name.lower(), backend.lower()))

    def list_kernels(self) -> list[dict]:
        """Return sorted list of {kernel_name, backend, description} dicts."""
        return sorted(
            [
                {
                    "kernel_name": e.kernel_name,
                    "backend": e.backend,
                    "description": e.description,
                }
                for e in self._store.values()
            ],
            key=lambda x: (x["kernel_name"], x["backend"]),
        )

    def list_backends_for(self, kernel_name: str) -> list[str]:
        kn = kernel_name.lower()
        return sorted({b for (k, b) in self._store if k == kn})

    # ------------------------------------------------------------------ #
    #  Lazy loader                                                         #
    # ------------------------------------------------------------------ #

    def _try_load_attr_from_file(self, file_path: str, attr: str) -> Optional[Callable]:
        """Load a module from an absolute file path and return one attribute."""
        try:
            module_name = f"_kernel_dispatch_{abs(hash((file_path, attr)))}"
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, attr, None)
        except Exception as exc:
            logger.debug("Could not load %s:%s: %s", file_path, attr, exc)
            return None

    def load_all(self, transformer_root: Optional[str] = None) -> None:
        """
        Discover and register all kernels from triton/ and tilelang/ sub-packages.
        Call this once at startup.
        """
        if transformer_root is None:
            transformer_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self._register_triton(transformer_root)
        self._register_tilelang(transformer_root)

    # ------------------------------------------------------------------ #
    #  Triton kernels                                                      #
    # ------------------------------------------------------------------ #

    def _register_triton(self, root: str) -> None:
        triton_root = os.path.join(root, "triton")

        # --- GEMM ---
        file_path = os.path.join(triton_root, "gemm", "gemm_kernels.py")
        for name, fn_name, desc in [
            ("gemm.bf16_gemm",   "bf16_gemm",    "BF16 WMMA GEMM (A[M,K] @ B[K,N])"),
            ("gemm.fp8_gemm",    "fp8_gemm",     "FP8 E4M3 GEMM with per-tensor scaling"),
            ("gemm.grouped_gemm","grouped_gemm", "Grouped / batched GEMM for MoE / LoRA"),
        ]:
            fn = self._try_load_attr_from_file(file_path, fn_name)
            if fn:
                self.register(name, "triton", fn, desc,
                    input_schema=_gemm_input_schema(name))

        # --- Normalization ---
        file_path = os.path.join(triton_root, "normalization", "norm_kernels.py")
        for name, fn_name, desc in [
            ("normalization.rmsnorm",      "rmsnorm",      "RMSNorm: y = w·x/√(mean(x²)+ε)"),
            ("normalization.layernorm",    "layernorm",    "LayerNorm: y = w·(x-μ)/√(σ²+ε)+b"),
            ("normalization.gemma_rmsnorm","gemma_rmsnorm","Gemma-style RMSNorm: y=(1+w)·x/rms"),
        ]:
            fn = self._try_load_attr_from_file(file_path, fn_name)
            if fn:
                self.register(name, "triton", fn, desc,
                    input_schema=_norm_input_schema(name))

        # --- Activations ---
        file_path = os.path.join(triton_root, "activations", "activation_kernels.py")
        for name, fn_name, desc in [
            ("activations.swiglu", "swiglu", "SwiGLU: SiLU(gate)×up, input=[B,2H]"),
            ("activations.geglu",  "geglu",  "GeGLU: GELU(gate)×up, input=[B,2H]"),
        ]:
            fn = self._try_load_attr_from_file(file_path, fn_name)
            if fn:
                self.register(name, "triton", fn, desc,
                    input_schema={"input": "Tensor[batch, 2*hidden] bf16"})

        # --- RoPE ---
        file_path = os.path.join(triton_root, "rope", "rope_kernel.py")
        for name, fn_name, desc in [
            ("rope.apply_rope",        "apply_rope",        "Standard RoPE (LLaMA 1/2, base=10000)"),
            ("rope.apply_rope_llama31","apply_rope_llama31","LLaMA 3.1 extended-context RoPE"),
        ]:
            fn = self._try_load_attr_from_file(file_path, fn_name)
            if fn:
                self.register(name, "triton", fn, desc,
                    input_schema={"x": "Tensor[batch, seq, heads, dim] bf16/fp16"})

        # --- Qwen3NeXt ---
        file_path = os.path.join(triton_root, "qwen3next", "qwen3next_kernels.py")
        for name, fn_name, desc in [
            ("qwen3next.zero_centered_rmsnorm",  "zero_centered_rmsnorm",
             "Zero-Centered RMSNorm: y=(1+w)·x/rms, w init to 0"),
            ("qwen3next.gated_softmax_attention", "gated_softmax_attention",
             "Gated Softmax Attention: O=σ(g)·softmax(QKᵀ/√d)·V"),
            ("qwen3next.gated_delta_rule",        "gated_delta_rule",
             "Gated Delta Rule: linear attention S_t=S_{t-1}+β(v-Sk)⊗k"),
        ]:
            fn = self._try_load_attr_from_file(file_path, fn_name)
            if fn:
                self.register(name, "triton", fn, desc)

    # ------------------------------------------------------------------ #
    #  TileLang kernels                                                    #
    # ------------------------------------------------------------------ #

    def _register_tilelang(self, root: str) -> None:
        tilelang_root = os.path.join(root, "tilelang")

        # --- GEMM ---
        file_path = os.path.join(tilelang_root, "gemm", "gemm_kernels.py")
        for name, fn_name, desc in [
            ("gemm.bf16_gemm",   "bf16_gemm_tilelang",   "BF16 GEMM (TileLang/TVM TIR)"),
            ("gemm.grouped_gemm","grouped_gemm_tiled",    "Grouped GEMM (TileLang tiled)"),
        ]:
            fn = self._try_load_attr_from_file(file_path, fn_name)
            if fn:
                self.register(name, "tilelang", fn, desc,
                    input_schema=_gemm_input_schema(name))

        # --- Normalization ---
        file_path = os.path.join(tilelang_root, "normalization", "norm_kernels.py")
        for name, fn_name, desc in [
            ("normalization.rmsnorm","rmsnorm_tilelang","RMSNorm (TileLang/TVM TIR tile)"),
        ]:
            fn = self._try_load_attr_from_file(file_path, fn_name)
            if fn:
                self.register(name, "tilelang", fn, desc,
                    input_schema=_norm_input_schema(name))

        # --- Activations ---
        file_path = os.path.join(tilelang_root, "activations", "activation_kernels.py")
        for name, fn_name, desc in [
            ("activations.swiglu","swiglu_tilelang","SwiGLU (TileLang tiled)"),
        ]:
            fn = self._try_load_attr_from_file(file_path, fn_name)
            if fn:
                self.register(name, "tilelang", fn, desc,
                    input_schema={"input": "Tensor[batch, 2*hidden] bf16"})

        # --- RoPE ---
        file_path = os.path.join(tilelang_root, "rope", "rope_kernel.py")
        rope_cls = self._try_load_attr_from_file(file_path, "TiledRoPE")
        if rope_cls:
            self.register("rope.apply_rope", "tilelang", _wrap_tilelang_rope(rope_cls),
                "RoPE (TileLang tiled)",
                input_schema={"x": "Tensor[batch, seq, heads, dim] bf16/fp16"})
        rope31_cls = self._try_load_attr_from_file(file_path, "TiledRoPELLaMA31")
        if rope31_cls:
            self.register("rope.apply_rope_llama31", "tilelang", _wrap_tilelang_rope(rope31_cls),
                "LLaMA 3.1 RoPE (TileLang tiled)",
                input_schema={"x": "Tensor[batch, seq, heads, dim] bf16/fp16"})

        # --- Qwen3NeXt ---
        file_path = os.path.join(tilelang_root, "qwen3next", "qwen3next_kernels.py")
        zero_centered_cls = self._try_load_attr_from_file(file_path, "TiledZeroCenteredRMSNorm")
        if zero_centered_cls:
            self.register(
                "qwen3next.zero_centered_rmsnorm",
                "tilelang",
                _wrap_tilelang_norm(zero_centered_cls),
                "Zero-Centered RMSNorm (TileLang)",
            )
        gated_softmax_cls = self._try_load_attr_from_file(file_path, "TiledGatedSoftmaxAttention")
        if gated_softmax_cls:
            self.register(
                "qwen3next.gated_softmax_attention",
                "tilelang",
                _wrap_tilelang_gated_softmax(gated_softmax_cls),
                "Gated Softmax Attention (TileLang)",
            )
        gated_delta_cls = self._try_load_attr_from_file(file_path, "TiledGatedDeltaRule")
        if gated_delta_cls:
            self.register(
                "qwen3next.gated_delta_rule",
                "tilelang",
                _wrap_tilelang_delta_rule(gated_delta_cls),
                "Gated Delta Rule (TileLang)",
            )


# ---------------------------------------------------------------------------
# Shared input schema helpers (documentation only)
# ---------------------------------------------------------------------------

def _gemm_input_schema(kernel_name: str) -> dict:
    base = {
        "A": "Tensor[M, K] bf16",
        "B": "Tensor[K, N] bf16",
    }
    if "fp8" in kernel_name:
        base.update({"scale_a": "float", "scale_b": "float"})
    if "grouped" in kernel_name:
        base["num_groups"] = "int"
    return base


def _norm_input_schema(kernel_name: str) -> dict:
    base = {"x": "Tensor[batch, hidden] bf16/fp32", "weight": "Tensor[hidden]"}
    if "layernorm" in kernel_name:
        base["bias"] = "Tensor[hidden]"
    base["eps"] = "float (default 1e-6)"
    return base


def _wrap_tilelang_rope(rope_cls: type) -> Callable:
    def _wrapped(x, **kwargs):
        rope = rope_cls(head_dim=x.shape[-1], **kwargs)
        return rope.forward(x)
    return _wrapped


def _wrap_tilelang_norm(norm_cls: type) -> Callable:
    def _wrapped(x, weight, **kwargs):
        norm = norm_cls(hidden_size=x.shape[-1], **kwargs)
        return norm.forward(x, weight)
    return _wrapped


def _wrap_tilelang_gated_softmax(attn_cls: type) -> Callable:
    def _wrapped(Q, K, V, gate, **kwargs):
        attn = attn_cls(**kwargs)
        return attn.forward(Q, K, V, gate)
    return _wrapped


def _wrap_tilelang_delta_rule(delta_cls: type) -> Callable:
    def _wrapped(Q, K, V, beta, **kwargs):
        delta = delta_cls(head_dim=Q.shape[-1], **kwargs)
        return delta.forward(Q, K, V, beta)
    return _wrapped


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[KernelRegistry] = None


def get_registry(auto_load: bool = True) -> KernelRegistry:
    """Return (and optionally populate) the global registry singleton."""
    global _registry
    if _registry is None:
        _registry = KernelRegistry()
        if auto_load:
            _registry.load_all()
    return _registry
