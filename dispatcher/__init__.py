"""
Transformer kernel dispatch layer.

Public entry points:
- KernelDispatcher: programmatic dispatch API
- get_registry:     inspect registered kernels / backends
"""
from .dispatcher import KernelDispatcher
from .registry import get_registry

__all__ = ["KernelDispatcher", "get_registry"]
