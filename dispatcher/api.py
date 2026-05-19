"""
FastAPI service exposing the kernel dispatch layer.

Run locally:
    uvicorn dispatcher.api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .dispatcher import KernelDispatcher
from .schemas import (
    ErrorResponse,
    KernelDetailResponse,
    KernelListResponse,
    KernelRunRequest,
    KernelRunResponse,
)

logger = logging.getLogger(__name__)

_dispatcher: KernelDispatcher | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _dispatcher
    _dispatcher = KernelDispatcher(device="cuda")
    logger.info("Kernel dispatcher initialised")
    yield


app = FastAPI(
    title="Transformer Kernel Dispatch API",
    version="1.0.0",
    description=(
        "Public API to invoke Transformer kernels by kernel_name + backend, "
        "then dispatch execution to the GPU platform."
    ),
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "kernel-dispatcher"}


@app.get("/api/v1/kernels", response_model=KernelListResponse)
def list_kernels() -> KernelListResponse:
    kernels = _require_dispatcher().list_kernels()
    return KernelListResponse(count=len(kernels), kernels=kernels)


@app.get("/api/v1/kernels/{kernel_name}", response_model=KernelDetailResponse)
def kernel_detail(kernel_name: str, backend: str) -> KernelDetailResponse:
    dispatcher = _require_dispatcher()
    try:
        info = dispatcher.kernel_info(kernel_name, backend)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return KernelDetailResponse(
        **info,
        backends_available=dispatcher.list_backends(kernel_name),
    )


@app.post(
    "/api/v1/kernels/run",
    response_model=KernelRunResponse,
    responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
def run_kernel(req: KernelRunRequest):
    dispatcher = _require_dispatcher()

    try:
        result = dispatcher.run_from_spec(
            kernel_name=req.kernel_name,
            backend=req.backend,
            tensor_specs=[tensor.model_dump() for tensor in req.inputs],
            scalar_params=req.params,
        )
        return KernelRunResponse(**result.to_dict())
    except KeyError as exc:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                error="kernel_not_found",
                detail=str(exc),
                kernel_name=req.kernel_name,
                backend=req.backend,
            ).model_dump(),
        )
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="kernel_dispatch_failed",
                detail=str(exc),
                kernel_name=req.kernel_name,
                backend=req.backend,
            ).model_dump(),
        )


def _require_dispatcher() -> KernelDispatcher:
    if _dispatcher is None:
        raise RuntimeError("Kernel dispatcher not initialised")
    return _dispatcher
