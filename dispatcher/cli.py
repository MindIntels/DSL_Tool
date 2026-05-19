"""
CLI entry point for the kernel dispatcher.

Examples
--------
List all kernels:
    python -m dispatcher.cli list

Inspect one kernel:
    python -m dispatcher.cli info --kernel normalization.rmsnorm --backend triton

Run a kernel (human-readable summary, default):
    python -m dispatcher.cli run \\
        --kernel normalization.rmsnorm \\
        --backend triton \\
        --input '{"shape":[32,4096],"dtype":"bfloat16","fill":"random"}' \\
        --input '{"shape":[4096],"dtype":"bfloat16","fill":"ones"}' \\
        --param eps=1e-6

Run and emit full base64 JSON (e.g. for piping to another tool):
    python -m dispatcher.cli run --raw ...
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

import torch

from .dispatcher import KernelDispatcher, DispatchResult, summarize_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer kernel dispatcher CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all registered kernels")

    p_info = sub.add_parser("info", help="Show one kernel's metadata")
    p_info.add_argument("--kernel", required=True, help="Kernel name, e.g. normalization.rmsnorm")
    p_info.add_argument("--backend", required=True, help="Backend, e.g. triton")

    p_run = sub.add_parser("run", help="Run a kernel")
    p_run.add_argument("--kernel", required=True, help="Kernel name, e.g. normalization.rmsnorm")
    p_run.add_argument("--backend", required=True, help="Backend, e.g. triton")
    p_run.add_argument(
        "--input",
        action="append",
        default=[],
        help="JSON tensor spec. Repeat for multiple tensor inputs.",
    )
    p_run.add_argument(
        "--param",
        action="append",
        default=[],
        help="Scalar param in key=value form. Repeat as needed.",
    )
    p_run.add_argument(
        "--raw",
        action="store_true",
        default=False,
        help="Emit full base64-encoded tensor data instead of a human-readable summary.",
    )

    args = parser.parse_args()
    dispatcher = KernelDispatcher(device="cuda")

    if args.command == "list":
        print(json.dumps(dispatcher.list_kernels(), indent=2, ensure_ascii=False))
        return

    if args.command == "info":
        info = dispatcher.kernel_info(args.kernel, args.backend)
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return

    if args.command == "run":
        tensor_specs = [json.loads(x) for x in args.input]
        params = _parse_params(args.param)
        result = dispatcher.run_from_spec(
            kernel_name=args.kernel,
            backend=args.backend,
            tensor_specs=tensor_specs,
            scalar_params=params,
        )
        if args.raw:
            # Full base64 JSON – suitable for piping to another tool
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            _print_summary(result)
        return


def _print_summary(result: DispatchResult) -> None:
    """Print a concise, human-readable result (no raw tensor bytes)."""
    print(f"kernel  : {result.kernel_name}")
    print(f"backend : {result.backend}")
    print(f"device  : {result.device_info}")
    print(f"elapsed : {result.elapsed_ms:.3f} ms")

    outputs = result.output
    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]

    if isinstance(outputs, (list, tuple)):
        for i, o in enumerate(outputs):
            if isinstance(o, torch.Tensor):
                s = summarize_tensor(o)
                label = "output" if len(outputs) == 1 else f"output[{i}]"
                print(
                    f"{label}  : shape={s['shape']} dtype={s['dtype']}"
                    f"  min={s['min']:.6g}  max={s['max']:.6g}"
                    f"  mean={s['mean']:.6g}  std={s['std']:.6g}"
                )
            else:
                print(f"output[{i}]: {o}")
    else:
        print(f"output  : {outputs}")


def _parse_params(raw_items: List[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for item in raw_items:
        if "=" not in item:
            raise ValueError(f"Invalid --param '{item}', expected key=value")
        key, value = item.split("=", 1)
        params[key] = _coerce_scalar(value)
    return params


def _coerce_scalar(value: str) -> Any:
    for fn in (int, float):
        try:
            return fn(value)
        except ValueError:
            pass
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value


if __name__ == "__main__":
    main()
