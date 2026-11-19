"""Load the locally rebuilt Bench2DriveZoo extensions before importing mmcv."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def prepare_zoo_runtime(zoo_root: Path) -> None:
    import torch  # noqa: F401: resolve libtorch symbols before loading extensions
    from numba.core import errors as numba_errors

    sys.modules.setdefault("numba.errors", numba_errors)
    project_root = Path(__file__).resolve().parents[1]
    extension_root = project_root / "artifacts" / "mmcv_custom_build" / "lib" / "mmcv"
    for name, path in (
        ("mmcv._ext", extension_root / "_ext.cpython-38-x86_64-linux-gnu.so"),
        ("mmcv.ops.iou3d_det.iou3d_cuda", extension_root / "ops/iou3d_det/iou3d_cuda.cpython-38-x86_64-linux-gnu.so"),
        ("mmcv.ops.roiaware_pool3d.roiaware_pool3d_ext", extension_root / "ops/roiaware_pool3d/roiaware_pool3d_ext.cpython-38-x86_64-linux-gnu.so"),
    ):
        if name in sys.modules:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Build the matching Bench2DriveZoo CUDA extension first: {path}")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load the Bench2DriveZoo CUDA extension: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    root = str(zoo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    if str(zoo_root.resolve().parent) not in sys.path:
        sys.path.insert(0, str(zoo_root.resolve().parent))
