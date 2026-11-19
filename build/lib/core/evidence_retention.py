"""Bounded evidence retention for newly evaluated Bench2Drive routes."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .failure_extract import extract_failures_from_log

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_FRAME_DIRS = ("rgb", "rgb_front", "bev")


def compact_new_route_evidence(
    *,
    evidence_root: Path,
    route_id: str,
    checkpoint: Path,
    record: dict[str, Any] | None,
    failure_config: dict[str, Any],
    preexisting_names: set[str],
    max_images: int = 3,
) -> list[dict[str, Any]]:
    """Retain measured trajectories and selected failure images; never touch prior routes.

    The supervisor calls this only after the official per-route evaluator exited.
    A failed evaluation with no checkpoint retains the last camera/BEV frames.
    """
    if max_images < 1:
        raise ValueError("max_images must be positive")
    root = evidence_root.resolve()
    if not root.is_dir():
        return []
    prefix = f"RouteScenario_{route_id}_rep0_"
    if record is not None:
        save_name = str(record.get("save_name", ""))
        if not save_name.startswith(prefix):
            raise ValueError(f"Route {route_id} has an unexpected evidence name: {save_name!r}")
        names = [save_name]
    else:
        names = [path.name for path in root.iterdir() if path.name.startswith(prefix)]

    summaries: list[dict[str, Any]] = []
    for name in names:
        if name in preexisting_names:
            continue
        route_dir = root / name
        if not route_dir.is_dir() or route_dir.is_symlink() or route_dir.resolve().parent != root:
            raise RuntimeError(f"Route evidence is missing or escapes its root: {route_dir}")
        metric_info = route_dir / "metric_info.json"
        if record is not None and (metric_info.is_symlink() or not metric_info.is_file()):
            raise RuntimeError(f"Scored route {route_id} lacks official motion trajectory: {metric_info}")

        retained: set[Path] = set()
        if record is not None:
            cfg = dict(failure_config)
            template = str(cfg.get("frame_dir_template", "{log_dir}/evidence/{save_name}/rgb"))
            frame_subdir = Path(template.format(log_dir="", save_name=name)).name
            if frame_subdir not in _FRAME_DIRS:
                raise ValueError(f"Unsupported failure evidence camera directory: {frame_subdir}")
            cfg["frame_dir_template"] = str(root / "{save_name}" / frame_subdir)
            cfg["metric_info_root"] = str(root)
            cfg["__log_file__"] = str(checkpoint)
            events = extract_failures_from_log(checkpoint, cfg)
            for event in events:
                context = event.metadata.get("context_summary", {})
                if not isinstance(context, dict):
                    continue
                for filename in context.get("evidence_frames", [])[:max_images]:
                    image = Path(filename)
                    if image.is_symlink() or not image.is_file() or route_dir not in image.parents:
                        raise RuntimeError(f"Failure evidence frame is unavailable or outside route: {image}")
                    retained.add(image)
            if events and not retained:
                raise RuntimeError(f"Route {route_id} has failure feedback but no preserved image frames")
        else:
            for folder in _FRAME_DIRS:
                directory = route_dir / folder
                if directory.is_dir() and not directory.is_symlink():
                    images = sorted((p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES))
                    retained.update(images[-max_images:])

        retained_stems = {image.stem for image in retained}
        removed = 0
        for folder in (*_FRAME_DIRS, "meta"):
            directory = route_dir / folder
            if not directory.exists():
                continue
            if directory.is_symlink() or not directory.is_dir() or directory.resolve().parent != route_dir:
                raise RuntimeError(f"Evidence folder escapes the route directory: {directory}")
            for path in directory.iterdir():
                if path.is_symlink():
                    raise RuntimeError(f"Refusing to alter symlinked evidence: {path}")
                if not path.is_file():
                    continue
                if folder == "meta":
                    if path.suffix != ".json" or path.stem in retained_stems:
                        continue
                elif path.suffix.lower() not in _IMAGE_SUFFIXES or path in retained:
                    continue
                path.unlink()
                removed += 1
        summaries.append({
            "route_id": route_id,
            "save_name": name,
            "retained_failure_frames": sorted(str(path.relative_to(root)) for path in retained),
            "removed_frame_files": removed,
            "metric_info_retained": metric_info.is_file(),
        })
    return summaries


def append_retention_receipts(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
