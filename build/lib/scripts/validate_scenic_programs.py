from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.scenic_library import index_scenic_programs  # noqa: E402


def _find_carla_egg(pythonapi_root: Path) -> Path | None:
    candidates = sorted((pythonapi_root / "PythonAPI" / "carla" / "dist").glob("carla-*.egg"))
    if not candidates:
        return None
    current_tag = f"py{sys.version_info.major}.{sys.version_info.minor}"
    compatible = [path for path in candidates if current_tag in path.name or "py3.7" in path.name]
    return (compatible or candidates)[-1]


def _validate_one(path: Path, *, timeout_s: float, env: dict[str, str]) -> dict[str, object]:
    command = [
        sys.executable, "-c",
        "import sys\n"
        "from scenic.syntax.translator import scenarioFromFile\n"
        "scene, _ = scenarioFromFile(sys.argv[1]).generate(maxIterations=2000)\n"
        "if not scene.objects:\n"
        "    raise RuntimeError('Scenic generated no actors')\n"
        "print('SCENIC_SCENE_GENERATED')\n",
        str(path),
    ]
    try:
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout_s)
        output = f"{completed.stdout}\n{completed.stderr}"
        timed_out = False
        return_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = f"{exc.stdout or ''}\n{exc.stderr or ''}"
        timed_out = True
        return_code = 124

    if "ScenicSyntaxError" in output:
        status = "syntax_error"
    elif return_code == 0 and "SCENIC_SCENE_GENERATED" in output:
        status = "compiled"
    elif timed_out:
        status = "timeout"
    else:
        status = "runtime_error"
    return {
        "program_id": path.stem,
        "source_path": str(path.resolve()),
        "status": status,
        "timed_out": timed_out,
        "return_code": return_code,
        "diagnostic_tail": output[-1200:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and sample native Scenic programs without a CARLA server")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "scenic")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "scenic_validation.json")
    parser.add_argument("--timeout-s", type=float, default=20.0)
    parser.add_argument("--carla-pythonapi-root", type=Path, default=ROOT.parent / "carla")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHON_EGG_CACHE"] = str((ROOT / "artifacts" / "scenic-egg-cache").resolve())
    egg = _find_carla_egg(args.carla_pythonapi_root.expanduser())
    if egg is not None:
        env["PYTHONPATH"] = os.pathsep.join([str(egg), env.get("PYTHONPATH", "")])

    index = index_scenic_programs(args.source_dir)
    programs = [Path(row["source_path"]) for row in index["programs"]]
    results = [_validate_one(path, timeout_s=args.timeout_s, env=env) for path in programs]
    payload = {
        "schema_version": "fdset.scenic_validation.v1",
        "source_dir": str(args.source_dir.expanduser().resolve()),
        "runtime": index["runtime"],
        "program_count": len(results),
        "compiled_count": sum(1 for row in results if row["status"] == "compiled"),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "program_count": payload["program_count"],
                "compiled_count": payload["compiled_count"],
                "statuses": {row["program_id"]: row["status"] for row in results},
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
