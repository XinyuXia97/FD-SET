from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.scenic_library import index_scenic_programs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Scenic source programs and report native-runtime readiness")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "scenic")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "scenic_library_index.json")
    parser.add_argument("--copy-to", type=Path)
    args = parser.parse_args()

    payload = index_scenic_programs(args.source_dir, copy_to=args.copy_to)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "program_count": payload["program_count"],
                "native_execution_ready": payload["summary"]["native_execution_ready"],
                "missing_map_count": payload["summary"]["missing_map_count"],
                "scenic_runtime_available": payload["runtime"]["available"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
