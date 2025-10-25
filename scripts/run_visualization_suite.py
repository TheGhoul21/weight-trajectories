#!/usr/bin/env python3
"""Run multiple visualization commands and stitch a markdown report."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch visualizations + markdown report")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON configuration file describing the visualization sweep",
    )
    parser.add_argument(
        "--report",
        help="Optional path where the markdown report will be written (overrides config)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use (defaults to current interpreter)",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    raw = path.read_text()
    data = json.loads(raw)
    if "experiments" not in data:
        raise ValueError("Configuration file must define an 'experiments' array")
    return data


def run_command(python_bin: str, args: List[str]) -> None:
    cmd = [python_bin, "-m", "src.visualize_trajectories"] + [str(item) for item in args]
    print("\n" + "=" * 72)
    print("Running:", " ".join(cmd))
    print("=" * 72)
    subprocess.run(cmd, check=True)


def write_report(report_path: Path, experiments: List[Dict[str, Any]]) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append(f"# Visualization Report ({timestamp})")

    for exp in experiments:
        name = exp.get("name", "Unnamed Experiment")
        description = exp.get("description", "")
        args = exp.get("args", [])
        command = "python -m src.visualize_trajectories " + " ".join(str(a) for a in args)
        lines.append("")
        lines.append(f"## {name}")
        if description:
            lines.append(description)
        lines.append("")
        lines.append(f"- Command: `{command}`")
        outputs = exp.get("images", [])
        if outputs:
            lines.append("- Outputs:")
            for img in outputs:
                rel = Path(img)
                status = "(missing)" if not rel.exists() else ""
                lines.append(f"  - `{rel}` {status}")
            lines.append("")
            for img in outputs:
                rel = Path(img)
                if rel.exists():
                    lines.append(f"![{rel.stem}]({rel.as_posix().replace('visualizations/', '')})")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"\nReport written to {report_path}")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    python_bin = args.python
    report_path = Path(args.report) if args.report else Path(config.get("report_path", "visualizations/visualization_report.md"))

    experiments: List[Dict[str, Any]] = config["experiments"]

    for exp in experiments:
        run_command(python_bin, exp.get("args", []))

    write_report(report_path, experiments)


if __name__ == "__main__":
    main()
