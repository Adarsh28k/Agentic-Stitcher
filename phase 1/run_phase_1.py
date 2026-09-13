#!/usr/bin/env python3
"""Run the Phase 1 loader and validation checks."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from capture_loader import validate_all_captures


CAPTURE_NAMES = [
    "single_room",
    "single_scan_floor_only",
    "single_scan_with_ceiling",
]


def get_project_root(script_path):
    """Input: this script path; output: the Agentic-Stitcher project folder."""
    script_folder = script_path.resolve().parent
    project_root = script_folder.parent
    return project_root


def build_report(project_root):
    """Input: project folder; output: a JSON-friendly Phase 1 report dictionary."""
    capture_reports = validate_all_captures(project_root, CAPTURE_NAMES)

    report = {
        "report_name": "Agentic-Stitcher Phase 1 normalized capture validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "capture_reports": capture_reports,
    }

    return report


def write_report(report, output_path):
    """Input: report dictionary and output path; output: formatted JSON written to disk."""
    with output_path.open("w") as json_file:
        json.dump(report, json_file, indent=2)
        json_file.write("\n")


def main():
    """Input: optional project path argument; output: Phase 1 report and status message."""
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1]).resolve()
    else:
        project_root = get_project_root(Path(__file__))

    output_path = Path(__file__).resolve().parent / "phase_1_report.json"
    report = build_report(project_root)
    write_report(report, output_path)

    print("Phase 1 validation completed.")
    print("Project:", project_root)
    print("Report:", output_path)


if __name__ == "__main__":
    main()
