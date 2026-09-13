#!/usr/bin/env python3
"""Run the Agentic-Stitcher phases in order and write one pipeline report."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


CAPTURE_NAMES = [
    "single_room",
    "single_scan_floor_only",
    "single_scan_with_ceiling",
]


PHASE_COMMANDS = [
    ("phase_0_validation", ["phase 0/validate_dataset.py"]),
    ("phase_1_loading", ["phase 1/run_phase_1.py"]),
    ("phase_2_diagnostics", ["phase 2/generate_diagnostics.py"]),
    ("phase_3_point_cloud", ["phase 3/build_point_cloud.py"]),
    ("phase_4_room_plan", ["phase 4/extract_room_plan.py"]),
    ("phase_5_robustness", ["phase 5/evaluate_robustness.py"]),
    ("phase_6_pose_graph", ["phase 6/build_pose_graph.py"]),
    ("phase_6b_optimization", ["phase 6/optimize_pose_graph.py"]),
]


def parse_arguments():
    """Input: command-line arguments; output: parsed project path and options."""
    parser = argparse.ArgumentParser(
        description="Run the Agentic-Stitcher pipeline from Phase 0 through Phase 6B."
    )
    parser.add_argument(
        "project_root",
        nargs="?",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Folder containing the capture folders and phase folders.",
    )
    return parser.parse_args()


def check_project_structure(project_root):
    """Input: project path; output: a list of missing required paths."""
    required_paths = []

    for capture_name in CAPTURE_NAMES:
        required_paths.extend(
            [
                project_root / capture_name,
                project_root / capture_name / "depth",
                project_root / capture_name / "confidence",
                project_root / capture_name / "odometry.csv",
                project_root / capture_name / "camera_matrix.csv",
            ]
        )

    for _, command in PHASE_COMMANDS:
        required_paths.append(project_root / command[0])

    missing_paths = []

    for required_path in required_paths:
        if not required_path.exists():
            missing_paths.append(str(required_path))

    return missing_paths


def build_environment():
    """Input: no arguments; output: subprocess environment with a writable plot cache."""
    environment = os.environ.copy()
    cache_folder = tempfile.mkdtemp(prefix="agentic_stitcher_mpl_")
    environment["MPLCONFIGDIR"] = cache_folder
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["VECLIB_MAXIMUM_THREADS"] = "1"
    return environment


def run_phase(project_root, phase_name, command, environment):
    """Input: project, phase name, command, and environment; output: execution record."""
    full_command = [sys.executable] + command + [str(project_root)]
    start_time = time.time()
    print("\nRunning " + phase_name + "...")
    completed_process = subprocess.run(
        full_command,
        cwd=project_root,
        env=environment,
        text=True,
    )
    duration = time.time() - start_time
    status = "completed" if completed_process.returncode == 0 else "failed"
    print(phase_name + " status: " + status)

    return {
        "phase": phase_name,
        "command": full_command,
        "status": status,
        "return_code": int(completed_process.returncode),
        "duration_seconds": float(duration),
    }


def get_report_paths(project_root):
    """Input: project path; output: expected phase report paths."""
    report_paths = []

    for phase_folder in [
        "phase 0",
        "phase 1",
        "phase 2",
        "phase 3",
        "phase 4",
        "phase 5",
        "phase 6",
    ]:
        report_paths.extend(str(path) for path in (project_root / phase_folder).glob("*.json"))

    return sorted(report_paths)


def write_pipeline_report(project_root, output_folder, phase_records, overall_status):
    """Input: run details and output folder; output: one combined pipeline JSON report."""
    output_folder.mkdir(parents=True, exist_ok=True)
    report_path = output_folder / "pipeline_report.json"
    report = {
        "pipeline_name": "Agentic-Stitcher offline pipeline",
        "project_root": str(project_root),
        "overall_status": overall_status,
        "phase_records": phase_records,
        "phase_reports": get_report_paths(project_root),
        "limitations": [
            "Geometric outputs remain provisional until sensor conventions and physical measurements are confirmed.",
            "Phase 6 optimization is diagnostic and may not improve the raw odometry cloud.",
            "Damage detection and RGB-only processing are not included in this pipeline.",
        ],
    }

    with report_path.open("w") as json_file:
        json.dump(report, json_file, indent=2)
        json_file.write("\n")

    return report_path


def main():
    """Input: optional project path; output: phase reports and one pipeline report."""
    arguments = parse_arguments()
    project_root = arguments.project_root.resolve()

    if not project_root.is_dir():
        raise FileNotFoundError("Project folder does not exist: " + str(project_root))

    missing_paths = check_project_structure(project_root)

    if missing_paths:
        print("The project is missing required paths:")
        for missing_path in missing_paths:
            print("- " + missing_path)
        return 1

    environment = build_environment()
    phase_records = []
    overall_status = "completed"

    for phase_name, command in PHASE_COMMANDS:
        phase_record = run_phase(project_root, phase_name, command, environment)
        phase_records.append(phase_record)

        if phase_record["status"] == "failed":
            overall_status = "failed"
            break

    report_path = write_pipeline_report(
        project_root,
        project_root / "pipeline_outputs",
        phase_records,
        overall_status,
    )
    print("\nPipeline status:", overall_status)
    print("Pipeline report:", report_path)

    return 0 if overall_status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
