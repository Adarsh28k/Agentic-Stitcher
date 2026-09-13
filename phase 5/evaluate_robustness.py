#!/usr/bin/env python3
"""Compare the existing Phase 3 and Phase 4 results across captures."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CAPTURE_NAMES = [
    "single_scan_with_ceiling",
    "single_scan_floor_only",
    "single_room",
]

CAPTURE_ROLES = {
    "single_scan_with_ceiling": "development",
    "single_scan_floor_only": "robustness",
    "single_room": "held_out_smoke_test",
}


def read_json(file_path):
    """Input: a JSON path; output: the decoded JSON object."""
    with file_path.open("r") as json_file:
        return json.load(json_file)


def index_capture_reports(report):
    """Input: a phase report; output: reports indexed by capture name."""
    indexed_reports = {}

    for capture_report in report.get("capture_reports", []):
        capture_name = capture_report.get("capture_name")
        if capture_name is not None:
            indexed_reports[capture_name] = capture_report

    return indexed_reports


def get_nested_value(source, keys, default=None):
    """Input: a dictionary, key sequence, and fallback; output: a nested value."""
    current_value = source

    for key in keys:
        if not isinstance(current_value, dict) or key not in current_value:
            return default
        current_value = current_value[key]

    return current_value


def calculate_bounds_diagonal(bounds):
    """Input: minimum and maximum bounds; output: their 3D diagonal length."""
    if not isinstance(bounds, dict):
        return None

    minimum = bounds.get("minimum")
    maximum = bounds.get("maximum")

    if minimum is None or maximum is None:
        return None

    return float(np.linalg.norm(np.asarray(maximum) - np.asarray(minimum)))


def output_files_exist(capture_report):
    """Input: a Phase 4 capture report; output: whether its referenced images exist."""
    output_paths = capture_report.get("outputs", {})
    expected_names = [
        "detected_planes_3d",
        "room_outline_2d",
        "floor_ceiling_candidates",
    ]
    existing_count = 0

    for name in expected_names:
        output_path = output_paths.get(name)
        if output_path is not None and Path(output_path).exists():
            existing_count += 1

    return existing_count == len(expected_names)


def make_capture_row(phase_three_report, phase_four_report):
    """Input: one Phase 3 and Phase 4 report; output: one comparison row."""
    phase_three_accumulation = phase_three_report.get("accumulation", {})
    phase_three_bounds = phase_three_report.get("bounds", {}).get("filtered", {})
    outline = phase_four_report.get("room_outline", {})
    outline_vertices = outline.get("vertices", [])

    return {
        "capture_name": phase_four_report.get("capture_name"),
        "capture_role": CAPTURE_ROLES.get(
            phase_four_report.get("capture_name"), "unknown"
        ),
        "plan_status": phase_four_report.get("plan_status"),
        "point_count": phase_four_report.get("point_count"),
        "phase_3_frame_count": phase_three_accumulation.get("frame_count"),
        "planes_detected": phase_four_report.get("planes_detected"),
        "wall_plane_count": phase_four_report.get("wall_plane_count"),
        "floor_detected": phase_four_report.get("floor_detected"),
        "ceiling_detected": phase_four_report.get("ceiling_detected"),
        "ceiling_height_estimate": phase_four_report.get(
            "ceiling_height_estimate"
        ),
        "outline_area": outline.get("area"),
        "outline_perimeter": outline.get("perimeter"),
        "outline_vertex_count": len(outline_vertices),
        "filtered_point_cloud_diagonal": calculate_bounds_diagonal(
            phase_three_bounds
        ),
        "phase_4_outputs_exist": output_files_exist(phase_four_report),
    }


def build_summary_table(phase_three_reports, phase_four_reports):
    """Input: indexed Phase 3 and Phase 4 reports; output: a pandas summary table."""
    rows = []

    for capture_name in CAPTURE_NAMES:
        phase_three_report = phase_three_reports.get(capture_name, {})
        phase_four_report = phase_four_reports.get(capture_name, {})

        if not phase_three_report or not phase_four_report:
            raise ValueError("Missing report for capture: " + capture_name)

        rows.append(make_capture_row(phase_three_report, phase_four_report))

    return pd.DataFrame(rows)


def save_metric_plot(summary_table, output_path):
    """Input: summary table and output path; output: a four-panel metric PNG."""
    metric_definitions = [
        ("planes_detected", "Detected planes"),
        ("wall_plane_count", "Candidate walls"),
        ("outline_area", "Outline area"),
        ("ceiling_height_estimate", "Ceiling height"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = ["#4c78a8", "#f58518", "#54a24b"]

    for axis, (column_name, title) in zip(axes.ravel(), metric_definitions):
        values = pd.to_numeric(summary_table[column_name], errors="coerce")
        axis.bar(summary_table["capture_name"], values, color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.3)

    figure.suptitle("Phase 5 cross-capture metric comparison")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_outline_plot(phase_four_reports, output_path):
    """Input: indexed Phase 4 reports and output path; output: an outline overlay PNG."""
    figure, axis = plt.subplots(figsize=(8, 8))
    colors = ["#4c78a8", "#f58518", "#54a24b"]

    for color, capture_name in zip(colors, CAPTURE_NAMES):
        report = phase_four_reports[capture_name]
        vertices = np.asarray(
            get_nested_value(report, ["room_outline", "vertices"], []),
            dtype=float,
        )

        if len(vertices) < 2:
            continue

        closed_vertices = np.vstack((vertices, vertices[0]))
        axis.plot(
            closed_vertices[:, 0],
            closed_vertices[:, 1],
            color=color,
            linewidth=2,
            label=capture_name,
        )

    axis.set_title("Provisional Phase 4 room outlines")
    axis.set_xlabel("Horizontal coordinate; provisional")
    axis.set_ylabel("Horizontal coordinate; provisional")
    axis.axis("equal")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_bounds_plot(phase_three_reports, output_path):
    """Input: indexed Phase 3 reports and output path; output: horizontal bounds PNG."""
    figure, axis = plt.subplots(figsize=(9, 7))
    colors = ["#4c78a8", "#f58518", "#54a24b"]

    for color, capture_name in zip(colors, CAPTURE_NAMES):
        report = phase_three_reports[capture_name]
        bounds = get_nested_value(report, ["bounds", "filtered"], {})
        minimum = np.asarray(bounds.get("minimum", [0, 0, 0]), dtype=float)
        maximum = np.asarray(bounds.get("maximum", [0, 0, 0]), dtype=float)
        width = maximum[0] - minimum[0]
        height = maximum[2] - minimum[2]
        rectangle = plt.Rectangle(
            (minimum[0], minimum[2]),
            width,
            height,
            fill=False,
            linewidth=2,
            color=color,
            label=capture_name,
        )
        axis.add_patch(rectangle)

    axis.set_title("Phase 3 provisional horizontal point-cloud bounds")
    axis.set_xlabel("World X; provisional")
    axis.set_ylabel("World Z; provisional")
    axis.grid(True, alpha=0.3)
    axis.axis("equal")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def calculate_comparison(summary_table):
    """Input: summary table; output: diagnostic consistency statistics and status."""
    area_values = pd.to_numeric(summary_table["outline_area"], errors="coerce").dropna()
    ceiling_values = pd.to_numeric(
        summary_table["ceiling_height_estimate"], errors="coerce"
    ).dropna()
    plan_statuses = summary_table["plan_status"].dropna().tolist()

    comparison = {
        "evaluation_status": "diagnostic_only",
        "formal_ground_truth_available": False,
        "all_phase_4_outputs_exist": bool(
            summary_table["phase_4_outputs_exist"].all()
        ),
        "plan_statuses": plan_statuses,
        "outline_area_minimum": float(area_values.min()) if len(area_values) else None,
        "outline_area_maximum": float(area_values.max()) if len(area_values) else None,
        "outline_area_range": float(area_values.max() - area_values.min())
        if len(area_values)
        else None,
        "ceiling_height_minimum": float(ceiling_values.min())
        if len(ceiling_values)
        else None,
        "ceiling_height_maximum": float(ceiling_values.max())
        if len(ceiling_values)
        else None,
        "ceiling_height_range": float(ceiling_values.max() - ceiling_values.min())
        if len(ceiling_values)
        else None,
        "interpretation": (
            "The captures are different trajectories and coverage patterns. "
            "Differences are reported as robustness evidence, not as accuracy error."
        ),
    }
    return comparison


def build_report(project_root, output_root):
    """Input: project and output folders; output: Phase 5 report and generated files."""
    phase_three_path = project_root / "phase 3" / "phase_3_report.json"
    phase_four_path = project_root / "phase 4" / "phase_4_report.json"
    phase_three_reports = index_capture_reports(read_json(phase_three_path))
    phase_four_reports = index_capture_reports(read_json(phase_four_path))
    summary_table = build_summary_table(phase_three_reports, phase_four_reports)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "capture_summary.csv"
    metric_plot_path = output_root / "plan_metrics_comparison.png"
    outline_plot_path = output_root / "room_outline_comparison.png"
    bounds_plot_path = output_root / "point_cloud_bounds_comparison.png"
    summary_table.to_csv(summary_path, index=False)
    save_metric_plot(summary_table, metric_plot_path)
    save_outline_plot(phase_four_reports, outline_plot_path)
    save_bounds_plot(phase_three_reports, bounds_plot_path)

    return {
        "report_name": "Agentic-Stitcher Phase 5 cross-capture robustness evaluation",
        "project_root": str(project_root),
        "input_reports": {
            "phase_3": str(phase_three_path),
            "phase_4": str(phase_four_path),
        },
        "capture_roles": CAPTURE_ROLES,
        "comparison": calculate_comparison(summary_table),
        "captures": summary_table.to_dict(orient="records"),
        "outputs": {
            "capture_summary_csv": str(summary_path),
            "plan_metrics_comparison": str(metric_plot_path),
            "room_outline_comparison": str(outline_plot_path),
            "point_cloud_bounds_comparison": str(bounds_plot_path),
        },
    }


def write_report(report, output_path):
    """Input: report dictionary and output path; output: formatted JSON on disk."""
    with output_path.open("w") as json_file:
        json.dump(report, json_file, indent=2)
        json_file.write("\n")


def main():
    """Input: no arguments or optional project path; output: Phase 5 comparison files."""
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1]).resolve()
    else:
        project_root = Path(__file__).resolve().parent.parent

    phase_five_folder = Path(__file__).resolve().parent
    output_root = phase_five_folder / "outputs"
    report_path = phase_five_folder / "phase_5_report.json"
    report = build_report(project_root, output_root)
    write_report(report, report_path)
    print("Phase 5 robustness evaluation completed.")
    print("Outputs:", output_root)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
