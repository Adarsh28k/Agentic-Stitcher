#!/usr/bin/env python3
"""Generate readable diagnostic images for the supplied captures."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


PHASE_ONE_FOLDER = Path(__file__).resolve().parent.parent / "phase 1"
sys.path.insert(0, str(PHASE_ONE_FOLDER))

from capture_loader import get_frame, load_capture


CAPTURE_NAMES = [
    "single_room",
    "single_scan_floor_only",
    "single_scan_with_ceiling",
]


def choose_sample_frame_ids(capture):
    """Input: a loaded capture; output: first, middle, and final frame IDs."""
    frame_ids = sorted(capture.depth_files.keys())
    selected_ids = []

    if len(frame_ids) == 0:
        return selected_ids

    selected_ids.append(frame_ids[0])
    selected_ids.append(frame_ids[len(frame_ids) // 2])
    selected_ids.append(frame_ids[-1])
    return selected_ids


def get_depth_display_values(depth):
    """Input: a raw depth array; output: clipped values and the display range."""
    valid_values = depth[depth > 0]

    if len(valid_values) == 0:
        empty_values = np.zeros(depth.shape, dtype=float)
        return empty_values, 0.0, 1.0

    lower_value = float(np.percentile(valid_values, 2))
    upper_value = float(np.percentile(valid_values, 98))

    if upper_value <= lower_value:
        upper_value = lower_value + 1.0

    display_values = depth.astype(float)
    display_values = np.clip(display_values, lower_value, upper_value)
    return display_values, lower_value, upper_value


def save_depth_image(frame, output_path):
    """Input: one loaded frame and output path; output: a labeled colorized depth PNG."""
    depth = frame["depth"]
    display_values, lower_value, upper_value = get_depth_display_values(depth)
    masked_values = np.ma.masked_where(depth <= 0, display_values)

    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(masked_values, cmap="viridis")
    axis.set_title("Depth map — frame " + str(frame["frame_id"]))
    axis.set_xlabel("Image column")
    axis.set_ylabel("Image row")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Raw depth value; unit not confirmed")
    axis.text(
        0.01,
        -0.16,
        "Dark/blue is closer within this image; yellow is farther. Black pixels are invalid.",
        transform=axis.transAxes,
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    return {
        "frame_id": frame["frame_id"],
        "minimum_valid_value": float(np.min(depth[depth > 0])) if np.any(depth > 0) else None,
        "maximum_valid_value": float(np.max(depth[depth > 0])) if np.any(depth > 0) else None,
        "display_lower_percentile": lower_value,
        "display_upper_percentile": upper_value,
        "invalid_pixel_count": int(np.sum(depth <= 0)),
        "output_file": str(output_path),
    }


def save_confidence_image(frame, output_path):
    """Input: one loaded frame and output path; output: a labeled confidence PNG."""
    confidence = frame["confidence"]
    confidence_colors = ListedColormap(["black", "#f4c542", "#2ca25f"])

    figure, axis = plt.subplots(figsize=(8, 5))
    image = axis.imshow(confidence, cmap=confidence_colors, vmin=0, vmax=2)
    axis.set_title("Confidence map — frame " + str(frame["frame_id"]))
    axis.set_xlabel("Image column")
    axis.set_ylabel("Image row")
    colorbar = figure.colorbar(image, ax=axis, ticks=[0, 1, 2])
    colorbar.ax.set_yticklabels(["0: invalid", "1: medium", "2: strongest"])
    axis.text(
        0.01,
        -0.16,
        "This is a quality mask, not a brightness image.",
        transform=axis.transAxes,
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    unique_values, counts = np.unique(confidence, return_counts=True)
    value_counts = {}

    for value, count in zip(unique_values, counts):
        value_counts[str(int(value))] = int(count)

    return {
        "frame_id": frame["frame_id"],
        "value_counts": value_counts,
        "output_file": str(output_path),
    }


def save_depth_confidence_pair(frame, output_path):
    """Input: one loaded frame and output path; output: a side-by-side diagnostic PNG."""
    depth = frame["depth"]
    confidence = frame["confidence"]
    display_values, _, _ = get_depth_display_values(depth)
    masked_values = np.ma.masked_where(depth <= 0, display_values)
    confidence_colors = ListedColormap(["black", "#f4c542", "#2ca25f"])

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(masked_values, cmap="viridis")
    axes[0].set_title("Depth")
    axes[0].set_xlabel("Image column")
    axes[0].set_ylabel("Image row")
    axes[1].imshow(confidence, cmap=confidence_colors, vmin=0, vmax=2)
    axes[1].set_title("Confidence")
    axes[1].set_xlabel("Image column")
    axes[1].set_ylabel("Image row")
    figure.suptitle("Sensor maps — frame " + str(frame["frame_id"]))
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    return str(output_path)


def save_trajectory_plot(capture, output_path):
    """Input: a loaded capture and output path; output: raw odometry trajectory PNG."""
    odometry = capture.odometry
    x_values = odometry["x"].to_numpy(dtype=float)
    y_values = odometry["y"].to_numpy(dtype=float)
    z_values = odometry["z"].to_numpy(dtype=float)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(x_values, y_values, color="#1565c0", linewidth=1)
    axes[0].scatter(x_values[0], y_values[0], color="green", label="start")
    axes[0].scatter(x_values[-1], y_values[-1], color="red", label="end")
    axes[0].set_title("Raw odometry: x versus y")
    axes[0].set_xlabel("Odometry x; unit unknown")
    axes[0].set_ylabel("Odometry y; unit unknown")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(x_values, z_values, color="#6a1b9a", linewidth=1)
    axes[1].scatter(x_values[0], z_values[0], color="green", label="start")
    axes[1].scatter(x_values[-1], z_values[-1], color="red", label="end")
    axes[1].set_title("Raw odometry: x versus z")
    axes[1].set_xlabel("Odometry x; unit unknown")
    axes[1].set_ylabel("Odometry z; unit unknown")
    axes[1].legend()
    axes[1].grid(True)

    figure.suptitle(capture.name + " — trajectory without coordinate interpretation")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

    return {
        "output_file": str(output_path),
        "point_count": len(x_values),
        "start": [float(x_values[0]), float(y_values[0]), float(z_values[0])],
        "end": [float(x_values[-1]), float(y_values[-1]), float(z_values[-1])],
    }


def generate_capture_diagnostics(project_root, capture_name, output_root):
    """Input: project folder, capture name, and output folder; output: diagnostic report."""
    capture_path = project_root / capture_name
    capture = load_capture(capture_path)
    capture_output = output_root / capture_name
    capture_output.mkdir(parents=True, exist_ok=True)
    sample_ids = choose_sample_frame_ids(capture)
    depth_reports = []
    confidence_reports = []
    pair_files = []

    for frame_id in sample_ids:
        frame = get_frame(capture, frame_id)
        depth_path = capture_output / ("frame_" + str(frame_id).zfill(6) + "_depth.png")
        confidence_path = capture_output / ("frame_" + str(frame_id).zfill(6) + "_confidence.png")
        pair_path = capture_output / ("frame_" + str(frame_id).zfill(6) + "_sensor_maps.png")
        depth_reports.append(save_depth_image(frame, depth_path))
        confidence_reports.append(save_confidence_image(frame, confidence_path))
        pair_files.append(save_depth_confidence_pair(frame, pair_path))

    trajectory_path = capture_output / "trajectory.png"
    trajectory_report = save_trajectory_plot(capture, trajectory_path)

    report = {
        "capture_name": capture_name,
        "sample_frame_ids": sample_ids,
        "depth_images": depth_reports,
        "confidence_images": confidence_reports,
        "sensor_map_pairs": pair_files,
        "trajectory": trajectory_report,
        "rgb_preview": {
            "generated": False,
            "reason": "RGB frame extraction is not enabled yet; the current environment has no video decoder available to this script.",
        },
    }

    return report


def build_report(project_root, output_root):
    """Input: project and output folders; output: reports for all three captures."""
    capture_reports = []

    for capture_name in CAPTURE_NAMES:
        report = generate_capture_diagnostics(project_root, capture_name, output_root)
        capture_reports.append(report)

    return {
        "report_name": "Agentic-Stitcher Phase 2 visual diagnostics",
        "project_root": str(project_root),
        "output_root": str(output_root),
        "capture_reports": capture_reports,
    }


def write_report(report, output_path):
    """Input: report dictionary and output path; output: formatted JSON written to disk."""
    with output_path.open("w") as json_file:
        json.dump(report, json_file, indent=2)
        json_file.write("\n")


def main():
    """Input: no arguments or an optional project path; output: diagnostic images and JSON report."""
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1]).resolve()
    else:
        project_root = Path(__file__).resolve().parent.parent

    phase_two_folder = Path(__file__).resolve().parent
    output_root = phase_two_folder / "outputs"
    report_path = phase_two_folder / "phase_2_report.json"
    report = build_report(project_root, output_root)
    write_report(report, report_path)

    print("Phase 2 diagnostics completed.")
    print("Images:", output_root)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
