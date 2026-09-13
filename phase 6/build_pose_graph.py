#!/usr/bin/env python3
"""Build a provisional multi-room pose graph from one long capture."""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHASE_ONE_FOLDER = Path(__file__).resolve().parent.parent / "phase 1"
sys.path.insert(0, str(PHASE_ONE_FOLDER))

from capture_loader import load_capture


CAPTURE_NAME = "single_scan_with_ceiling"
POSE_SAMPLE_STEP = 30
LOOP_CLOSURE_DISTANCE = 0.75
MINIMUM_LOOP_FRAME_GAP = 240
TURNING_ANGLE_DEGREES = 45.0
MAX_LOOP_CLOSURE_CANDIDATES = 500
MAX_PLOT_LOOP_EDGES = 80
HORIZONTAL_AXES = [0, 2]


def make_transform(odometry_row):
    """Input: one odometry row; output: a provisional 4x4 pose transform."""
    qx = float(odometry_row["qx"])
    qy = float(odometry_row["qy"])
    qz = float(odometry_row["qz"])
    qw = float(odometry_row["qw"])
    quaternion = np.array([qx, qy, qz, qw], dtype=float)
    quaternion = quaternion / np.linalg.norm(quaternion)
    qx, qy, qz, qw = quaternion
    rotation = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = [
        float(odometry_row["x"]),
        float(odometry_row["y"]),
        float(odometry_row["z"]),
    ]
    return transform


def choose_pose_rows(odometry):
    """Input: the full odometry table; output: regularly sampled odometry rows."""
    sampled_rows = odometry.iloc[::POSE_SAMPLE_STEP].copy()

    if len(sampled_rows) > 0 and sampled_rows.index[-1] != odometry.index[-1]:
        sampled_rows = pd.concat([sampled_rows, odometry.iloc[[-1]]])

    return sampled_rows.reset_index(drop=True)


def build_pose_nodes(odometry):
    """Input: sampled odometry; output: JSON-safe pose-node records and positions."""
    sampled_rows = choose_pose_rows(odometry)
    first_transform = make_transform(sampled_rows.iloc[0])
    first_inverse = np.linalg.inv(first_transform)
    nodes = []
    positions = []

    for node_id, (_, row) in enumerate(sampled_rows.iterrows()):
        relative_transform = first_inverse @ make_transform(row)
        position = relative_transform[:3, 3]
        positions.append(position)
        nodes.append(
            {
                "node_id": int(node_id),
                "frame_id": int(row["frame"]),
                "timestamp": float(row["timestamp"]),
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
            }
        )

    return nodes, np.asarray(positions, dtype=float)


def find_loop_closure_candidates(nodes, positions):
    """Input: pose nodes and positions; output: nearby non-neighbor pose pairs."""
    candidates = []

    for first_index in range(len(nodes)):
        for second_index in range(first_index + 1, len(nodes)):
            frame_gap = nodes[second_index]["frame_id"] - nodes[first_index]["frame_id"]

            if frame_gap < MINIMUM_LOOP_FRAME_GAP:
                continue

            distance = float(np.linalg.norm(positions[second_index] - positions[first_index]))

            if distance <= LOOP_CLOSURE_DISTANCE:
                candidates.append(
                    {
                        "first_node_id": nodes[first_index]["node_id"],
                        "second_node_id": nodes[second_index]["node_id"],
                        "first_frame_id": nodes[first_index]["frame_id"],
                        "second_frame_id": nodes[second_index]["frame_id"],
                        "frame_gap": int(frame_gap),
                        "position_distance": distance,
                    }
                )

    candidates.sort(key=lambda item: item["position_distance"])
    return candidates[:MAX_LOOP_CLOSURE_CANDIDATES]


def calculate_turning_points(nodes, positions):
    """Input: pose nodes and positions; output: large-heading-change candidates."""
    turning_points = []

    if len(positions) < 3:
        return turning_points

    for index in range(1, len(positions) - 1):
        first_step = positions[index][HORIZONTAL_AXES] - positions[index - 1][HORIZONTAL_AXES]
        second_step = positions[index + 1][HORIZONTAL_AXES] - positions[index][HORIZONTAL_AXES]
        first_length = np.linalg.norm(first_step)
        second_length = np.linalg.norm(second_step)

        if first_length == 0 or second_length == 0:
            continue

        cosine = np.dot(first_step, second_step) / (first_length * second_length)
        angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

        if angle >= TURNING_ANGLE_DEGREES:
            turning_points.append(
                {
                    "node_id": nodes[index]["node_id"],
                    "frame_id": nodes[index]["frame_id"],
                    "timestamp": nodes[index]["timestamp"],
                    "turning_angle_degrees": angle,
                    "x": nodes[index]["x"],
                    "z": nodes[index]["z"],
                }
            )

    return turning_points


def save_pose_graph_plot(nodes, positions, loop_candidates, output_path):
    """Input: nodes, positions, loop candidates, and path; output: trajectory graph PNG."""
    figure, axis = plt.subplots(figsize=(11, 8))
    colors = np.arange(len(positions))
    axis.plot(
        positions[:, HORIZONTAL_AXES[0]],
        positions[:, HORIZONTAL_AXES[1]],
        color="gray",
        linewidth=1,
        label="consecutive odometry edges",
    )
    axis.scatter(
        positions[:, HORIZONTAL_AXES[0]],
        positions[:, HORIZONTAL_AXES[1]],
        c=colors,
        cmap="viridis",
        s=12,
        label="sampled poses",
    )

    for candidate in loop_candidates[:MAX_PLOT_LOOP_EDGES]:
        first = positions[candidate["first_node_id"]]
        second = positions[candidate["second_node_id"]]
        axis.plot(
            [first[HORIZONTAL_AXES[0]], second[HORIZONTAL_AXES[0]]],
            [first[HORIZONTAL_AXES[1]], second[HORIZONTAL_AXES[1]]],
            color="red",
            alpha=0.25,
            linewidth=0.8,
        )

    axis.scatter(
        positions[0, HORIZONTAL_AXES[0]],
        positions[0, HORIZONTAL_AXES[1]],
        color="green",
        s=60,
        label="start",
    )
    axis.scatter(
        positions[-1, HORIZONTAL_AXES[0]],
        positions[-1, HORIZONTAL_AXES[1]],
        color="black",
        s=60,
        label="end",
    )
    axis.set_title("Phase 6 provisional pose graph; red lines are loop candidates")
    axis.set_xlabel("Provisional world X")
    axis.set_ylabel("Provisional world Z")
    axis.axis("equal")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_turning_point_plot(nodes, positions, turning_points, output_path):
    """Input: nodes, positions, turns, and path; output: turning-point diagnostic PNG."""
    figure, axis = plt.subplots(figsize=(11, 8))
    axis.plot(
        positions[:, HORIZONTAL_AXES[0]],
        positions[:, HORIZONTAL_AXES[1]],
        color="steelblue",
        linewidth=1,
    )
    axis.scatter(
        positions[0, HORIZONTAL_AXES[0]],
        positions[0, HORIZONTAL_AXES[1]],
        color="green",
        s=60,
        label="start",
    )

    if len(turning_points) > 0:
        turn_ids = [item["node_id"] for item in turning_points]
        axis.scatter(
            positions[turn_ids, HORIZONTAL_AXES[0]],
            positions[turn_ids, HORIZONTAL_AXES[1]],
            color="red",
            s=24,
            label="turn candidates",
        )

    axis.set_title("Phase 6 candidate turning or transition regions")
    axis.set_xlabel("Provisional world X")
    axis.set_ylabel("Provisional world Z")
    axis.axis("equal")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def calculate_trajectory_metrics(positions):
    """Input: ordered pose positions; output: path length and coverage metrics."""
    if len(positions) < 2:
        return {
            "node_count": int(len(positions)),
            "path_length": 0.0,
            "start_to_end_distance": 0.0,
            "horizontal_bounds": None,
            "finite": bool(np.isfinite(positions).all()),
        }

    steps = np.diff(positions, axis=0)
    path_length = float(np.sum(np.linalg.norm(steps, axis=1)))
    start_to_end = float(np.linalg.norm(positions[-1] - positions[0]))
    horizontal_positions = positions[:, HORIZONTAL_AXES]

    return {
        "node_count": int(len(positions)),
        "path_length": path_length,
        "start_to_end_distance": start_to_end,
        "horizontal_bounds": {
            "minimum": [float(value) for value in np.min(horizontal_positions, axis=0)],
            "maximum": [float(value) for value in np.max(horizontal_positions, axis=0)],
        },
        "finite": bool(np.isfinite(positions).all()),
    }


def write_csv(records, output_path):
    """Input: list of dictionaries and output path; output: a CSV table."""
    if len(records) == 0:
        output_path.write_text("")
        return

    field_names = list(records[0].keys())

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(records)


def build_report(project_root, output_root):
    """Input: project and output folders; output: Phase 6 graph report and files."""
    capture = load_capture(project_root / CAPTURE_NAME)
    nodes, positions = build_pose_nodes(capture.odometry)
    loop_candidates = find_loop_closure_candidates(nodes, positions)
    turning_points = calculate_turning_points(nodes, positions)
    output_root.mkdir(parents=True, exist_ok=True)
    graph_csv_path = output_root / "pose_graph.csv"
    loop_csv_path = output_root / "loop_closure_candidates.csv"
    turn_csv_path = output_root / "turning_point_candidates.csv"
    graph_plot_path = output_root / "trajectory_pose_graph.png"
    turn_plot_path = output_root / "trajectory_turning_points.png"
    write_csv(nodes, graph_csv_path)
    write_csv(loop_candidates, loop_csv_path)
    write_csv(turning_points, turn_csv_path)
    save_pose_graph_plot(nodes, positions, loop_candidates, graph_plot_path)
    save_turning_point_plot(nodes, positions, turning_points, turn_plot_path)

    return {
        "report_name": "Agentic-Stitcher Phase 6 multi-room stitching prototype",
        "capture_name": CAPTURE_NAME,
        "capture_duration_seconds": float(
            capture.odometry["timestamp"].iloc[-1]
            - capture.odometry["timestamp"].iloc[0]
        ),
        "trajectory_metrics": calculate_trajectory_metrics(positions),
        "pose_graph": {
            "node_count": len(nodes),
            "consecutive_edge_count": max(len(nodes) - 1, 0),
            "loop_closure_candidate_count": len(loop_candidates),
            "turning_point_candidate_count": len(turning_points),
            "room_boundaries_inferred": False,
        },
        "parameters": {
            "pose_sample_step_frames": POSE_SAMPLE_STEP,
            "loop_closure_distance": LOOP_CLOSURE_DISTANCE,
            "minimum_loop_frame_gap": MINIMUM_LOOP_FRAME_GAP,
            "turning_angle_degrees": TURNING_ANGLE_DEGREES,
            "horizontal_axes": HORIZONTAL_AXES,
            "coordinate_status": "provisional",
        },
        "evaluation": {
            "status": "diagnostic_only",
            "loop_candidates_require_depth_or_rgb_verification": True,
            "room_count_verified": False,
            "ground_truth_available": False,
        },
        "outputs": {
            "pose_graph_csv": str(graph_csv_path),
            "loop_closure_candidates_csv": str(loop_csv_path),
            "turning_point_candidates_csv": str(turn_csv_path),
            "trajectory_pose_graph": str(graph_plot_path),
            "trajectory_turning_points": str(turn_plot_path),
        },
    }


def write_report(report, output_path):
    """Input: report dictionary and output path; output: formatted JSON on disk."""
    with output_path.open("w") as json_file:
        json.dump(report, json_file, indent=2)
        json_file.write("\n")


def main():
    """Input: no arguments or optional project path; output: Phase 6 diagnostic files."""
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1]).resolve()
    else:
        project_root = Path(__file__).resolve().parent.parent

    phase_six_folder = Path(__file__).resolve().parent
    output_root = phase_six_folder / "outputs"
    report_path = phase_six_folder / "phase_6_report.json"
    report = build_report(project_root, output_root)
    write_report(report, report_path)
    print("Phase 6 pose graph prototype completed.")
    print("Outputs:", output_root)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
