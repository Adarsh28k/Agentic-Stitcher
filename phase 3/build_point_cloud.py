#!/usr/bin/env python3
"""Build an initial point cloud from depth maps, confidence maps, and odometry."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


PHASE_ONE_FOLDER = Path(__file__).resolve().parent.parent / "phase 1"
sys.path.insert(0, str(PHASE_ONE_FOLDER))

from capture_loader import get_frame, load_capture


CAPTURE_NAMES = [
    "single_room",
    "single_scan_floor_only",
    "single_scan_with_ceiling",
]

DEPTH_TO_POSE_SCALE = 0.001
INTRINSIC_SOURCE_WIDTH = 1920.0
INTRINSIC_SOURCE_HEIGHT = 1440.0
CONFIDENCE_THRESHOLD = 1
PIXEL_STRIDE = 4
SINGLE_FRAME_PIXEL_STRIDE = 2
MAX_ACCUMULATED_FRAMES = 40
VOXEL_SIZE = 0.02
MAX_PLOT_POINTS = 30000


def choose_frame_ids(capture, maximum_frame_count):
    """Input: a loaded capture and maximum count; output: evenly spaced frame IDs."""
    all_frame_ids = sorted(capture.depth_files.keys())

    if len(all_frame_ids) <= maximum_frame_count:
        return all_frame_ids

    selected_positions = np.linspace(
        0,
        len(all_frame_ids) - 1,
        maximum_frame_count,
        dtype=int,
    )
    selected_ids = []

    for position in selected_positions:
        selected_ids.append(all_frame_ids[int(position)])

    return selected_ids


def get_frame_intrinsics(frame):
    """Input: a loaded frame; output: resolution-scaled fx, fy, cx, and cy values."""
    odometry = frame["odometry"]
    values = {}

    for name in ["fx", "fy", "cx", "cy"]:
        value = odometry.get(name)

        if value is None or not np.isfinite(float(value)):
            raise ValueError("Missing usable intrinsic value: " + name)

        values[name] = float(value)

    depth_height, depth_width = frame["depth"].shape
    horizontal_scale = depth_width / INTRINSIC_SOURCE_WIDTH
    vertical_scale = depth_height / INTRINSIC_SOURCE_HEIGHT
    values["fx"] = values["fx"] * horizontal_scale
    values["cx"] = values["cx"] * horizontal_scale
    values["fy"] = values["fy"] * vertical_scale
    values["cy"] = values["cy"] * vertical_scale

    return values["fx"], values["fy"], values["cx"], values["cy"]


def project_depth_to_camera(frame, pixel_stride, confidence_threshold):
    """Input: frame and filtering settings; output: camera points, depths, and confidence values."""
    depth = frame["depth"]
    confidence = frame["confidence"]
    fx, fy, cx, cy = get_frame_intrinsics(frame)

    row_values = np.arange(0, depth.shape[0], pixel_stride)
    column_values = np.arange(0, depth.shape[1], pixel_stride)
    column_grid, row_grid = np.meshgrid(column_values, row_values)
    sampled_depth = depth[row_grid, column_grid].astype(float)
    sampled_confidence = confidence[row_grid, column_grid].astype(float)

    valid_mask = sampled_depth > 0
    valid_mask = valid_mask & np.isfinite(sampled_depth)

    if confidence_threshold is not None:
        valid_mask = valid_mask & (sampled_confidence >= confidence_threshold)

    pixel_columns = column_grid[valid_mask].astype(float)
    pixel_rows = row_grid[valid_mask].astype(float)
    depth_values = sampled_depth[valid_mask] * DEPTH_TO_POSE_SCALE
    confidence_values = sampled_confidence[valid_mask]

    x_values = (pixel_columns - cx) * depth_values / fx
    y_values = (pixel_rows - cy) * depth_values / fy
    z_values = depth_values
    camera_points = np.column_stack((x_values, y_values, z_values))

    return camera_points, depth_values, confidence_values


def quaternion_to_rotation_matrix(odometry_row):
    """Input: odometry row with qx,qy,qz,qw; output: a 3x3 rotation matrix."""
    qx = float(odometry_row["qx"])
    qy = float(odometry_row["qy"])
    qz = float(odometry_row["qz"])
    qw = float(odometry_row["qw"])
    quaternion = np.array([qx, qy, qz, qw], dtype=float)
    quaternion_norm = np.linalg.norm(quaternion)

    if quaternion_norm == 0:
        raise ValueError("Odometry quaternion has zero length")

    qx, qy, qz, qw = quaternion / quaternion_norm

    rotation = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )

    return rotation


def transform_camera_points_to_world(camera_points, odometry_row):
    """Input: camera points and odometry row; output: points transformed into the provisional world frame."""
    rotation = quaternion_to_rotation_matrix(odometry_row)
    translation = np.array(
        [
            float(odometry_row["x"]),
            float(odometry_row["y"]),
            float(odometry_row["z"]),
        ],
        dtype=float,
    )
    with np.errstate(all="ignore"):
        world_points = camera_points @ rotation.T
    world_points = world_points + translation
    return world_points


def make_point_cloud(points):
    """Input: an Nx3 NumPy array; output: an Open3D point-cloud object."""
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return point_cloud


def process_point_cloud(point_cloud):
    """Input: an Open3D point cloud; output: voxel-downsampled and filtered clouds."""
    if len(point_cloud.points) == 0:
        return point_cloud, point_cloud

    downsampled_cloud = point_cloud.voxel_down_sample(VOXEL_SIZE)

    if len(downsampled_cloud.points) < 30:
        return downsampled_cloud, downsampled_cloud

    neighbor_count = min(20, len(downsampled_cloud.points) - 1)
    filtered_cloud, _ = downsampled_cloud.remove_statistical_outlier(
        nb_neighbors=neighbor_count,
        std_ratio=2.0,
    )
    return downsampled_cloud, filtered_cloud


def get_plot_points(points):
    """Input: an Nx3 point array; output: at most MAX_PLOT_POINTS points for plotting."""
    if len(points) <= MAX_PLOT_POINTS:
        return points

    selected_positions = np.linspace(0, len(points) - 1, MAX_PLOT_POINTS, dtype=int)
    return points[selected_positions]


def set_equal_3d_axes(axis, points):
    """Input: a 3D axis and point array; output: axis limits with equal visual scale."""
    if len(points) == 0:
        return

    minimums = np.min(points, axis=0)
    maximums = np.max(points, axis=0)
    centers = (minimums + maximums) / 2.0
    half_range = float(np.max(maximums - minimums)) / 2.0

    if half_range == 0:
        half_range = 1.0

    axis.set_xlim(centers[0] - half_range, centers[0] + half_range)
    axis.set_ylim(centers[1] - half_range, centers[1] + half_range)
    axis.set_zlim(centers[2] - half_range, centers[2] + half_range)


def save_single_frame_plot(points, output_path, frame_id):
    """Input: camera-space points, output path, and frame ID; output: a 3D PNG."""
    plotted_points = get_plot_points(points)
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")

    if len(plotted_points) > 0:
        axis.scatter(
            plotted_points[:, 0],
            plotted_points[:, 1],
            plotted_points[:, 2],
            c=plotted_points[:, 2],
            cmap="viridis",
            s=1,
        )

    axis.set_title("Single-frame camera-space point cloud — frame " + str(frame_id))
    axis.set_xlabel("Camera X")
    axis.set_ylabel("Camera Y")
    axis.set_zlabel("Camera Z")
    axis.text2D(0.02, 0.02, "Depth scale is provisional; units are not confirmed.", transform=axis.transAxes)
    set_equal_3d_axes(axis, plotted_points)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_confidence_comparison(unfiltered_points, filtered_points, output_path):
    """Input: unfiltered and filtered camera points; output: a side-by-side comparison PNG."""
    figure = plt.figure(figsize=(12, 5))
    axes = [
        figure.add_subplot(121, projection="3d"),
        figure.add_subplot(122, projection="3d"),
    ]
    point_groups = [unfiltered_points, filtered_points]
    titles = ["Valid depth only", "Depth plus confidence filter"]

    for axis, points, title in zip(axes, point_groups, titles):
        plotted_points = get_plot_points(points)

        if len(plotted_points) > 0:
            axis.scatter(
                plotted_points[:, 0],
                plotted_points[:, 1],
                plotted_points[:, 2],
                c=plotted_points[:, 2],
                cmap="viridis",
                s=1,
            )

        axis.set_title(title)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        set_equal_3d_axes(axis, plotted_points)

    figure.suptitle("Effect of confidence threshold " + str(CONFIDENCE_THRESHOLD))
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_accumulated_plot(points, output_path, title):
    """Input: world-space points, output path, and title; output: an accumulated 3D PNG."""
    plotted_points = get_plot_points(points)
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")

    if len(plotted_points) > 0:
        axis.scatter(
            plotted_points[:, 0],
            plotted_points[:, 1],
            plotted_points[:, 2],
            c=plotted_points[:, 2],
            cmap="viridis",
            s=1,
        )

    axis.set_title(title)
    axis.set_xlabel("World X; provisional units")
    axis.set_ylabel("World Y; provisional units")
    axis.set_zlabel("World Z; provisional units")
    set_equal_3d_axes(axis, plotted_points)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_camera_and_cloud_plot(points, camera_positions, output_path):
    """Input: world points, camera positions, and output path; output: combined 3D PNG."""
    plotted_points = get_plot_points(points)
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")

    if len(plotted_points) > 0:
        axis.scatter(
            plotted_points[:, 0],
            plotted_points[:, 1],
            plotted_points[:, 2],
            c=plotted_points[:, 2],
            cmap="viridis",
            s=1,
            alpha=0.5,
        )

    if len(camera_positions) > 0:
        axis.plot(
            camera_positions[:, 0],
            camera_positions[:, 1],
            camera_positions[:, 2],
            color="red",
            linewidth=2,
            label="camera trajectory",
        )
        axis.scatter(
            camera_positions[0, 0],
            camera_positions[0, 1],
            camera_positions[0, 2],
            color="green",
            s=40,
            label="trajectory start",
        )
        axis.scatter(
            camera_positions[-1, 0],
            camera_positions[-1, 1],
            camera_positions[-1, 2],
            color="red",
            s=40,
            label="trajectory end",
        )

    axis.set_title("Accumulated points and provisional camera trajectory")
    axis.set_xlabel("World X; provisional units")
    axis.set_ylabel("World Y; provisional units")
    axis.set_zlabel("World Z; provisional units")
    axis.legend()
    set_equal_3d_axes(axis, plotted_points)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_point_cloud(point_cloud, output_path):
    """Input: an Open3D point cloud and output path; output: a PLY point-cloud file."""
    success = o3d.io.write_point_cloud(str(output_path), point_cloud, write_ascii=True)

    if not success:
        raise IOError("Open3D could not write point cloud: " + str(output_path))


def get_point_bounds(points):
    """Input: an Nx3 point array; output: minimum and maximum XYZ values."""
    if len(points) == 0:
        return {"minimum": None, "maximum": None}

    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)

    return {
        "minimum": [float(value) for value in minimum],
        "maximum": [float(value) for value in maximum],
    }


def accumulate_capture_points(capture, frame_ids):
    """Input: a loaded capture and frame IDs; output: points, camera positions, and frame statistics."""
    all_world_points = []
    camera_positions = []
    frame_statistics = []

    for frame_id in frame_ids:
        frame = get_frame(capture, frame_id)
        camera_points, _, _ = project_depth_to_camera(
            frame,
            PIXEL_STRIDE,
            CONFIDENCE_THRESHOLD,
        )
        world_points = transform_camera_points_to_world(camera_points, frame["odometry"])
        all_world_points.append(world_points)
        camera_positions.append(
            [
                float(frame["odometry"]["x"]),
                float(frame["odometry"]["y"]),
                float(frame["odometry"]["z"]),
            ]
        )
        frame_statistics.append(
            {
                "frame_id": frame_id,
                "point_count": int(len(world_points)),
            }
        )

    if len(all_world_points) == 0:
        world_points = np.empty((0, 3), dtype=float)
    else:
        world_points = np.concatenate(all_world_points, axis=0)

    return world_points, np.asarray(camera_positions, dtype=float), frame_statistics


def generate_capture_outputs(project_root, capture_name, output_root):
    """Input: project folder, capture name, and output folder; output: Phase 3 report for one capture."""
    capture = load_capture(project_root / capture_name)
    capture_output = output_root / capture_name
    capture_output.mkdir(parents=True, exist_ok=True)
    all_frame_ids = sorted(capture.depth_files.keys())
    single_frame_id = all_frame_ids[len(all_frame_ids) // 2]
    single_frame = get_frame(capture, single_frame_id)
    unfiltered_points, _, _ = project_depth_to_camera(
        single_frame,
        SINGLE_FRAME_PIXEL_STRIDE,
        None,
    )
    filtered_points, _, _ = project_depth_to_camera(
        single_frame,
        SINGLE_FRAME_PIXEL_STRIDE,
        CONFIDENCE_THRESHOLD,
    )
    accumulation_frame_ids = choose_frame_ids(capture, MAX_ACCUMULATED_FRAMES)
    world_points, camera_positions, frame_statistics = accumulate_capture_points(
        capture,
        accumulation_frame_ids,
    )
    raw_cloud = make_point_cloud(world_points)
    downsampled_cloud, filtered_cloud = process_point_cloud(raw_cloud)
    downsampled_points = np.asarray(downsampled_cloud.points)
    filtered_points_world = np.asarray(filtered_cloud.points)

    single_plot_path = capture_output / "single_frame_camera_space.png"
    confidence_plot_path = capture_output / "confidence_filter_comparison.png"
    raw_plot_path = capture_output / "accumulated_world_space_raw.png"
    filtered_plot_path = capture_output / "accumulated_world_space_filtered.png"
    combined_plot_path = capture_output / "camera_and_point_cloud.png"
    raw_ply_path = capture_output / "accumulated_raw.ply"
    filtered_ply_path = capture_output / "accumulated_filtered.ply"

    save_single_frame_plot(unfiltered_points, single_plot_path, single_frame_id)
    save_confidence_comparison(unfiltered_points, filtered_points, confidence_plot_path)
    save_accumulated_plot(world_points, raw_plot_path, "Accumulated world-space point cloud before filtering")
    save_accumulated_plot(
        filtered_points_world,
        filtered_plot_path,
        "Accumulated world-space point cloud after Open3D filtering",
    )
    save_camera_and_cloud_plot(filtered_points_world, camera_positions, combined_plot_path)
    write_point_cloud(raw_cloud, raw_ply_path)
    write_point_cloud(filtered_cloud, filtered_ply_path)

    report = {
        "capture_name": capture_name,
        "single_frame_id": single_frame_id,
        "accumulation_frame_ids": accumulation_frame_ids,
        "single_frame": {
            "unfiltered_point_count": int(len(unfiltered_points)),
            "confidence_filtered_point_count": int(len(filtered_points)),
        },
        "accumulation": {
            "frame_count": len(accumulation_frame_ids),
            "raw_point_count": int(len(world_points)),
            "voxel_downsampled_point_count": int(len(downsampled_points)),
            "statistical_filtered_point_count": int(len(filtered_points_world)),
            "frame_statistics": frame_statistics,
            "camera_position_count": int(len(camera_positions)),
        },
        "bounds": {
            "raw": get_point_bounds(world_points),
            "filtered": get_point_bounds(filtered_points_world),
        },
        "assumptions": {
            "depth_to_pose_scale": DEPTH_TO_POSE_SCALE,
            "intrinsic_source_resolution": [INTRINSIC_SOURCE_WIDTH, INTRINSIC_SOURCE_HEIGHT],
            "intrinsics_scaled_to_depth_resolution": True,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "pixel_stride": PIXEL_STRIDE,
            "single_frame_pixel_stride": SINGLE_FRAME_PIXEL_STRIDE,
            "voxel_size": VOXEL_SIZE,
            "quaternion_order": "qx, qy, qz, qw",
            "pose_direction": "camera-to-world; provisional",
            "depth_unit": "unknown; scale 0.001 is provisional",
        },
        "outputs": {
            "single_frame_plot": str(single_plot_path),
            "confidence_comparison_plot": str(confidence_plot_path),
            "raw_accumulated_plot": str(raw_plot_path),
            "filtered_accumulated_plot": str(filtered_plot_path),
            "camera_and_cloud_plot": str(combined_plot_path),
            "raw_ply": str(raw_ply_path),
            "filtered_ply": str(filtered_ply_path),
        },
    }

    return report


def build_report(project_root, output_root):
    """Input: project and output folders; output: Phase 3 report for all captures."""
    capture_reports = []

    for capture_name in CAPTURE_NAMES:
        capture_report = generate_capture_outputs(project_root, capture_name, output_root)
        capture_reports.append(capture_report)

    return {
        "report_name": "Agentic-Stitcher Phase 3 point-cloud reconstruction",
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
    """Input: no arguments or an optional project path; output: point clouds, images, and report."""
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1]).resolve()
    else:
        project_root = Path(__file__).resolve().parent.parent

    phase_three_folder = Path(__file__).resolve().parent
    output_root = phase_three_folder / "outputs"
    report_path = phase_three_folder / "phase_3_report.json"
    report = build_report(project_root, output_root)
    write_report(report, report_path)

    print("Phase 3 point-cloud reconstruction completed.")
    print("Outputs:", output_root)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
