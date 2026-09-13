#!/usr/bin/env python3
"""Optimize the Phase 6 pose graph with verified depth loop closures."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy import sparse


PHASE_ONE_FOLDER = Path(__file__).resolve().parent.parent / "phase 1"
PHASE_THREE_FOLDER = Path(__file__).resolve().parent.parent / "phase 3"
sys.path.insert(0, str(PHASE_ONE_FOLDER))
sys.path.insert(0, str(PHASE_THREE_FOLDER))

from capture_loader import get_frame, load_capture
from build_point_cloud import project_depth_to_camera


CAPTURE_NAME = "single_scan_with_ceiling"
POSE_SAMPLE_STEP = 30
PIXEL_STRIDE = 8
ICP_DISTANCE_THRESHOLD = 0.75
ICP_MAX_ITERATIONS = 40
ICP_LOOP_EDGE_WEIGHT = 2.0
MAX_LOOP_TRANSLATION_CORRECTION = 1.5
MAX_LOOP_ROTATION_CORRECTION_DEGREES = 30.0
VOXEL_SIZE = 0.02
MAX_PLOT_POINTS = 30000

VERIFIED_LOOP_PAIRS = [
    (2760, 7530),
    (5040, 9600),
    (4560, 9120),
    (3720, 8400),
    (3720, 8430),
    (2790, 7560),
]


def make_transform(odometry_row):
    """Input: one odometry row; output: a provisional 4x4 pose transform."""
    quaternion = np.array(
        [
            float(odometry_row["qx"]),
            float(odometry_row["qy"]),
            float(odometry_row["qz"]),
            float(odometry_row["qw"]),
        ],
        dtype=float,
    )
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
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = rotation
    pose[:3, 3] = [
        float(odometry_row["x"]),
        float(odometry_row["y"]),
        float(odometry_row["z"]),
    ]
    return pose


def choose_frame_ids(odometry):
    """Input: the full odometry table; output: sampled frame IDs for the graph."""
    frame_ids = odometry["frame"].astype(int).tolist()
    selected_ids = frame_ids[::POSE_SAMPLE_STEP]

    if len(frame_ids) > 0 and selected_ids[-1] != frame_ids[-1]:
        selected_ids.append(frame_ids[-1])

    return selected_ids


def build_raw_poses(odometry, frame_ids):
    """Input: odometry and frame IDs; output: poses normalized to the first pose."""
    odometry_by_frame = odometry.set_index("frame")
    absolute_first_pose = make_transform(odometry_by_frame.loc[frame_ids[0]])
    first_inverse = np.linalg.inv(absolute_first_pose)
    poses = []

    for frame_id in frame_ids:
        absolute_pose = make_transform(odometry_by_frame.loc[frame_id])
        poses.append(first_inverse @ absolute_pose)

    return poses


def transform_points(points, pose):
    """Input: camera points and a 4x4 pose; output: world-space points."""
    with np.errstate(all="ignore"):
        return points @ pose[:3, :3].T + pose[:3, 3]


def load_depth_points(capture, frame_id, cache):
    """Input: capture, frame ID, and cache; output: sampled camera-space depth points."""
    if frame_id not in cache:
        frame = get_frame(capture, frame_id)
        points, _, _ = project_depth_to_camera(
            frame,
            PIXEL_STRIDE,
            1,
        )
        cache[frame_id] = {
            "points": points,
            "pose": make_transform(frame["odometry"]),
        }

    return cache[frame_id]


def score_transform(source_points, target_points, pose):
    """Input: source, target, and transform; output: correspondence fitness and RMSE."""
    target_tree = cKDTree(target_points)
    distances, _ = target_tree.query(
        transform_points(source_points, pose),
        distance_upper_bound=ICP_DISTANCE_THRESHOLD,
        workers=1,
    )
    valid_mask = np.isfinite(distances) & (distances < ICP_DISTANCE_THRESHOLD)

    if not np.any(valid_mask):
        return 0.0, float("inf")

    fitness = float(np.mean(valid_mask))
    rmse = float(np.sqrt(np.mean(distances[valid_mask] ** 2)))
    return fitness, rmse


def estimate_rigid_transform(source_points, target_points):
    """Input: corresponding point arrays; output: least-squares rigid transform."""
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center

    with np.errstate(all="ignore"):
        left_vectors, _, right_vectors_transposed = np.linalg.svd(
            source_centered.T @ target_centered
        )

    rotation = right_vectors_transposed.T @ left_vectors.T

    if np.linalg.det(rotation) < 0:
        right_vectors_transposed[-1, :] *= -1
        rotation = right_vectors_transposed.T @ left_vectors.T

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def run_icp(source_points, target_points, initial_pose):
    """Input: source, target, and initial pose; output: refined pose and scores."""
    target_tree = cKDTree(target_points)
    current_pose = initial_pose.copy()

    for _ in range(ICP_MAX_ITERATIONS):
        moved_points = transform_points(source_points, current_pose)
        distances, target_indices = target_tree.query(
            moved_points,
            distance_upper_bound=ICP_DISTANCE_THRESHOLD,
            workers=1,
        )
        valid_mask = np.isfinite(distances) & (distances < ICP_DISTANCE_THRESHOLD)

        if np.count_nonzero(valid_mask) < 3:
            break

        delta_pose = estimate_rigid_transform(
            moved_points[valid_mask],
            target_points[target_indices[valid_mask]],
        )
        current_pose = delta_pose @ current_pose

        if np.max(np.abs(delta_pose - np.eye(4))) < 1e-6:
            break

    fitness, rmse = score_transform(source_points, target_points, current_pose)
    return current_pose, fitness, rmse


def rotation_difference_degrees(first_pose, second_pose):
    """Input: two poses; output: angle between their rotations in degrees."""
    relative_rotation = first_pose[:3, :3].T @ second_pose[:3, :3]
    cosine = (np.trace(relative_rotation) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def build_loop_measurements(capture, raw_poses, frame_ids):
    """Input: capture and raw graph poses; output: verified loop-edge measurements."""
    frame_to_node = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    point_cache = {}
    measurements = []

    for first_frame, second_frame in VERIFIED_LOOP_PAIRS:
        if first_frame not in frame_to_node or second_frame not in frame_to_node:
            continue

        first = load_depth_points(capture, first_frame, point_cache)
        second = load_depth_points(capture, second_frame, point_cache)
        first_node = frame_to_node[first_frame]
        second_node = frame_to_node[second_frame]
        odometry_relative = np.linalg.inv(raw_poses[first_node]) @ raw_poses[second_node]
        refined_pose, fitness, rmse = run_icp(
            second["points"],
            first["points"],
            odometry_relative,
        )
        before_fitness, before_rmse = score_transform(
            second["points"],
            first["points"],
            odometry_relative,
        )

        correction = refined_pose @ np.linalg.inv(odometry_relative)
        correction_translation = float(np.linalg.norm(correction[:3, 3]))
        correction_rotation = rotation_difference_degrees(
            odometry_relative,
            refined_pose,
        )
        measurements.append(
            {
                "first_frame_id": first_frame,
                "second_frame_id": second_frame,
                "first_node_id": first_node,
                "second_node_id": second_node,
                "measurement": refined_pose,
                "before_fitness": before_fitness,
                "before_rmse": before_rmse,
                "after_fitness": fitness,
                "after_rmse": rmse,
                "correction_translation": correction_translation,
                "correction_rotation_degrees": correction_rotation,
                "accepted": bool(
                    fitness >= 0.30
                    and rmse <= 0.10
                    and fitness > before_fitness + 0.02
                    and correction_translation <= MAX_LOOP_TRANSLATION_CORRECTION
                    and correction_rotation <= MAX_LOOP_ROTATION_CORRECTION_DEGREES
                ),
            }
        )

    return measurements


def pose_to_vector(pose):
    """Input: a 4x4 pose; output: three rotation-vector and three translation values."""
    rotation_vector = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
    return np.concatenate([rotation_vector, pose[:3, 3]])


def vector_to_pose(values):
    """Input: six pose values; output: a 4x4 pose matrix."""
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = Rotation.from_rotvec(values[:3]).as_matrix()
    pose[:3, 3] = values[3:6]
    return pose


def make_edges(raw_poses, loop_measurements):
    """Input: raw poses and loop measurements; output: weighted pose-graph edges."""
    edges = []

    for node_id in range(len(raw_poses) - 1):
        measurement = np.linalg.inv(raw_poses[node_id]) @ raw_poses[node_id + 1]
        edges.append(
            {
                "first_node_id": node_id,
                "second_node_id": node_id + 1,
                "measurement": measurement,
                "weight": 1.0,
                "edge_type": "odometry",
            }
        )

    for loop in loop_measurements:
        edges.append(
            {
                "first_node_id": loop["first_node_id"],
                "second_node_id": loop["second_node_id"],
                "measurement": loop["measurement"],
                "weight": ICP_LOOP_EDGE_WEIGHT,
                "edge_type": "verified_loop_closure",
            }
        )

    return edges


def make_jacobian_sparsity(edges, node_count):
    """Input: graph edges and node count; output: sparse Jacobian dependency pattern."""
    edge_count = len(edges)
    variable_count = max(node_count - 1, 0) * 6
    sparsity = sparse.lil_matrix((edge_count * 6, variable_count), dtype=int)

    for edge_id, edge in enumerate(edges):
        row_start = edge_id * 6
        for node_id in [edge["first_node_id"], edge["second_node_id"]]:
            if 0 < node_id < node_count:
                column_start = (node_id - 1) * 6
                sparsity[row_start : row_start + 6, column_start : column_start + 6] = 1

    return sparsity.tocsr()


def optimize_poses(raw_poses, edges):
    """Input: initial poses and graph edges; output: optimized poses and solver result."""
    if len(raw_poses) <= 1:
        return raw_poses, None

    fixed_first_pose = pose_to_vector(raw_poses[0])
    initial_values = np.concatenate([pose_to_vector(pose) for pose in raw_poses[1:]])

    def unpack(values):
        pose_vectors = [fixed_first_pose]
        pose_vectors.extend(values.reshape((-1, 6)))
        return [vector_to_pose(vector) for vector in pose_vectors]

    def residual(values):
        poses = unpack(values)
        residual_values = []

        for edge in edges:
            first_pose = poses[edge["first_node_id"]]
            second_pose = poses[edge["second_node_id"]]
            actual_relative = np.linalg.inv(first_pose) @ second_pose
            error = np.linalg.inv(edge["measurement"]) @ actual_relative
            rotation_error = Rotation.from_matrix(error[:3, :3]).as_rotvec()
            translation_error = error[:3, 3]
            weight = edge["weight"]
            residual_values.extend((rotation_error * weight).tolist())
            residual_values.extend((translation_error * weight).tolist())

        return np.asarray(residual_values, dtype=float)

    jacobian_sparsity = make_jacobian_sparsity(edges, len(raw_poses))
    result = least_squares(
        residual,
        initial_values,
        jac_sparsity=jacobian_sparsity,
        loss="soft_l1",
        f_scale=0.05,
        max_nfev=100,
        verbose=0,
    )
    return unpack(result.x), result


def edge_error(pose_first, pose_second, measurement):
    """Input: two poses and an edge measurement; output: translation and rotation errors."""
    actual_relative = np.linalg.inv(pose_first) @ pose_second
    error = np.linalg.inv(measurement) @ actual_relative
    translation_error = float(np.linalg.norm(error[:3, 3]))
    rotation_error = float(np.linalg.norm(Rotation.from_matrix(error[:3, :3]).as_rotvec()))
    return translation_error, np.degrees(rotation_error)


def calculate_edge_errors(poses, edges, edge_type):
    """Input: poses, graph edges, and edge type; output: error records for that edge type."""
    records = []

    for edge in edges:
        if edge["edge_type"] != edge_type:
            continue

        translation_error, rotation_error = edge_error(
            poses[edge["first_node_id"]],
            poses[edge["second_node_id"]],
            edge["measurement"],
        )
        records.append(
            {
                "first_node_id": edge["first_node_id"],
                "second_node_id": edge["second_node_id"],
                "translation_error": translation_error,
                "rotation_error_degrees": rotation_error,
            }
        )

    return records


def calculate_graph_residual_sum(poses, edges):
    """Input: poses and graph edges; output: weighted squared residual sum."""
    residual_sum = 0.0

    for edge in edges:
        actual_relative = (
            np.linalg.inv(poses[edge["first_node_id"]])
            @ poses[edge["second_node_id"]]
        )
        error = np.linalg.inv(edge["measurement"]) @ actual_relative
        rotation_error = Rotation.from_matrix(error[:3, :3]).as_rotvec()
        translation_error = error[:3, 3]
        weight = edge["weight"]
        residual_sum += float(
            np.sum((rotation_error * weight) ** 2)
            + np.sum((translation_error * weight) ** 2)
        )

    return residual_sum


def downsample_points(points):
    """Input: an Nx3 point array; output: one representative point per voxel."""
    voxel_keys = np.floor(points / VOXEL_SIZE).astype(np.int64)
    _, first_indices = np.unique(voxel_keys, axis=0, return_index=True)
    return points[np.sort(first_indices)]


def build_accumulated_cloud(records, poses):
    """Input: frame point records and poses; output: a voxel-downsampled global cloud."""
    world_points = []

    for record, pose in zip(records, poses):
        world_points.append(transform_points(record["points"], pose))

    return downsample_points(np.concatenate(world_points, axis=0))


def write_ascii_ply(points, output_path):
    """Input: point array and output path; output: an ASCII PLY point cloud."""
    with output_path.open("w") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex " + str(len(points)) + "\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("end_header\n")

        for point in points:
            ply_file.write("{:.6f} {:.6f} {:.6f}\n".format(*point))


def calculate_plane_ratio(points):
    """Input: point array; output: deterministic approximate largest-plane ratio."""
    if len(points) < 3:
        return 0.0

    random_generator = np.random.default_rng(7)
    sample_count = min(len(points), 30000)
    sample = points

    if len(points) > sample_count:
        sample = points[random_generator.choice(len(points), sample_count, replace=False)]

    best_count = 0

    for _ in range(400):
        indices = random_generator.choice(len(sample), 3, replace=False)
        first, second, third = sample[indices]
        normal = np.cross(second - first, third - first)
        normal_size = np.linalg.norm(normal)

        if normal_size < 1e-8:
            continue

        normal = normal / normal_size
        with np.errstate(all="ignore"):
            distances = np.abs((sample - first) @ normal)
        best_count = max(best_count, int(np.count_nonzero(distances < 0.04)))

    return float(best_count / len(sample))


def save_trajectory_plot(raw_poses, optimized_poses, loop_measurements, output_path):
    """Input: raw/optimized poses, loops, and path; output: trajectory comparison PNG."""
    raw_positions = np.asarray([pose[:3, 3] for pose in raw_poses])
    optimized_positions = np.asarray([pose[:3, 3] for pose in optimized_poses])
    figure, axis = plt.subplots(figsize=(11, 8))
    axis.plot(raw_positions[:, 0], raw_positions[:, 2], color="red", label="raw odometry")
    axis.plot(
        optimized_positions[:, 0],
        optimized_positions[:, 2],
        color="blue",
        label="optimized pose graph",
    )

    for loop in loop_measurements:
        first = optimized_positions[loop["first_node_id"]]
        second = optimized_positions[loop["second_node_id"]]
        axis.plot(
            [first[0], second[0]],
            [first[2], second[2]],
            color="green",
            linewidth=2,
            alpha=0.7,
        )

    axis.scatter(raw_positions[0, 0], raw_positions[0, 2], color="black", label="start")
    axis.set_title("Raw versus optimized camera trajectory")
    axis.set_xlabel("Provisional world X")
    axis.set_ylabel("Provisional world Z")
    axis.axis("equal")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def get_plot_points(points):
    """Input: points; output: at most MAX_PLOT_POINTS evenly sampled points."""
    if len(points) <= MAX_PLOT_POINTS:
        return points
    indices = np.linspace(0, len(points) - 1, MAX_PLOT_POINTS, dtype=int)
    return points[indices]


def set_equal_axes(axis, points):
    """Input: 3D axis and points; output: equal visual limits."""
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum) / 2.0), 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def save_cloud_plot(raw_cloud, optimized_cloud, output_path):
    """Input: raw and optimized clouds; output: side-by-side comparison PNG."""
    figure = plt.figure(figsize=(14, 6))
    axes = [
        figure.add_subplot(121, projection="3d"),
        figure.add_subplot(122, projection="3d"),
    ]

    for axis, points, title in zip(
        axes,
        [raw_cloud, optimized_cloud],
        ["Raw odometry cloud", "Optimized pose-graph cloud"],
    ):
        plotted_points = get_plot_points(points)
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
        set_equal_axes(axis, plotted_points)

    figure.suptitle("Point-cloud comparison after pose-graph optimization")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_loop_error_plot(raw_errors, optimized_errors, output_path):
    """Input: raw/optimized loop errors and path; output: loop-error comparison PNG."""
    labels = [
        str(item["first_node_id"]) + "→" + str(item["second_node_id"])
        for item in raw_errors
    ]
    raw_values = [item["translation_error"] for item in raw_errors]
    optimized_values = [item["translation_error"] for item in optimized_errors]
    positions = np.arange(len(labels))
    width = 0.35
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.bar(positions - width / 2, raw_values, width, label="raw")
    axis.bar(positions + width / 2, optimized_values, width, label="optimized")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=30)
    axis.set_ylabel("Loop translation error")
    axis.set_title("Loop-closure error before and after optimization")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def build_report(project_root, output_root):
    """Input: project and output folders; output: optimization report and files."""
    capture = load_capture(project_root / CAPTURE_NAME)
    frame_ids = choose_frame_ids(capture.odometry)
    raw_poses = build_raw_poses(capture.odometry, frame_ids)
    all_loop_measurements = build_loop_measurements(capture, raw_poses, frame_ids)
    loop_measurements = [
        item for item in all_loop_measurements if item["accepted"]
    ]
    edges = make_edges(raw_poses, loop_measurements)
    optimized_poses, solver_result = optimize_poses(raw_poses, edges)
    output_root.mkdir(parents=True, exist_ok=True)
    point_cache = {}
    records = []

    for frame_id in frame_ids:
        record = load_depth_points(capture, frame_id, point_cache)
        records.append(record)

    raw_cloud = build_accumulated_cloud(records, raw_poses)
    optimized_cloud = build_accumulated_cloud(records, optimized_poses)
    raw_plane_ratio = calculate_plane_ratio(raw_cloud)
    optimized_plane_ratio = calculate_plane_ratio(optimized_cloud)
    raw_cloud_path = output_root / "raw_odometry_cloud.ply"
    optimized_cloud_path = output_root / "optimized_pose_graph_cloud.ply"
    trajectory_path = output_root / "trajectory_before_after.png"
    cloud_path = output_root / "point_cloud_before_after.png"
    loop_error_path = output_root / "loop_error_before_after.png"
    write_ascii_ply(raw_cloud, raw_cloud_path)
    write_ascii_ply(optimized_cloud, optimized_cloud_path)
    save_trajectory_plot(raw_poses, optimized_poses, loop_measurements, trajectory_path)
    save_cloud_plot(raw_cloud, optimized_cloud, cloud_path)
    raw_loop_errors = calculate_edge_errors(raw_poses, edges, "verified_loop_closure")
    optimized_loop_errors = calculate_edge_errors(
        optimized_poses,
        edges,
        "verified_loop_closure",
    )
    save_loop_error_plot(raw_loop_errors, optimized_loop_errors, loop_error_path)

    raw_residual = calculate_graph_residual_sum(raw_poses, edges)
    optimized_residual = calculate_graph_residual_sum(optimized_poses, edges)
    if optimized_plane_ratio > raw_plane_ratio * 1.02:
        alignment_verdict = "optimized_cloud_improved"
        recommended_alignment = "optimized_pose_graph"
    else:
        alignment_verdict = "optimized_cloud_not_improved"
        recommended_alignment = "raw_odometry"

    return {
        "report_name": "Agentic-Stitcher Phase 6B pose-graph optimization",
        "capture_name": CAPTURE_NAME,
        "frame_count": len(frame_ids),
        "loop_measurement_candidates_checked": len(all_loop_measurements),
        "verified_loop_measurement_count": len(loop_measurements),
        "verified_loop_measurements": [
            {
                key: value
                for key, value in item.items()
                if key != "measurement"
            }
            for item in loop_measurements
        ],
        "solver": {
            "success": bool(solver_result.success) if solver_result is not None else False,
            "message": solver_result.message if solver_result is not None else "No optimization was needed.",
            "function_evaluations": int(solver_result.nfev) if solver_result is not None else 0,
            "raw_residual_sum": raw_residual,
            "optimized_residual_sum": optimized_residual,
            "loop_edge_weight": ICP_LOOP_EDGE_WEIGHT,
            "maximum_loop_translation_correction": MAX_LOOP_TRANSLATION_CORRECTION,
            "maximum_loop_rotation_correction_degrees": MAX_LOOP_ROTATION_CORRECTION_DEGREES,
        },
        "loop_errors_before": raw_loop_errors,
        "loop_errors_after": optimized_loop_errors,
        "cloud_metrics": {
            "raw_point_count": int(len(raw_cloud)),
            "optimized_point_count": int(len(optimized_cloud)),
            "raw_largest_plane_ratio": calculate_plane_ratio(raw_cloud),
            "optimized_largest_plane_ratio": calculate_plane_ratio(optimized_cloud),
            "alignment_verdict": alignment_verdict,
            "recommended_alignment": recommended_alignment,
            "raw_bounds_minimum": [float(value) for value in raw_cloud.min(axis=0)],
            "raw_bounds_maximum": [float(value) for value in raw_cloud.max(axis=0)],
            "optimized_bounds_minimum": [float(value) for value in optimized_cloud.min(axis=0)],
            "optimized_bounds_maximum": [float(value) for value in optimized_cloud.max(axis=0)],
        },
        "evaluation": {
            "status": "diagnostic_only",
            "loop_edges_are_depth_verified": True,
            "room_boundaries_inferred": False,
            "physical_ground_truth_available": False,
        },
        "outputs": {
            "raw_cloud": str(raw_cloud_path),
            "optimized_cloud": str(optimized_cloud_path),
            "trajectory_comparison": str(trajectory_path),
            "point_cloud_comparison": str(cloud_path),
            "loop_error_comparison": str(loop_error_path),
        },
    }


def write_report(report, output_path):
    """Input: report dictionary and output path; output: formatted JSON on disk."""
    with output_path.open("w") as json_file:
        json.dump(report, json_file, indent=2)
        json_file.write("\n")


def main():
    """Input: no arguments or optional project path; output: optimized graph files."""
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1]).resolve()
    else:
        project_root = Path(__file__).resolve().parent.parent

    phase_six_folder = Path(__file__).resolve().parent
    output_root = phase_six_folder / "outputs_optimized"
    report_path = phase_six_folder / "phase_6_optimized_report.json"
    report = build_report(project_root, output_root)
    write_report(report, report_path)
    print("Phase 6B pose-graph optimization completed.")
    print("Outputs:", output_root)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
