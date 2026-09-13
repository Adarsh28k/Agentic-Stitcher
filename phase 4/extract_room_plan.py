#!/usr/bin/env python3
"""Detect provisional floor, ceiling, and wall planes from Phase 3 point clouds."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull


CAPTURE_NAMES = [
    "single_room",
    "single_scan_floor_only",
    "single_scan_with_ceiling",
]

UP_AXIS = 1
HORIZONTAL_AXES = [0, 2]
PLANE_DISTANCE_THRESHOLD = 0.04
RANSAC_ITERATIONS = 1000
MAX_PLANES = 12
MIN_PLANE_POINTS = 150
HORIZONTAL_NORMAL_THRESHOLD = 0.75
VERTICAL_NORMAL_THRESHOLD = 0.30
MIN_FLOOR_CEILING_SEPARATION = 0.20
MAX_PLOT_POINTS = 30000


def load_filtered_point_cloud(project_root, capture_name):
    """Input: project folder and capture name; output: point cloud and source path."""
    point_cloud_path = (
        project_root
        / "phase 3"
        / "outputs"
        / capture_name
        / "accumulated_filtered.ply"
    )

    if not point_cloud_path.exists():
        raise FileNotFoundError("Missing Phase 3 point cloud: " + str(point_cloud_path))

    point_cloud = o3d.io.read_point_cloud(str(point_cloud_path))

    if len(point_cloud.points) == 0:
        raise ValueError("Point cloud is empty: " + str(point_cloud_path))

    return point_cloud, point_cloud_path


def normalize_plane_model(plane_model):
    """Input: raw plane coefficients; output: normalized normal and offset."""
    plane_values = np.asarray(plane_model, dtype=float)
    normal = plane_values[:3]
    offset = float(plane_values[3])
    normal_length = float(np.linalg.norm(normal))

    if normal_length == 0:
        raise ValueError("RANSAC returned a zero-length plane normal")

    normalized_normal = normal / normal_length
    normalized_offset = offset / normal_length
    return normalized_normal, normalized_offset


def segment_planes(points):
    """Input: an Nx3 point array; output: plane records and remaining points."""
    remaining_points = points.copy()
    remaining_indices = np.arange(len(points), dtype=int)
    detected_planes = []

    for plane_number in range(MAX_PLANES):
        if len(remaining_points) < MIN_PLANE_POINTS:
            break

        working_cloud = o3d.geometry.PointCloud()
        working_cloud.points = o3d.utility.Vector3dVector(remaining_points)
        plane_model, local_inlier_indices = working_cloud.segment_plane(
            distance_threshold=PLANE_DISTANCE_THRESHOLD,
            ransac_n=3,
            num_iterations=RANSAC_ITERATIONS,
        )
        local_inlier_indices = np.asarray(local_inlier_indices, dtype=int)

        if len(local_inlier_indices) < MIN_PLANE_POINTS:
            break

        normal, offset = normalize_plane_model(plane_model)
        original_inlier_indices = remaining_indices[local_inlier_indices]
        detected_planes.append(
            {
                "plane_number": plane_number,
                "normal": normal,
                "offset": offset,
                "inlier_indices": original_inlier_indices,
                "inlier_count": int(len(original_inlier_indices)),
            }
        )

        keep_mask = np.ones(len(remaining_points), dtype=bool)
        keep_mask[local_inlier_indices] = False
        remaining_points = remaining_points[keep_mask]
        remaining_indices = remaining_indices[keep_mask]

    return detected_planes, remaining_points


def classify_plane(plane, points):
    """Input: a plane record and all points; output: orientation, height, and residual details."""
    normal = plane["normal"]
    inlier_points = points[plane["inlier_indices"]]
    vertical_component = abs(float(normal[UP_AXIS]))
    heights = inlier_points[:, UP_AXIS]
    residuals = np.abs(np.einsum("ij,j->i", inlier_points, normal) + plane["offset"])

    if vertical_component >= HORIZONTAL_NORMAL_THRESHOLD:
        orientation = "horizontal"
    elif vertical_component <= VERTICAL_NORMAL_THRESHOLD:
        orientation = "vertical"
    else:
        orientation = "other"

    if orientation == "horizontal":
        label = "horizontal_candidate"
    elif orientation == "vertical":
        label = "wall_candidate"
    else:
        label = "other_candidate"

    plane["orientation"] = orientation
    plane["label"] = label
    plane["height_mean"] = float(np.mean(heights))
    plane["height_minimum"] = float(np.min(heights))
    plane["height_maximum"] = float(np.max(heights))
    plane["mean_residual"] = float(np.mean(residuals))
    plane["normal_vertical_component"] = vertical_component
    return plane


def choose_floor_and_ceiling(planes):
    """Input: classified plane records; output: selected floor and ceiling records or None."""
    horizontal_planes = []

    for plane in planes:
        if plane["orientation"] == "horizontal":
            horizontal_planes.append(plane)

    horizontal_planes.sort(key=lambda item: item["height_mean"])
    floor = horizontal_planes[0] if len(horizontal_planes) > 0 else None
    ceiling = None

    if len(horizontal_planes) > 1:
        possible_ceiling = horizontal_planes[-1]
        separation = possible_ceiling["height_mean"] - floor["height_mean"]

        if separation >= MIN_FLOOR_CEILING_SEPARATION:
            ceiling = possible_ceiling

    if floor is not None:
        floor["label"] = "floor"

    if ceiling is not None:
        ceiling["label"] = "ceiling"

    return floor, ceiling


def select_wall_planes(planes):
    """Input: classified plane records; output: planes classified as candidate walls."""
    wall_planes = []

    for plane in planes:
        if plane["orientation"] == "vertical":
            plane["label"] = "wall"
            wall_planes.append(plane)

    return wall_planes


def get_floor_height(floor):
    """Input: a floor plane or None; output: the reference floor height."""
    if floor is None:
        return 0.0

    return float(floor["height_mean"])


def intersect_wall_lines(first_plane, second_plane, floor_height):
    """Input: two wall planes and floor height; output: a 2D wall-line intersection or None."""
    first_normal = first_plane["normal"]
    second_normal = second_plane["normal"]
    first_horizontal = first_normal[HORIZONTAL_AXES]
    second_horizontal = second_normal[HORIZONTAL_AXES]
    first_right_side = -(
        first_plane["offset"] + first_normal[UP_AXIS] * floor_height
    )
    second_right_side = -(
        second_plane["offset"] + second_normal[UP_AXIS] * floor_height
    )
    matrix = np.vstack((first_horizontal, second_horizontal))
    determinant = float(np.linalg.det(matrix))

    if abs(determinant) < 0.10:
        return None

    intersection = np.linalg.solve(
        matrix,
        np.array([first_right_side, second_right_side], dtype=float),
    )
    return intersection


def build_wall_intersections(wall_planes, floor_height, points):
    """Input: wall planes, floor height, and cloud points; output: plausible 2D wall intersections."""
    if len(wall_planes) < 2:
        return np.empty((0, 2), dtype=float)

    horizontal_points = points[:, HORIZONTAL_AXES]
    minimums = np.min(horizontal_points, axis=0) - 1.0
    maximums = np.max(horizontal_points, axis=0) + 1.0
    intersections = []

    for first_index in range(len(wall_planes)):
        for second_index in range(first_index + 1, len(wall_planes)):
            intersection = intersect_wall_lines(
                wall_planes[first_index],
                wall_planes[second_index],
                floor_height,
            )

            if intersection is None:
                continue

            inside_bounds = np.all(intersection >= minimums) and np.all(intersection <= maximums)

            if inside_bounds:
                intersections.append(intersection)

    if len(intersections) == 0:
        return np.empty((0, 2), dtype=float)

    return np.asarray(intersections, dtype=float)


def build_room_outline(intersections):
    """Input: 2D wall intersections; output: convex-hull vertices, area, and perimeter."""
    if len(intersections) < 3:
        return {
            "vertices": [],
            "area": None,
            "perimeter": None,
            "status": "not_enough_intersections",
        }

    hull = ConvexHull(intersections)
    vertices = intersections[hull.vertices]
    next_vertices = np.roll(vertices, -1, axis=0)
    cross_products = vertices[:, 0] * next_vertices[:, 1]
    cross_products = cross_products - vertices[:, 1] * next_vertices[:, 0]
    area = abs(float(np.sum(cross_products))) / 2.0
    edge_lengths = np.linalg.norm(next_vertices - vertices, axis=1)
    perimeter = float(np.sum(edge_lengths))

    return {
        "vertices": [[float(value) for value in vertex] for vertex in vertices],
        "area": area,
        "perimeter": perimeter,
        "status": "estimated_convex_hull",
    }


def make_plane_labels(point_count, planes):
    """Input: point count and plane records; output: an integer label per point."""
    labels = np.full(point_count, -1, dtype=int)

    for plane_index, plane in enumerate(planes):
        labels[plane["inlier_indices"]] = plane_index

    return labels


def get_plot_indices(point_count):
    """Input: point count; output: evenly spaced indices for a readable plot."""
    if point_count <= MAX_PLOT_POINTS:
        return np.arange(point_count, dtype=int)

    return np.linspace(0, point_count - 1, MAX_PLOT_POINTS, dtype=int)


def save_plane_plot(points, planes, output_path):
    """Input: point array, planes, and output path; output: a plane-colored 3D PNG."""
    plot_indices = get_plot_indices(len(points))
    labels = make_plane_labels(len(points), planes)
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    plotted_points = points[plot_indices]
    plotted_labels = labels[plot_indices]
    colors = np.zeros((len(plot_indices), 3), dtype=float)
    colors[:] = [0.65, 0.65, 0.65]
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(planes), 1)))

    for plane_index in range(len(planes)):
        colors[plotted_labels == plane_index] = palette[plane_index][:3]

    axis.scatter(
        plotted_points[:, 0],
        plotted_points[:, 1],
        plotted_points[:, 2],
        c=colors,
        s=1,
    )
    axis.set_title("Detected planes; gray points were not assigned")
    axis.set_xlabel("World X; provisional")
    axis.set_ylabel("World Y; provisional up axis")
    axis.set_zlabel("World Z; provisional")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_outline_plot(points, wall_planes, outline, output_path):
    """Input: points, wall planes, outline, and output path; output: a 2D room-outline PNG."""
    figure, axis = plt.subplots(figsize=(8, 8))

    for plane_index, plane in enumerate(wall_planes):
        wall_points = points[plane["inlier_indices"]]
        wall_points = wall_points[:, HORIZONTAL_AXES]
        plot_indices = get_plot_indices(len(wall_points))
        axis.scatter(
            wall_points[plot_indices, 0],
            wall_points[plot_indices, 1],
            s=2,
            label="wall " + str(plane_index + 1),
        )

    vertices = np.asarray(outline["vertices"], dtype=float)

    if len(vertices) > 0:
        closed_vertices = np.vstack((vertices, vertices[0]))
        axis.plot(
            closed_vertices[:, 0],
            closed_vertices[:, 1],
            color="black",
            linewidth=2,
            label="estimated outline",
        )

    axis.set_title("Provisional room outline")
    axis.set_xlabel("Horizontal axis " + str(HORIZONTAL_AXES[0]) + "; provisional")
    axis.set_ylabel("Horizontal axis " + str(HORIZONTAL_AXES[1]) + "; provisional")
    axis.axis("equal")
    axis.grid(True)

    if len(wall_planes) > 0 and len(wall_planes) <= 8:
        axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_floor_ceiling_plot(points, floor, ceiling, output_path):
    """Input: points, floor plane, ceiling plane, and output path; output: a plane-height PNG."""
    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d")
    assigned_indices = []

    if floor is not None:
        floor_points = points[floor["inlier_indices"]]
        floor_plot_indices = get_plot_indices(len(floor_points))
        axis.scatter(
            floor_points[floor_plot_indices, 0],
            floor_points[floor_plot_indices, 1],
            floor_points[floor_plot_indices, 2],
            color="saddlebrown",
            s=2,
            label="floor",
        )
        assigned_indices.extend(floor["inlier_indices"].tolist())

    if ceiling is not None:
        ceiling_points = points[ceiling["inlier_indices"]]
        ceiling_plot_indices = get_plot_indices(len(ceiling_points))
        axis.scatter(
            ceiling_points[ceiling_plot_indices, 0],
            ceiling_points[ceiling_plot_indices, 1],
            ceiling_points[ceiling_plot_indices, 2],
            color="royalblue",
            s=2,
            label="ceiling",
        )
        assigned_indices.extend(ceiling["inlier_indices"].tolist())

    axis.set_title("Floor and ceiling candidates")
    axis.set_xlabel("World X; provisional")
    axis.set_ylabel("World Y; provisional up axis")
    axis.set_zlabel("World Z; provisional")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def make_plane_report(plane):
    """Input: an internal plane record; output: a JSON-safe plane summary."""
    return {
        "plane_number": int(plane["plane_number"]),
        "label": plane["label"],
        "orientation": plane["orientation"],
        "normal": [float(value) for value in plane["normal"]],
        "offset": float(plane["offset"]),
        "inlier_count": int(plane["inlier_count"]),
        "height_mean": float(plane["height_mean"]),
        "height_minimum": float(plane["height_minimum"]),
        "height_maximum": float(plane["height_maximum"]),
        "normal_vertical_component": float(plane["normal_vertical_component"]),
        "mean_residual": float(plane["mean_residual"]),
    }


def build_capture_report(project_root, capture_name, output_root):
    """Input: project folder, capture name, and output folder; output: Phase 4 report for one capture."""
    point_cloud, point_cloud_path = load_filtered_point_cloud(project_root, capture_name)
    points = np.asarray(point_cloud.points, dtype=float)
    detected_planes, _ = segment_planes(points)

    for plane in detected_planes:
        classify_plane(plane, points)

    floor, ceiling = choose_floor_and_ceiling(detected_planes)
    wall_planes = select_wall_planes(detected_planes)
    floor_height = get_floor_height(floor)
    intersections = build_wall_intersections(wall_planes, floor_height, points)
    outline = build_room_outline(intersections)
    capture_output = output_root / capture_name
    capture_output.mkdir(parents=True, exist_ok=True)
    plane_plot_path = capture_output / "detected_planes_3d.png"
    outline_plot_path = capture_output / "room_outline_2d.png"
    floor_ceiling_plot_path = capture_output / "floor_ceiling_candidates.png"

    save_plane_plot(points, detected_planes, plane_plot_path)
    save_outline_plot(points, wall_planes, outline, outline_plot_path)
    save_floor_ceiling_plot(points, floor, ceiling, floor_ceiling_plot_path)

    ceiling_height = None

    if floor is not None and ceiling is not None:
        ceiling_height = float(ceiling["height_mean"] - floor["height_mean"])

    report = {
        "capture_name": capture_name,
        "input_point_cloud": str(point_cloud_path),
        "point_count": int(len(points)),
        "planes_detected": len(detected_planes),
        "floor_detected": floor is not None,
        "ceiling_detected": ceiling is not None,
        "wall_plane_count": len(wall_planes),
        "floor_height": floor_height if floor is not None else None,
        "ceiling_height_estimate": ceiling_height,
        "room_outline": outline,
        "plan_status": "diagnostic_only",
        "plan_status_reason": "Pose convention and metric ground truth are not confirmed.",
        "planes": [make_plane_report(plane) for plane in detected_planes],
        "assumptions": {
            "up_axis": UP_AXIS,
            "horizontal_axes": HORIZONTAL_AXES,
            "plane_distance_threshold": PLANE_DISTANCE_THRESHOLD,
            "ransac_iterations": RANSAC_ITERATIONS,
            "minimum_plane_points": MIN_PLANE_POINTS,
            "coordinate_system_status": "provisional",
            "metric_accuracy_status": "not evaluated without physical ground truth",
        },
        "outputs": {
            "detected_planes_3d": str(plane_plot_path),
            "room_outline_2d": str(outline_plot_path),
            "floor_ceiling_candidates": str(floor_ceiling_plot_path),
        },
    }

    return report


def build_report(project_root, output_root):
    """Input: project and output folders; output: Phase 4 report for all captures."""
    capture_reports = []

    for capture_name in CAPTURE_NAMES:
        capture_reports.append(
            build_capture_report(project_root, capture_name, output_root)
        )

    return {
        "report_name": "Agentic-Stitcher Phase 4 provisional room plan extraction",
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
    """Input: no arguments or an optional project path; output: plane images and JSON report."""
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1]).resolve()
    else:
        project_root = Path(__file__).resolve().parent.parent

    phase_four_folder = Path(__file__).resolve().parent
    output_root = phase_four_folder / "outputs"
    report_path = phase_four_folder / "phase_4_report.json"
    report = build_report(project_root, output_root)
    write_report(report, report_path)

    print("Phase 4 room-plan extraction completed.")
    print("Outputs:", output_root)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
