#!/usr/bin/env python3
"""Load one Agentic-Stitcher capture into a consistent Python structure."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class CaptureData:
    """The files and tables belonging to one capture folder."""

    name: str
    root: Path
    camera_matrix: np.ndarray
    imu: pd.DataFrame
    odometry: pd.DataFrame
    depth_files: dict
    confidence_files: dict
    rgb_video: Path
    metadata: dict


def read_camera_matrix(file_path):
    """Input: a headerless camera-matrix CSV; output: a 3x3 NumPy array."""
    matrix_table = pd.read_csv(file_path, header=None, dtype=float)
    matrix = matrix_table.to_numpy(dtype=np.float64)

    if matrix.shape != (3, 3):
        raise ValueError("Camera matrix must have shape (3, 3): " + str(file_path))

    return matrix


def read_table(file_path):
    """Input: a CSV path with a header; output: a cleaned pandas DataFrame."""
    table = pd.read_csv(file_path, skipinitialspace=True)
    cleaned_columns = []

    for column_name in table.columns:
        cleaned_columns.append(str(column_name).strip())

    table.columns = cleaned_columns
    return table


def index_png_files(folder_path):
    """Input: a PNG folder; output: a dictionary mapping frame IDs to PNG paths."""
    frame_files = {}
    frame_pattern = re.compile(r"(\d+)\.png")

    if not folder_path.exists():
        raise FileNotFoundError("Missing image folder: " + str(folder_path))

    for file_path in sorted(folder_path.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() != ".png":
            continue

        match = frame_pattern.fullmatch(file_path.name)

        if match is None:
            raise ValueError("Image filename is not a numeric frame: " + str(file_path))

        frame_id = int(match.group(1))

        if frame_id in frame_files:
            raise ValueError("Duplicate frame ID in image folder: " + str(frame_id))

        frame_files[frame_id] = file_path

    return frame_files


def add_frame_id_column(odometry):
    """Input: an odometry DataFrame; output: the same table with an integer frame_id column."""
    if "frame" not in odometry.columns:
        raise ValueError("Odometry CSV does not contain a frame column")

    odometry["frame_id"] = pd.to_numeric(odometry["frame"], errors="coerce")

    if odometry["frame_id"].isna().any():
        raise ValueError("Odometry contains an invalid frame ID")

    odometry["frame_id"] = odometry["frame_id"].astype(int)
    return odometry


def clean_numeric_columns(odometry, imu):
    """Input: odometry and IMU tables; output: tables with numeric sensor columns."""
    odometry_columns = [
        "timestamp",
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
        "fx",
        "fy",
        "cx",
        "cy",
    ]

    imu_columns = [
        "timestamp",
        "a_x",
        "a_y",
        "a_z",
        "alpha_x",
        "alpha_y",
        "alpha_z",
    ]

    for column_name in odometry_columns:
        if column_name in odometry.columns:
            odometry[column_name] = pd.to_numeric(odometry[column_name], errors="coerce")

    for column_name in imu_columns:
        if column_name in imu.columns:
            imu[column_name] = pd.to_numeric(imu[column_name], errors="coerce")

    return odometry, imu


def make_metadata(root, odometry, imu, depth_files, confidence_files):
    """Input: capture paths and loaded tables; output: basic normalized capture metadata."""
    metadata = {
        "depth_units": "unknown",
        "coordinate_convention": "unknown",
        "quaternion_convention": "unknown",
        "pose_direction": "unknown",
        "frame_count_depth": len(depth_files),
        "frame_count_confidence": len(confidence_files),
        "frame_count_odometry": len(odometry),
        "imu_row_count": len(imu),
        "project_path": str(root),
    }

    return metadata


def load_capture(capture_path):
    """Input: one capture folder; output: a CaptureData object with indexed files and tables."""
    capture_path = Path(capture_path).resolve()
    camera_matrix_path = capture_path / "camera_matrix.csv"
    imu_path = capture_path / "imu.csv"
    odometry_path = capture_path / "odometry.csv"
    depth_path = capture_path / "depth"
    confidence_path = capture_path / "confidence"
    rgb_video_path = capture_path / "rgb.mp4"

    required_paths = [
        camera_matrix_path,
        imu_path,
        odometry_path,
        depth_path,
        confidence_path,
        rgb_video_path,
    ]

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError("Missing capture item: " + str(required_path))

    camera_matrix = read_camera_matrix(camera_matrix_path)
    imu = read_table(imu_path)
    odometry = read_table(odometry_path)
    odometry = add_frame_id_column(odometry)
    odometry, imu = clean_numeric_columns(odometry, imu)
    depth_files = index_png_files(depth_path)
    confidence_files = index_png_files(confidence_path)
    metadata = make_metadata(capture_path, odometry, imu, depth_files, confidence_files)

    capture = CaptureData(
        name=capture_path.name,
        root=capture_path,
        camera_matrix=camera_matrix,
        imu=imu,
        odometry=odometry,
        depth_files=depth_files,
        confidence_files=confidence_files,
        rgb_video=rgb_video_path,
        metadata=metadata,
    )

    return capture


def read_image(file_path):
    """Input: a PNG path; output: the image pixels as a NumPy array without scaling."""
    with Image.open(file_path) as image:
        pixels = np.array(image)

    return pixels


def get_odometry_row(capture, frame_id):
    """Input: a CaptureData object and frame ID; output: the matching odometry row."""
    matching_rows = capture.odometry[capture.odometry["frame_id"] == frame_id]

    if len(matching_rows) == 0:
        raise KeyError("No odometry row found for frame: " + str(frame_id))

    if len(matching_rows) > 1:
        raise ValueError("More than one odometry row found for frame: " + str(frame_id))

    return matching_rows.iloc[0]


def get_frame(capture, frame_id):
    """Input: a CaptureData object and frame ID; output: depth, confidence, pose, and timestamp data."""
    if frame_id not in capture.depth_files:
        raise KeyError("No depth image found for frame: " + str(frame_id))

    if frame_id not in capture.confidence_files:
        raise KeyError("No confidence image found for frame: " + str(frame_id))

    odometry_row = get_odometry_row(capture, frame_id)
    depth = read_image(capture.depth_files[frame_id])
    confidence = read_image(capture.confidence_files[frame_id])

    frame = {
        "frame_id": frame_id,
        "depth": depth,
        "confidence": confidence,
        "odometry": odometry_row,
        "timestamp": float(odometry_row["timestamp"]),
        "camera_matrix": capture.camera_matrix,
    }

    return frame


def get_nearest_imu_sample(capture, timestamp):
    """Input: a CaptureData object and timestamp; output: the nearest IMU row."""
    if "timestamp" not in capture.imu.columns:
        raise ValueError("IMU table does not contain a timestamp column")

    imu_timestamps = capture.imu["timestamp"].to_numpy(dtype=float)
    differences = np.abs(imu_timestamps - timestamp)
    nearest_index = int(np.argmin(differences))
    nearest_row = capture.imu.iloc[nearest_index]
    return nearest_row


def frame_ids_match(capture):
    """Input: a CaptureData object; output: whether depth, confidence, and odometry IDs match."""
    depth_ids = set(capture.depth_files.keys())
    confidence_ids = set(capture.confidence_files.keys())
    odometry_ids = set(capture.odometry["frame_id"].tolist())

    return depth_ids == confidence_ids and depth_ids == odometry_ids


def timestamps_are_monotonic(table):
    """Input: a sensor DataFrame; output: whether its timestamps never decrease."""
    if "timestamp" not in table.columns:
        return False

    timestamps = table["timestamp"].dropna().to_numpy(dtype=float)

    if len(timestamps) < 2:
        return True

    differences = np.diff(timestamps)
    return bool(np.all(differences >= 0))


def validate_sample_frame(capture, frame_id):
    """Input: a CaptureData object and frame ID; output: shape, dtype, and IMU lookup checks."""
    frame = get_frame(capture, frame_id)
    nearest_imu = get_nearest_imu_sample(capture, frame["timestamp"])
    imu_time_difference = abs(float(nearest_imu["timestamp"]) - frame["timestamp"])

    result = {
        "frame_id": frame_id,
        "depth_shape": list(frame["depth"].shape),
        "depth_dtype": str(frame["depth"].dtype),
        "confidence_shape": list(frame["confidence"].shape),
        "confidence_dtype": str(frame["confidence"].dtype),
        "odometry_timestamp": frame["timestamp"],
        "nearest_imu_time_difference_seconds": imu_time_difference,
    }

    return result


def validate_capture(capture):
    """Input: a CaptureData object; output: a Phase 1 validation report for that capture."""
    frame_ids = sorted(capture.depth_files.keys())
    sample_ids = []

    if len(frame_ids) > 0:
        sample_ids.append(frame_ids[0])
        sample_ids.append(frame_ids[len(frame_ids) // 2])
        sample_ids.append(frame_ids[-1])

    sample_results = []

    for frame_id in sample_ids:
        sample_results.append(validate_sample_frame(capture, frame_id))

    result = {
        "capture_name": capture.name,
        "capture_path": str(capture.root),
        "frame_ids_match": frame_ids_match(capture),
        "depth_frame_count": len(capture.depth_files),
        "confidence_frame_count": len(capture.confidence_files),
        "odometry_row_count": len(capture.odometry),
        "imu_row_count": len(capture.imu),
        "imu_timestamps_monotonic": timestamps_are_monotonic(capture.imu),
        "odometry_timestamps_monotonic": timestamps_are_monotonic(capture.odometry),
        "camera_matrix_shape": list(capture.camera_matrix.shape),
        "sample_frames": sample_results,
        "metadata": capture.metadata,
    }

    return result


def validate_all_captures(project_root, capture_names):
    """Input: project folder and capture names; output: validation reports for every capture."""
    reports = []

    for capture_name in capture_names:
        capture_path = project_root / capture_name
        capture = load_capture(capture_path)
        report = validate_capture(capture)
        reports.append(report)

    return reports

