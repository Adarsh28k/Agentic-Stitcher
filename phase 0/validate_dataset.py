
"""Validate the supplied Agentic-Stitcher capture folders for Phase 0."""

import json
import re
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


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


def read_csv_rows(file_path):
    """Input: a CSV path; output: a pandas DataFrame with trimmed column names."""
    rows = pd.read_csv(file_path, skipinitialspace=True)
    cleaned_columns = []

    for column_name in rows.columns:
        cleaned_columns.append(str(column_name).strip())

    rows.columns = cleaned_columns
    return rows


def get_csv_headers(file_path):
    """Input: a CSV path; output: the ordered list of column names."""
    empty_dataframe = pd.read_csv(file_path, skipinitialspace=True, nrows=0)
    headers = []

    for column_name in empty_dataframe.columns:
        headers.append(str(column_name).strip())

    return headers


def read_numeric_csv_rows(file_path):
    """Input: a headerless numeric CSV path; output: a NumPy floating-point array."""
    rows = pd.read_csv(file_path, header=None, skipinitialspace=True)
    numeric_values = rows.to_numpy(dtype=float)
    return numeric_values


def get_png_files(folder_path):
    """Input: a PNG folder; output: sorted PNG paths found directly inside it."""
    png_files = []

    if not folder_path.exists():
        return png_files

    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".png":
            png_files.append(file_path)

    png_files.sort()
    return png_files


def get_frame_id(file_path):
    """Input: a frame filename; output: its integer ID, or None when the name is invalid."""
    match = re.fullmatch(r"(\d+)\.png", file_path.name)

    if match is None:
        return None

    return int(match.group(1))


def get_frame_ids(file_paths):
    """Input: PNG paths; output: frame IDs and filenames that could not be parsed."""
    frame_ids = []
    invalid_names = []

    for file_path in file_paths:
        frame_id = get_frame_id(file_path)

        if frame_id is None:
            invalid_names.append(file_path.name)
        else:
            frame_ids.append(frame_id)

    return frame_ids, invalid_names


def check_frame_sequence(frame_ids):
    """Input: frame IDs; output: a report showing duplicates, gaps, and the expected range."""
    sorted_ids = sorted(frame_ids)
    duplicate_ids = []
    missing_ids = []

    previous_id = None

    for frame_id in sorted_ids:
        if previous_id == frame_id and frame_id not in duplicate_ids:
            duplicate_ids.append(frame_id)
        previous_id = frame_id

    if len(sorted_ids) > 0:
        expected_ids = set(range(sorted_ids[0], sorted_ids[-1] + 1))
        actual_ids = set(sorted_ids)
        missing_ids = sorted(expected_ids - actual_ids)

    return {
        "count": len(frame_ids),
        "first_id": sorted_ids[0] if sorted_ids else None,
        "last_id": sorted_ids[-1] if sorted_ids else None,
        "duplicate_ids": duplicate_ids,
        "missing_ids": missing_ids,
        "starts_at_zero": bool(sorted_ids) and sorted_ids[0] == 0,
    }


def read_png_header(file_path):
    """Input: a PNG path; output: PNG width, height, bit depth, and color type."""
    png_signature = b"\x89PNG\r\n\x1a\n"

    with file_path.open("rb") as png_file:
        signature = png_file.read(8)

        if signature != png_signature:
            return {"valid_png_signature": False}

        chunk_length_bytes = png_file.read(4)
        chunk_type = png_file.read(4)

        if len(chunk_length_bytes) != 4 or chunk_type != b"IHDR":
            return {"valid_png_signature": True, "has_ihdr": False}

        chunk_length = struct.unpack(">I", chunk_length_bytes)[0]
        ihdr_data = png_file.read(chunk_length)

    if len(ihdr_data) < 13:
        return {"valid_png_signature": True, "has_ihdr": False}

    width, height, bit_depth, color_type = struct.unpack(">IIBB", ihdr_data[:10])

    return {
        "valid_png_signature": True,
        "has_ihdr": True,
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
    }


def summarize_timestamps(rows, column_name):
    """Input: a DataFrame and timestamp column; output: timestamp range and monotonicity details."""
    if column_name not in rows.columns:
        return {
            "count": 0,
            "invalid_values": [],
            "first": None,
            "last": None,
            "span_seconds": None,
            "monotonic_non_decreasing": False,
            "decrease_indices": [],
            "minimum_interval_seconds": None,
            "maximum_interval_seconds": None,
            "average_interval_seconds": None,
        }

    timestamp_series = pd.to_numeric(rows[column_name], errors="coerce")
    invalid_values = rows.loc[timestamp_series.isna(), column_name].tolist()
    timestamps = timestamp_series.dropna().to_numpy(dtype=float)
    intervals = np.diff(timestamps)
    decrease_indices = np.where(intervals < 0)[0] + 1

    summary = {
        "count": len(timestamps),
        "invalid_values": invalid_values,
        "first": float(timestamps[0]) if len(timestamps) > 0 else None,
        "last": float(timestamps[-1]) if len(timestamps) > 0 else None,
        "span_seconds": None,
        "monotonic_non_decreasing": len(decrease_indices) == 0,
        "decrease_indices": decrease_indices.tolist(),
        "minimum_interval_seconds": None,
        "maximum_interval_seconds": None,
        "average_interval_seconds": None,
    }

    if len(timestamps) > 1:
        summary["span_seconds"] = float(timestamps[-1] - timestamps[0])
        summary["minimum_interval_seconds"] = float(np.min(intervals))
        summary["maximum_interval_seconds"] = float(np.max(intervals))
        summary["average_interval_seconds"] = float(np.mean(intervals))

    return summary


def get_numeric_value(row, column_name):
    """Input: a pandas row and column name; output: a float value or None."""
    if column_name not in row.index:
        return None

    value = pd.to_numeric(row[column_name], errors="coerce")

    if pd.isna(value):
        return None

    return float(value)


def compare_intrinsics(camera_matrix_rows, odometry_rows):
    """Input: a NumPy matrix and odometry DataFrame; output: intrinsic values and differences."""
    matrix_values = np.asarray(camera_matrix_rows, dtype=float).reshape(-1)

    matrix_fx = matrix_values[0] if len(matrix_values) > 0 else None
    matrix_fy = matrix_values[4] if len(matrix_values) > 4 else None
    matrix_cx = matrix_values[2] if len(matrix_values) > 2 else None
    matrix_cy = matrix_values[5] if len(matrix_values) > 5 else None

    first_odometry = odometry_rows.iloc[0] if len(odometry_rows) > 0 else pd.Series(dtype=float)
    last_odometry = odometry_rows.iloc[-1] if len(odometry_rows) > 0 else pd.Series(dtype=float)

    result = {
        "camera_matrix": {
            "fx": matrix_fx,
            "fy": matrix_fy,
            "cx": matrix_cx,
            "cy": matrix_cy,
        },
        "odometry_first_row": {},
        "odometry_last_row": {},
    }

    for label, row in [
        ("odometry_first_row", first_odometry),
        ("odometry_last_row", last_odometry),
    ]:
        result[label] = {
            "fx": get_numeric_value(row, "fx"),
            "fy": get_numeric_value(row, "fy"),
            "cx": get_numeric_value(row, "cx"),
            "cy": get_numeric_value(row, "cy"),
        }

    return result


def compare_frame_sources(depth_files, confidence_files, odometry_rows):
    """Input: PNG files and odometry DataFrame; output: frame alignment checks."""
    depth_ids, depth_invalid_names = get_frame_ids(depth_files)
    confidence_ids, confidence_invalid_names = get_frame_ids(confidence_files)

    odometry_frame_values = pd.to_numeric(odometry_rows["frame"], errors="coerce")
    invalid_odometry_frames = odometry_rows.loc[
        odometry_frame_values.isna(), "frame"
    ].tolist()
    odometry_ids = odometry_frame_values.dropna().astype(int).tolist()

    depth_set = set(depth_ids)
    confidence_set = set(confidence_ids)
    odometry_set = set(odometry_ids)

    return {
        "depth_sequence": check_frame_sequence(depth_ids),
        "confidence_sequence": check_frame_sequence(confidence_ids),
        "odometry_sequence": check_frame_sequence(odometry_ids),
        "invalid_depth_names": depth_invalid_names,
        "invalid_confidence_names": confidence_invalid_names,
        "invalid_odometry_frames": invalid_odometry_frames,
        "depth_and_confidence_match": depth_set == confidence_set,
        "depth_and_odometry_match": depth_set == odometry_set,
        "confidence_and_odometry_match": confidence_set == odometry_set,
    }


def inspect_image_examples(depth_files, confidence_files):
    """Input: depth and confidence file lists; output: PNG metadata for representative frames."""
    examples = {}

    selected_depth_files = []
    selected_confidence_files = []

    if depth_files:
        selected_depth_files.append(depth_files[0])
        selected_depth_files.append(depth_files[len(depth_files) // 2])
        selected_depth_files.append(depth_files[-1])

    if confidence_files:
        selected_confidence_files.append(confidence_files[0])
        selected_confidence_files.append(confidence_files[len(confidence_files) // 2])
        selected_confidence_files.append(confidence_files[-1])

    for file_path in selected_depth_files:
        key = "depth/" + file_path.name
        examples[key] = read_png_header(file_path)

    for file_path in selected_confidence_files:
        key = "confidence/" + file_path.name
        examples[key] = read_png_header(file_path)

    return examples


def build_capture_report(capture_path):
    """Input: one capture folder; output: a complete Phase 0 report for that capture."""
    camera_matrix_path = capture_path / "camera_matrix.csv"
    imu_path = capture_path / "imu.csv"
    odometry_path = capture_path / "odometry.csv"
    depth_path = capture_path / "depth"
    confidence_path = capture_path / "confidence"
    video_path = capture_path / "rgb.mp4"

    camera_matrix_rows = []

    if camera_matrix_path.exists():
        camera_matrix_rows = read_numeric_csv_rows(camera_matrix_path)

    imu_rows = []
    if imu_path.exists():
        imu_rows = read_csv_rows(imu_path)

    odometry_rows = []
    if odometry_path.exists():
        odometry_rows = read_csv_rows(odometry_path)

    depth_files = get_png_files(depth_path)
    confidence_files = get_png_files(confidence_path)

    report = {
        "capture_name": capture_path.name,
        "capture_path": str(capture_path),
        "files_present": {
            "camera_matrix_csv": camera_matrix_path.exists(),
            "imu_csv": imu_path.exists(),
            "odometry_csv": odometry_path.exists(),
            "depth_folder": depth_path.exists(),
            "confidence_folder": confidence_path.exists(),
            "rgb_mp4": video_path.exists(),
        },
        "csv_headers": {
            "camera_matrix": [],
            "imu": get_csv_headers(imu_path) if imu_path.exists() else [],
            "odometry": get_csv_headers(odometry_path) if odometry_path.exists() else [],
        },
        "row_counts": {
            "camera_matrix": len(camera_matrix_rows),
            "imu": len(imu_rows),
            "odometry": len(odometry_rows),
        },
        "timestamps": {
            "imu": summarize_timestamps(imu_rows, "timestamp"),
            "odometry": summarize_timestamps(odometry_rows, "timestamp"),
        },
        "frame_alignment": compare_frame_sources(depth_files, confidence_files, odometry_rows),
        "image_examples": inspect_image_examples(depth_files, confidence_files),
        "intrinsics": compare_intrinsics(camera_matrix_rows, odometry_rows),
        "video": {
            "exists": video_path.exists(),
            "size_bytes": video_path.stat().st_size if video_path.exists() else None,
            "frame_timestamp_mapping_verified": False,
            "note": "The video file exists, but this validator does not decode its frame metadata yet.",
        },
        "unresolved_assumptions": [
            "Depth PNG units are not documented in the supplied files.",
            "Camera and world coordinate conventions are not documented in the supplied files.",
            "Quaternion convention and pose direction are not documented in the supplied files.",
            "The exact RGB-video to odometry timestamp mapping is not verified here.",
        ],
    }

    return report


def build_dataset_report(project_root):
    """Input: the project folder; output: the complete dataset report for all expected captures."""
    capture_reports = []

    for capture_name in CAPTURE_NAMES:
        capture_path = project_root / capture_name

        if capture_path.exists():
            capture_reports.append(build_capture_report(capture_path))
        else:
            capture_reports.append(
                {
                    "capture_name": capture_name,
                    "capture_path": str(capture_path),
                    "missing_capture_folder": True,
                }
            )

    report = {
        "report_name": "Agentic-Stitcher Phase 0 dataset validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "confirmed_facts": [
            "The expected capture folders were checked.",
            "Depth and confidence files are checked by PNG header metadata.",
            "Depth, confidence, and odometry frame IDs are compared.",
            "IMU and odometry timestamp ordering is checked.",
        ],
        "capture_reports": capture_reports,
    }

    return report


def write_report(report, output_path):
    """Input: a report dictionary and output path; output: a formatted JSON report on disk."""
    with output_path.open("w") as json_file:
        json.dump(report, json_file, indent=2)
        json_file.write("\n")


def main():
    """Input: optional project path argument; output: validation report and process status."""
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1]).resolve()
    else:
        project_root = get_project_root(Path(__file__))

    output_path = Path(__file__).resolve().parent / "phase_0_report.json"
    report = build_dataset_report(project_root)
    write_report(report, output_path)

    print("Phase 0 validation completed.")
    print("Project:", project_root)
    print("Report:", output_path)


if __name__ == "__main__":
    main()
