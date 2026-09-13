# Phase 1 — Normalized capture loading

Phase 1 creates one consistent Python representation for each capture folder. Later geometry code can use this representation instead of reading CSV and PNG files directly.

## Dependencies

```bash
python3 -m pip install pandas numpy pillow
```

## Run the validation

From the `Agentic-Stitcher` folder:

```bash
python3 "phase 1/run_phase_1.py"
```

The report is written to:

```text
phase 1/phase_1_report.json
```

## Loader usage

The loader can be used from another Python file like this:

```python
from pathlib import Path

from capture_loader import get_frame, load_capture


capture_path = Path("single_scan_with_ceiling")
capture = load_capture(capture_path)
frame = get_frame(capture, 0)

print(frame["depth"].shape)
print(frame["confidence"].shape)
print(frame["timestamp"])
print(frame["odometry"])
```

`depth` and `confidence` are returned as NumPy arrays without scaling. The depth unit is intentionally not guessed.

## What Phase 1 checks

- All required capture files exist.
- Camera matrix loads as a 3×3 NumPy array.
- IMU and odometry tables load as pandas DataFrames.
- Numeric sensor columns are converted to numeric values.
- Depth and confidence files are indexed by frame ID.
- Depth, confidence, and odometry frame IDs match.
- Sample frames can be loaded from the beginning, middle, and end.
- The nearest IMU sample can be found for an odometry timestamp.
- IMU and odometry timestamps remain monotonic.

## Phase 1 boundary

This phase does not interpret the coordinate system, depth units, quaternion direction, or camera pose convention. It preserves the values and records those items as unresolved metadata.
