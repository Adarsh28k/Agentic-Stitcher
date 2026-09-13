# Phase 0 — Dataset validation

This phase does not reconstruct the room yet. It answers a simpler question:

> Are the supplied files present, readable, and aligned well enough to begin reconstruction?

The validator uses `pandas` for CSV files and `numpy` for numeric checks.

If they are not already installed, install them with:

```bash
python3 -m pip install pandas numpy
```

## Run it

From the `Agentic-Stitcher` folder:

```bash
python3 "phase 0/validate_dataset.py"
```

The validator writes:

```text
phase 0/phase_0_report.json
```

You can also provide the project folder explicitly:

```bash
python3 "phase 0/validate_dataset.py" "/path/to/Agentic-Stitcher"
```

## What it checks

- Required CSV, PNG, and video files
- CSV headers and row counts
- Depth, confidence, and odometry frame IDs
- Missing or duplicate frame IDs
- PNG dimensions, bit depth, and color type
- IMU and odometry timestamp ordering
- Camera-matrix values versus odometry intrinsics
- Presence and size of each RGB video

## What it cannot confirm yet

The validator cannot determine from these files alone:

- Whether depth values are metres or millimetres
- The camera/world coordinate convention
- Quaternion interpretation and pose direction
- Exact RGB-video frame timestamps
- Whether the odometry pose is already in the desired world frame

Those items are recorded as unresolved assumptions in the JSON report. They should be resolved before the point-cloud reconstruction phase.

## Expected Phase 0 result

Phase 0 is complete when:

1. All three folders pass the structural checks.
2. Depth, confidence, and odometry frame IDs match.
3. Timestamps are monotonic.
4. The unresolved assumptions have documented answers or an explicit test plan.
5. The report can be regenerated with one command.
