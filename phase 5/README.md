# Phase 5 — Cross-capture robustness evaluation

## Goal

Phase 5 checks how the same Phase 4 room-plan pipeline behaves on all three supplied captures.

It does not rebuild the point clouds and it does not change Phase 3 or Phase 4. It reads their existing JSON reports and compares the results using the same capture roles:

| Capture | Role |
|---|---|
| `single_scan_with_ceiling` | Development capture with the richest coverage |
| `single_scan_floor_only` | Robustness capture with limited ceiling coverage |
| `single_room` | Held-out smoke test with a different trajectory |

These are not three identical repetitions. They cover the same room differently, so this is a robustness comparison, not a formal repeatability evaluation.

## Inputs

- `phase 3/phase_3_report.json`
- `phase 4/phase_4_report.json`
- Existing Phase 3 and Phase 4 output paths referenced by those reports

The evaluator uses the results already produced by Phase 4. It does not silently rerun Phase 4 with different settings.

## What is compared

For each capture, the report records:

- Number of reconstructed points
- Number of detected planes
- Number of candidate wall planes
- Whether a floor was detected
- Whether a ceiling was detected
- Estimated ceiling height, when available
- Estimated room-outline area and perimeter
- Number of outline vertices
- Whether the expected diagnostic files exist

It also compares the provisional point-cloud bounds from Phase 3. These values are useful for finding large differences, but they are not proof of physical accuracy because the pose convention and depth scale are still provisional.

## Expected outputs

The script creates:

- `phase_5_report.json`: machine-readable comparison and evaluation status
- `capture_summary.csv`: easy-to-read table of all capture metrics
- `plan_metrics_comparison.png`: side-by-side metric charts
- `room_outline_comparison.png`: overlay of available provisional outlines
- `point_cloud_bounds_comparison.png`: provisional horizontal coverage comparison

## How to read the images

### `plan_metrics_comparison.png`

Each color represents one capture. Similar values can suggest stable behavior, while large differences suggest sensitivity to capture coverage or pose drift.

The charts do not mean the values are correct. For example, a similar estimated area across captures is encouraging, but it is not a substitute for measuring the room with a tape or laser.

### `room_outline_comparison.png`

The colored lines are Phase 4's provisional convex-hull outlines. Look for:

- Similar overall shape
- Similar relative width and length
- Large isolated spikes
- Self-intersections or disconnected-looking outlines
- One capture being much larger than the others

The axes are still provisional world coordinates.

### `point_cloud_bounds_comparison.png`

Each rectangle shows the horizontal bounding box of a Phase 3 point cloud. This is only a coverage diagnostic. A larger rectangle may mean a longer camera path, accumulated drift, or genuinely wider coverage.

## Evaluation rules

Phase 5 reports `diagnostic_only` until physical measurements are available. It does not claim that the assignment's accuracy targets have been met.

Formal evaluation still requires:

- Wall and room measurements
- Ceiling height measurements
- Door and window measurements
- At least two comparable captures for repeatability

No trained vision model, annotation set, or GPU is required for this phase. It uses pandas, NumPy, and Matplotlib to compare existing geometric outputs.

## Run

From the `Agentic-Stitcher` folder:

```bash
python3 "phase 5/evaluate_robustness.py"
```

Outputs are written under:

```text
phase 5/outputs/
```

