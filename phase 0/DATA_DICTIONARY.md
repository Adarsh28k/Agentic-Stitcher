# Phase 0 Data Dictionary

This document records what we know about the supplied files before building geometry.

## Confirmed from the files

| Item | Finding |
|---|---|
| Capture folders | `single_room`, `single_scan_floor_only`, `single_scan_with_ceiling` |
| Depth format | PNG, grayscale, 16-bit, 256×192 |
| Confidence format | PNG, grayscale, 8-bit, 256×192 |
| Depth frame names | Zero-padded numeric names such as `000000.png` |
| Confidence frame names | Same naming pattern as depth |
| Odometry frame names | Numeric frame IDs matching depth and confidence IDs |
| Odometry columns | Timestamp, frame, position, quaternion, and camera intrinsics |
| IMU columns | Timestamp, acceleration values, and angular-rate values |
| Camera matrix | 3×3 numeric matrix in each capture folder |
| Timestamp order | IMU and odometry timestamps are monotonic in all three folders |
| Frame alignment | Depth, confidence, and odometry frame IDs match in all three folders |
| RGB input | An `rgb.mp4` file exists in every capture folder |

## Current capture summaries

| Capture | Odometry frames | IMU rows | Odometry span |
|---|---:|---:|---:|
| `single_room` | 1,715 | 3,689 | ~37.17 seconds |
| `single_scan_floor_only` | 5,251 | 11,397 | ~114.78 seconds |
| `single_scan_with_ceiling` | 9,745 | 21,339 | ~214.93 seconds |

## Unresolved items

### Depth units

The depth values are numerically consistent with millimetres, but the files do not state the unit. We must confirm this before calculating dimensions.

### Coordinate convention

The files do not state which camera axis points forward, which axis points up, or whether positions describe camera-in-world or world-in-camera transforms.

### Quaternion convention

The files do not state whether the quaternion is ordered as `(qx, qy, qz, qw)` or another order, or whether it represents a camera-to-world or world-to-camera rotation.

### IMU units and frame

The column names identify acceleration and angular-rate values, but the units and sensor coordinate frame are not documented.

### RGB video timing

The video file exists, but the exact relationship between RGB video frames and the numbered depth/odometry frames has not yet been verified.

### Intrinsic source

Each folder has a static `camera_matrix.csv`, while `odometry.csv` also contains per-frame `fx`, `fy`, `cx`, and `cy`. The values differ slightly at the beginning of a capture. This needs a deliberate choice or a documented interpolation rule.

## Phase 0 decision needed before point-cloud work

We need either documentation from the capture/export tool or a small controlled test that resolves the coordinate and unit questions. Without that, a point cloud can still be rendered, but its scale or orientation may be wrong.
