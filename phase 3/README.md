# Phase 3 — Point-cloud reconstruction

## Main idea

Phase 3 converts the numeric depth images into 3D points.

For a pixel at image position `(u, v)` with depth `d`, and camera intrinsics `fx`, `fy`, `cx`, and `cy`, the first camera-space estimate is:

```text
X = (u - cx) × d / fx
Y = (v - cy) × d / fy
Z = d
```

The pipeline then uses the odometry position and quaternion to move those camera-space points into a shared world-space coordinate system.

## What this phase produces

For every capture, the script creates:

- A single-frame 3D point-cloud image
- A multi-frame accumulated point-cloud image
- A confidence-filter comparison image
- A camera-and-point-cloud image
- An ASCII `.ply` point-cloud file
- A JSON summary containing point counts, bounds, and assumptions

The output is an early geometric reconstruction, not yet a finished floor plan.

## Important assumptions

The supplied files do not document the depth unit or pose convention. This first implementation makes the following visible, provisional assumptions:

1. Depth values are converted using `0.001` so values that look like millimetres become pose-scale units.
2. The intrinsics describe a 1920×1440 image, so `fx`, `fy`, `cx`, and `cy` are scaled to the 256×192 depth-image resolution.
3. The camera model uses `x` right, `y` down, and `z` forward.
4. The quaternion columns are ordered `(qx, qy, qz, qw)`.
5. The quaternion rotates camera-space points into the odometry/world frame.
6. The translation columns use the same scale as the converted depth.

These assumptions are recorded in the report. The point cloud must not be treated as metrically correct until they are confirmed.

## Libraries

This implementation uses:

- `numpy` for projection and transformations
- `pandas` through the Phase 1 loader
- `Pillow` through the Phase 1 loader
- `matplotlib` for static 3D images
- `open3d` for point-cloud objects, voxel downsampling, statistical filtering, and `.ply` export

Install the Phase 3 dependencies with:

```bash
python3 -m pip install pandas numpy pillow matplotlib open3d
```

The generated `.ply` files can also be opened in CloudCompare, MeshLab, or another point-cloud viewer.

## Run it

From the `Agentic-Stitcher` folder:

```bash
python3 "phase 3/build_point_cloud.py"
```

The outputs are written under:

```text
phase 3/outputs/<capture-name>/
```

The overall report is:

```text
phase 3/phase_3_report.json
```

## How to understand the images

### Single-frame point cloud

Open `single_frame_camera_space.png` first. Each dot is a valid depth pixel converted into 3D. This shows what one camera view contributes before odometry is used.

### Accumulated point cloud

Open `accumulated_world_space.png`. This combines selected frames after applying their poses. A successful result should begin to show coherent surfaces instead of unrelated clouds.

### Confidence comparison

Open `confidence_filter_comparison.png`. The left side uses valid depth values before the confidence filter. The right side uses only pixels at or above the chosen confidence threshold. The right side should contain fewer, more trustworthy points.

### Camera and point cloud

Open `camera_and_point_cloud.png`. The colored cloud shows accumulated points and the red line shows the camera trajectory. This helps reveal whether the pose path and reconstructed surfaces have compatible scale and orientation.

### PLY file

The `.ply` file contains the accumulated 3D points. It can be opened in a dedicated point-cloud viewer. Its coordinates currently use provisional units.

## Evaluation for this phase

We are not claiming final accuracy yet. We will check:

- A single frame produces non-empty 3D points.
- Confidence filtering reduces the point count.
- The point cloud has finite, bounded coordinates.
- Selected frames accumulate without immediate numerical explosions.
- Surfaces look more coherent than a random scatter.
- The camera trajectory and point-cloud scale are visually compatible.

The next phase can use this point cloud to detect floor, ceiling, and wall planes.

## Current Phase 3 interpretation

The single-frame reconstruction is the most reliable result at this stage: it confirms that depth pixels and scaled camera intrinsics can produce a sensible local 3D surface.

The accumulated reconstruction is still provisional. Its scattered regions indicate that the pose convention, odometry direction, or drift handling needs more work before it can be used for room dimensions. This is expected at this phase and is why we keep the raw and filtered outputs instead of presenting the result as a finished scan.

## Beginner explanation: what is a point cloud?

A point cloud is a collection of 3D points. Each point has three coordinates:

```text
(X, Y, Z)
```

For this project, each point starts as one pixel in a depth image. The depth image tells us how far that pixel is from the camera. After combining the depth value with the camera calibration, that pixel becomes a 3D point.

For example, a wall may produce thousands of points that all lie close to the same mathematical plane. A floor also produces a large, mostly flat group of points. Later phases will use these groups to estimate walls, floors, ceilings, and openings.

A point cloud is not yet a floor plan. It is the raw 3D material from which a floor plan can be extracted.

## What Phase 3 is doing

Phase 3 follows this sequence:

```text
Depth pixel
    ↓
3D point in the camera's coordinate system
    ↓
Move the point using the camera pose
    ↓
Repeat for many frames
    ↓
One accumulated point cloud for the room
```

The reason for doing this is that one depth frame sees only part of the room. As the camera moves, every frame sees a different part. Odometry tells us where the camera was for each frame, so we can place all those partial views into one shared coordinate system.

## Basic mathematics

### Camera intrinsics

The camera matrix gives four important values:

```text
fx, fy = focal lengths
cx, cy = optical center
```

These describe how the camera turns 3D rays into image pixels.

The supplied intrinsics describe a 1920×1440 image, but the depth PNGs are 256×192. We therefore scale the four values to the depth-image resolution before using them. Without this step, the point cloud becomes badly stretched because the camera calibration and depth pixels use different image sizes.

### Back-projecting a depth pixel

For a pixel at column `u`, row `v`, with depth `d`, the code uses:

```text
X = (u - cx) × d / fx
Y = (v - cy) × d / fy
Z = d
```

This is called back-projection or unprojection. It changes a 2D image measurement into a 3D camera-space point.

The current code uses a provisional depth scale of `0.001`. This treats values such as `2500` as approximately `2.5` pose-scale units. It is likely to be millimetres converted to metres, but the source files do not confirm that yet.

### Camera space and world space

Camera space is the coordinate system attached to the phone. The camera is at the origin, and the depth points are measured relative to it.

World space is a shared coordinate system for the entire scan. Each frame has a position and orientation from `odometry.csv`. We use those values to transform camera-space points into world space.

The rotation is represented by a quaternion:

```text
(qx, qy, qz, qw)
```

The code converts this quaternion into a 3×3 rotation matrix `R`, then applies:

```text
world_point = R × camera_point + translation
```

The exact quaternion convention and pose direction are not documented in the supplied data, so this transformation is still provisional.

## What sensor cleaning means here

The cleaning in Phase 3 is basic preprocessing. It does not repair incorrect odometry or magically improve the sensor. It removes measurements that are clearly unusable and reduces unnecessary point density.

### 1. Invalid depth removal

Depth values equal to zero are ignored because they do not describe a valid distance.

### 2. Confidence filtering

The confidence map contains values `0`, `1`, and `2`:

```text
0 = invalid
1 = medium confidence
2 = strongest confidence
```

The current accumulation keeps values at or above confidence `1`. This removes pixels explicitly marked invalid while retaining medium-confidence measurements for now.

### 3. Pixel stride

The code samples every fourth pixel during multi-frame accumulation. This is called subsampling or a pixel stride. It reduces computation and file size while preserving the large room surfaces needed for an initial diagnostic.

### 4. Voxel downsampling

Open3D groups nearby points into small 3D cubes called voxels and keeps one representative point per cube. The current provisional voxel size is `0.02` pose-scale units.

This reduces duplicate points from nearby frames. It is similar to reducing image resolution, but it happens in 3D space.

### 5. Statistical outlier removal

Some points may appear far away from the surfaces around them because of sensor noise or bad depth. Open3D compares each point with nearby points and removes points that are unusually isolated.

The output files preserve both the raw accumulated cloud and the filtered cloud so we can see what was removed.

## How to read the generated images

Use the `single_scan_with_ceiling` output first:

- [Single-frame camera-space cloud](<outputs/single_scan_with_ceiling/single_frame_camera_space.png>)
- [Confidence filter comparison](<outputs/single_scan_with_ceiling/confidence_filter_comparison.png>)
- [Raw accumulated cloud](<outputs/single_scan_with_ceiling/accumulated_world_space_raw.png>)
- [Filtered accumulated cloud](<outputs/single_scan_with_ceiling/accumulated_world_space_filtered.png>)
- [Camera and cloud together](<outputs/single_scan_with_ceiling/camera_and_point_cloud.png>)

### Single-frame camera-space cloud

This is the easiest image to understand. It shows what one depth image becomes after back-projection. The points should form a recognizable local surface rather than a random scatter.

The axes are camera coordinates. They do not describe the whole room yet.

### Confidence filter comparison

The left side uses every non-zero depth pixel. The right side also applies the confidence threshold. The right side should have fewer points, especially where the sensor marked measurements as invalid.

If both sides look nearly identical, that means most pixels already have acceptable confidence. That is what happens in many of these sample frames.

### Raw accumulated cloud

This combines selected frames after applying the provisional odometry poses. It shows the direct result before Open3D filtering.

Look for repeated surfaces, streaks, duplicate layers, and separate clouds. Repeated layers usually indicate pose error, scale mismatch, or drift.

### Filtered accumulated cloud

This is the same accumulated cloud after voxel downsampling and statistical outlier removal. It should look less noisy, but cleaning cannot fix a wrong pose convention. A smooth-looking but incorrectly placed cloud is still wrong.

### Camera and cloud together

The red line is the camera trajectory. The green point marks the beginning and the red point marks the end. The colored points are the reconstructed surfaces.

This image helps answer whether the camera movement and reconstructed geometry seem to share the same scale and orientation. If the cloud is far away from the trajectory or surfaces are spread into unrelated layers, pose interpretation needs investigation.

## What the current result tells us

The single-frame image shows that the depth pixels and resolution-scaled intrinsics can produce a coherent local 3D surface.

The multi-frame image is more scattered. That is evidence that at least one of the following still needs confirmation:

- Quaternion direction
- Camera-to-world versus world-to-camera pose interpretation
- Coordinate-axis convention
- Odometry drift
- Exact depth scale

This is a useful result, not a failure. Phase 3 has isolated the next technical problem: frame projection works better than multi-frame pose accumulation.

## Small terminology glossary

| Term | Meaning |
|---|---|
| Depth map | An image where each pixel stores distance instead of color |
| Intrinsics | Camera calibration values describing focal length and optical center |
| Back-projection | Converting a depth pixel into a 3D camera-space point |
| Point cloud | A collection of 3D points without surfaces or mesh faces |
| Pose | A camera's position and orientation |
| Quaternion | Four numbers used to represent 3D rotation |
| Rotation matrix | A 3×3 matrix used to rotate 3D points |
| World space | The shared coordinate system used to combine frames |
| Voxel | A small cube in 3D space used for grouping points |
| Outlier | A point that is unusually far from nearby points |
| Drift | Accumulated position or orientation error over a scan |
| Back-end reconstruction | The later processing that turns points into walls, rooms, and plans |
