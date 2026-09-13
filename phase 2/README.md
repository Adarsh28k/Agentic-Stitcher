# Phase 2 — Visual diagnostics

Phase 2 turns the numeric sensor data into images that a person can inspect. It does not reconstruct the room yet.

## Dependencies

```bash
python3 -m pip install pandas numpy pillow matplotlib
```

## Generate the images

From the `Agentic-Stitcher` folder:

```bash
python3 "phase 2/generate_diagnostics.py"
```

The images are written under:

```text
phase 2/outputs/<capture-name>/
```

The machine-readable summary is:

```text
phase 2/phase_2_report.json
```

## Start here: one example frame

Open these four files from the `single_scan_with_ceiling` capture:

- [Sensor maps](<outputs/single_scan_with_ceiling/frame_004872_sensor_maps.png>)
- [Depth map](<outputs/single_scan_with_ceiling/frame_004872_depth.png>)
- [Confidence map](<outputs/single_scan_with_ceiling/frame_004872_confidence.png>)
- [Camera trajectory](<outputs/single_scan_with_ceiling/trajectory.png>)

Read them in this order:

1. Open the sensor-map image to see depth and confidence side by side.
2. Open the depth map to inspect the spatial structure in more detail.
3. Open the confidence map to see which depth regions are trustworthy.
4. Open the camera trajectory to see how the camera moved during the scan.

## What the images mean

### Depth images

Files ending in `_depth.png` show the depth map with a color scale:

- Dark blue/purple means a smaller raw depth value within that image.
- Green/yellow means a larger raw depth value within that image.
- Black means an invalid or zero depth pixel.

The colors are normalized separately for each image. They do not yet mean metres. The depth unit is still unresolved. Therefore, yellow in one frame cannot yet be compared numerically with yellow in another frame.

### Confidence images

Files ending in `_confidence.png` use a simple fixed legend:

- Black (`0`) means invalid or unusable confidence.
- Yellow (`1`) means medium confidence.
- Green (`2`) means the strongest confidence value present in these files.

These are quality maps, not ordinary grayscale photographs.

### Sensor-map pair images

Files ending in `_sensor_maps.png` put depth and confidence side by side so you can compare geometry with measurement quality.

### Trajectory images

`trajectory.png` plots the raw odometry positions using the exporter’s `x`, `y`, and `z` columns. It marks the beginning in green and the end in red.

The axis directions and units are intentionally not interpreted yet. The plot is only used to see whether the camera moved smoothly or produced a suspiciously scattered path.

The trajectory may contain loops because the camera can walk around the room and revisit areas. A loop is not automatically an error. We will evaluate it after confirming the coordinate convention and comparing the reconstructed geometry with measurements.

## How to inspect them

Start with:

```text
phase 2/outputs/single_scan_with_ceiling/
```

Open one `_sensor_maps.png` file and its matching `_depth.png` and `_confidence.png` files. Then open `trajectory.png`.

Look for:

1. Depth structure that changes across the image rather than a completely flat field.
2. Confidence values that identify which depth regions are trustworthy.
3. A trajectory that looks continuous rather than jumping randomly.
4. Differences between the first, middle, and final frames.

In the example frame, the depth map contains meaningful spatial variation and the confidence map is mostly green. That means the sensor appears to contain usable depth data, but it does not yet prove that the final room dimensions will be correct.

## Current limitation

RGB preview extraction is not enabled in this phase because the current Python environment does not have a video decoder available to the script. The supplied MP4 files are still recorded in the Phase 1 manifest. RGB inspection will be added once a decoder is available.
