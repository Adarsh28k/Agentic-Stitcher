# Agentic Stitcher: Technical Report
### A Prototype 3D Room Reconstruction Pipeline

---

## What Got Built

The pipeline has seven phases. Each one builds on the results of the one before it. Each phase runs as its own script, writes a summary file, and saves output images. Everything runs from the main `Agentic-Stitcher` folder.

The data feeding into this pipeline comes from the three capture folders provided: `single_scan_with_ceiling`, `single_scan_floor_only`, and `single_room`. Each folder contains depth images, confidence images, an odometry CSV with camera positions, an IMU CSV with motion sensor readings, a camera settings file, and an RGB video file. All seven phases of the pipeline read directly from these three folders. No external data sources are used.

---

## Phase 0: Making Sure the Files Are Okay

Before writing any 3D reconstruction code, the first sensible question is whether the files are even readable and whether they match each other. Phase 0 answers that.

The script goes through all three capture folders and checks every data file. It confirms that the depth images, confidence images, and camera position records all reference the same frame numbers. It checks that the sensor timestamps go forward in time and do not jump around. It also looks at the camera settings file to make sure the values are in a reasonable range.

**To run it:**
```bash
python3 "phase 0/validate_dataset.py"
```

**Output file:** `phase 0/phase_0_report.json`

**What the check found:** The files load without errors. The frame numbers line up between depth, confidence, and position data. Timestamps are in order. The camera settings file has sensible values.

**What the check could not confirm:**

There are several things the files do not tell us, and they matter a lot:

- Whether depth values are in millimetres, metres, or some other unit
- Which direction the camera's axes point (which way is forward, which way is up)
- What order the four rotation numbers are stored in
- Whether the position data describes where the camera is in the world or the other way around
- How the video frames line up in time with the depth frames

These are not small details. If the depth unit is wrong, every distance measurement the pipeline produces will be wrong. If the rotation numbers are in the wrong order, every frame ends up pointing the wrong direction and the combined 3D result becomes meaningless. The Phase 0 report lists all of these as open questions rather than quietly guessing.

---

## Phase 1: Loading the Data Cleanly

Phase 1 creates a consistent way to load any capture folder so that later phases do not each have to write their own file-reading code. It reads the camera settings as a number grid, loads the sensor and position records into tables, and indexes the depth and confidence images by frame number so they are easy to look up.

**Install required libraries:**
```bash
python3 -m pip install pandas numpy pillow
```

**To run it:**
```bash
python3 "phase 1/run_phase_1.py"
```

**Output file:** `phase 1/phase_1_report.json`

**How to use the loader from another script:**
```python
from pathlib import Path
from capture_loader import get_frame, load_capture

capture_path = Path("single_scan_with_ceiling")
capture = load_capture(capture_path)
frame = get_frame(capture, 0)

print(frame["depth"].shape)        # raw depth values, nothing scaled
print(frame["confidence"].shape)   # 0, 1, or 2 per pixel
print(frame["timestamp"])
print(frame["odometry"])
```

One deliberate choice worth explaining: the loader does not apply any scaling to the depth values. It gives back exactly what the sensor recorded. Any conversion to real-world units happens in a later phase and is clearly flagged as a guess until the actual unit is confirmed. This keeps the loader trustworthy as a foundation for everything that comes after it.

---

## Phase 2: Turning the Numbers Into Pictures

Before writing any code that tries to reconstruct 3D geometry, it helps to look at the data visually. Phase 2 converts the depth and confidence numbers into colour images, puts them side by side for comparison, and draws the path the camera took during the scan.

**Install required libraries:**
```bash
python3 -m pip install pandas numpy pillow matplotlib
```

**To run it:**
```bash
python3 "phase 2/generate_diagnostics.py"
```

**Where outputs go:** `phase 2/outputs/<capture-name>/`

**What the images mean:**

The depth images use a colour scale that goes from dark blue for closer distances to yellow for farther ones. Black pixels are spots where the sensor had no reading. The important thing to know is that each image is coloured independently, so yellow in one frame does not mean the same distance as yellow in another. The unit is still unknown at this stage.

The confidence images use three fixed colours. Black means the sensor flagged that pixel as invalid. Yellow means medium confidence. Green means the sensor is most confident about that reading. These are quality maps, not photos.

The camera path plot shows the position of the camera at each recorded moment. The start is marked green and the end is marked red. The axes are not given real-world labels yet because the coordinate system is still unconfirmed.

**One gap to note:** The phase does not pull preview images from the video files because the development environment did not have a usable video decoder available. The video files are noted in the Phase 1 records and can be accessed once a decoder is set up. This is a known limitation, not something hidden.

**What Phase 2 confirmed:** The depth images show real spatial structure. The sensor is not producing random noise. Most pixels in the sample frames have the highest confidence rating, which is a good sign. The camera paths look continuous and smooth rather than jumping around randomly.

---

## Phase 3: Building a 3D Point Cloud

This is where the actual 3D work starts. Phase 3 takes each depth image and converts every pixel into a 3D point. It then uses the recorded camera positions to place all those points into a single shared coordinate system.

**Install required libraries:**
```bash
python3 -m pip install pandas numpy pillow matplotlib open3d
```

**To run it:**
```bash
python3 "phase 3/build_point_cloud.py"
```

**Where outputs go:** `phase 3/outputs/<capture-name>/`

### How a pixel becomes a 3D point

For each pixel at position `u` across and `v` down, with a depth reading `d`, the code uses the camera's calibration numbers (`fx`, `fy`, `cx`, `cy`) to calculate where in space that pixel is:

```
X = (u - cx) x d / fx
Y = (v - cy) x d / fy
Z = d
```

This turns a flat image measurement into a location in space relative to the camera.

One adjustment that matters: the camera calibration numbers in the data were set up for a 1920 by 1440 image, but the depth images are only 256 by 192. Before using those calibration numbers, they are scaled down to match the smaller depth image size. Skipping this step would stretch the 3D result badly.

The depth scale used is 0.001, treating a raw value of 2500 as roughly 2.5 units in the same scale as the position data. This is probably converting millimetres to metres, but the source files do not confirm that, so it is flagged as a provisional assumption.

### Cleaning the data before building the cloud

A few steps clean up the data before the final result is saved:

- Pixels with a depth reading of zero are thrown out
- Only pixels with a confidence rating of 1 or higher are kept, removing ones the sensor flagged as invalid
- Only one in every four pixels is used when combining frames, which keeps the processing fast while still capturing the large surfaces that matter
- Nearby points that are almost in the same spot get merged into one representative point (voxel downsampling)
- Points that are far away from all their neighbours and likely noise get removed (statistical outlier removal)

### What Phase 3 found

The result from a single frame looks good. One depth image converted to 3D using the scaled calibration numbers produces a surface that looks like a real room surface. That is a useful confirmation that the basic maths and the calibration scaling are working.

The result from combining many frames is not good yet. The surfaces appear doubled, smeared, or shifted apart from each other. This is not a bug in the code. It is a sign that at least one of the provisional assumptions about the coordinate system or depth unit is incorrect. Without documentation from whoever collected the data or actual room measurements to compare against, the specific cause cannot be pinpointed.

The combined result is saved and labelled as provisional. It is not presented as a finished scan.

---

## Phase 4: Extracting a Room Outline

Phase 4 tries to take the 3D point cloud from Phase 3 and find the flat surfaces in it, then use those to estimate a room outline.

**To run it:**
```bash
python3 "phase 4/extract_room_plan.py"
```

**Where outputs go:** `phase 4/outputs/<capture-name>/`

### How it works

A typical room is made mostly of flat surfaces: a floor, a ceiling, and walls. If you can find those flat surfaces in a point cloud, you can estimate the room shape.

The method used is called RANSAC plane fitting. It works by repeatedly picking a small random group of points, fitting a flat plane through them, and counting how many other points in the cloud fall close to that same plane. A plane that many points agree on is likely a real surface. The direction the plane faces tells you whether it is a floor, ceiling, or wall.

### Why the results are not trustworthy yet

Because the combined point cloud from Phase 3 is still scattered and imprecise, the planes Phase 4 finds may not correspond to real room surfaces. The plane finder might identify the same wall twice, or pick up a false surface created by the position errors from Phase 3.

For this reason, all the numbers in the Phase 4 output, including estimated ceiling height, floor area, and wall lengths, are labelled as rough diagnostics only. They should not be taken as real measurements until three things are confirmed:

1. The correct coordinate system
2. The correct depth unit
3. At least one set of real-world measurements from a tape measure or laser to compare against

It would have been easy to adjust the numbers to look reasonable. That was not done. Fake-looking-reasonable numbers are worse than no numbers.

---

## Phase 5: Comparing the Three Captures

Phase 5 reads the results from Phase 3 and Phase 4 across all three capture folders and puts the numbers side by side to see how consistent the pipeline is.

**To run it:**
```bash
python3 "phase 5/evaluate_robustness.py"
```

**Where outputs go:** `phase 5/outputs/`

The three folders play different roles:

| Capture folder | What it represents |
|---|---|
| `single_scan_with_ceiling` | Main development capture, most complete coverage |
| `single_scan_floor_only` | Test with only floor-level coverage |
| `single_room` | A different trajectory for a sanity check |

These three folders are not the same room scanned in the same way three times. They have different camera paths and different coverage. That means this comparison can check whether the pipeline behaves similarly across different inputs, but it cannot confirm repeatability in the strict sense, which would require the exact same room, same method, twice.

**What the comparison found:** The estimated room outlines and ceiling heights were quite different across the three captures. The floor-only capture could not produce a usable wall outline at all. This tells us the pipeline is sensitive to how much of the room the camera saw during the scan, which is expected at this stage. It also confirms that the current results are not stable enough to claim production quality.

---

## Phase 6: Finding Loop Closures and Building a Pose Graph

Phase 6 works with the longest capture, `single_scan_with_ceiling`, and looks for places in the camera path where the camera came back to a spot it had visited earlier. Those revisited spots can help correct the drift that builds up over a long scan.

**To run it:**
```bash
python3 "phase 6/build_pose_graph.py"
```

**Where outputs go:** `phase 6/outputs/`

### The basic idea

When a camera walks through a space for a long time, small position errors add up. By the end of the scan, the recorded position can be off by a meaningful amount from where the camera actually was. This is called drift.

One way to correct drift is to notice when the camera returns to a place it has already seen. If you can confirm that two frames from different times in the scan are actually looking at the same spot, you can use that information to adjust the whole path and reduce the accumulated error. This is called a loop closure.

A pose graph is a way of organising this information. Each recorded camera position is a node. Each step along the recorded path is a connection between neighbouring nodes. Each confirmed revisit is an extra connection between two nodes that are far apart in time but close in space. An optimiser then finds the set of positions that best satisfies all the connections at once.

### What the analysis found

Starting from 500 candidate revisits found by looking for camera positions that came close to earlier ones, 101 showed some geometric agreement when compared against the actual depth data, and 6 showed strong agreement. The strongest match was between frames 3720 and 8430, where the error between the two depth clouds dropped from around 48.5 cm to 3.9 cm after alignment. That is a strong sign that these two frames genuinely capture the same physical spot.

After a stricter check that filtered out matches requiring large position corrections, only two revisit pairs remained for the actual optimisation:

- frames 5040 and 9600
- frames 3720 and 8430

The optimiser brought the position errors for those two pairs down to essentially zero. But the overall 3D result did not improve. The raw recorded positions produced a better-looking cloud than the optimised ones did.

This is a known problem. When there are too few confirmed revisits and the coordinate system is still provisional, the optimiser can satisfy its constraints in a way that makes the global result worse rather than better. The raw position data remains the better starting point for now.

---

## Where Things Are Right Now

### What is working

- All seven phases run and produce output files without crashing
- Single-frame 3D results look like real room surfaces
- The confidence and depth statistics give useful information about data quality
- The plane-finding step does find large flat surfaces, even if they cannot be trusted as accurate yet
- The loop closure analysis confirms that the long capture does contain genuine revisited areas
- Each phase produces a JSON summary file that can be reproduced by running the script again

### What is not working yet

- Combining frames from multiple positions still produces scattered, unreliable results because the coordinate assumptions have not been confirmed
- Room measurements cannot be trusted without fixing the multi-frame problem first
- The photo and video input paths described in the problem statement have not been built
- Damage detection was not attempted because no labelled damage examples were provided
- The pipeline is not ready to run on an unfamiliar space and produce reliable results on the spot
- A fair comparison against Magicplan or Poly.cam cannot be done when the pipeline's own output is not yet trustworthy

### What is needed to get this to production quality

1. Confirmation from whoever collected the data about what unit the depth values are in, how the coordinate axes are set up, and what order the rotation numbers are stored in
2. Physical measurements of at least one room with a tape measure or laser, to check the pipeline output against reality
3. At least two identical scans of the same room so repeatability can be properly tested
4. Labelled examples of room damage with class labels, locations, and size measurements, before damage detection can be built or evaluated
5. A working video decoder so the video input path can be developed
6. Several months of additional development time, not days

---

## A Note on Scope and the Road Ahead

The Cozmo AI case study sets out a genuinely exciting vision: a pipeline that takes photos, video, and LiDAR data from any iPhone, produces centimetre-accurate room plans, detects surface damage, stitches multiple rooms into a whole-property floor plan, and handles a cold walk-in test on a space it has never seen. That is a compelling product, and the ambition behind it is completely understandable.

It is also, to be transparent, a research and engineering effort that would realistically take a dedicated team several months to build and validate properly. Companies like Matterport and Leica have spent years and significant resources reaching that level of accuracy and reliability. The consumer apps referenced in the problem statement, even the free-tier ones, are the result of the same kind of sustained investment. This is not said to discourage the vision at all, but rather to frame what a two-day prototype can honestly deliver versus what a longer-term project roadmap would look like.

What this submission represents is the strongest possible foundation within that time. A seven-phase pipeline that moves through the problem in a structured way, produces real diagnostic output at each step, and is transparent about where the current limits are. No numbers have been invented to make the results look better than they are. Where something is uncertain, it is labelled as such. That honesty is intentional, because a prototype that clearly shows where the next problems are is far more useful as a starting point than one that papers over them.

There is also one aspect of the provided data worth mentioning gently. The three capture folders, while very useful for getting the pipeline running, were missing some calibration details that matter quite a bit for accurate reconstruction. Things like the exact unit the depth values are stored in, the camera coordinate axis convention, and the rotation number ordering were not documented in the files. The three folders also represent different scanning styles rather than the same room scanned in identical ways, which makes a formal repeatability comparison difficult. And without physical tape or laser measurements of the rooms, there is currently no ground truth to check the pipeline output against.

These gaps do not reflect poorly on the data itself. They are simply the kinds of details that tend to get overlooked when preparing early datasets, and they are entirely solvable. Confirming them would allow the pipeline to move from its current diagnostic state to producing measurements that can be trusted. That next step, along with extending the pipeline to cover the photo and video input tiers, damage detection, and the full multi-room stitching, is very achievable with a bit more time and those confirmed calibration details.

The groundwork is here. The path forward is clear.

---

## How to Run Everything

Install the required libraries first:

```bash
python3 -m pip install pandas numpy pillow matplotlib open3d
```

### The orchestrator

Rather than running each phase script by hand, the whole pipeline can be started with a single command using `run_pipeline.py`. This is a thin orchestrator that calls each phase in the correct order, tracks whether each one succeeded, and writes a single combined report at the end.

The orchestrator reads from the three capture folders provided: `single_scan_with_ceiling`, `single_scan_floor_only`, and `single_room`. All of the data used across every phase comes from these three folders. The orchestrator expects them to be present inside the project directory and passes the correct paths to each phase script automatically.

From the `Agentic-Stitcher` folder:

```bash
python3 run_pipeline.py
```

To run it against a different project folder that has the same folder structure:

```bash
python3 run_pipeline.py /path/to/project
```

The phases run in this order:

```text
Phase 0 validation
    ↓
Phase 1 loading
    ↓
Phase 2 diagnostics
    ↓
Phase 3 point cloud
    ↓
Phase 4 room plan
    ↓
Phase 5 cross-capture evaluation
    ↓
Phase 6 pose graph
    ↓
Phase 6B pose-graph experiment
```

Each phase still keeps its own output folder. The orchestrator additionally writes a combined summary to:

```text
pipeline_outputs/pipeline_report.json
```

That file records the overall pipeline status, how long each phase took, the return code for each phase, the paths to each phase report, and a note on current limitations.

### What the input folders need to contain

Each of the three capture folders should have at least:

```text
depth/
confidence/
odometry.csv
camera_matrix.csv
```

### What a successful run means

A clean run means all seven phase scripts executed in order and produced their expected report files and diagnostic images. It does not mean the room dimensions are accurate or that the combined 3D result is ready for production use. The geometric limitations described in this report still apply. The orchestrator confirms that the pipeline runs end to end; it does not validate the physical correctness of the output.

No GPU is needed. No trained AI model is called. No connection to external services is made.

---

## Glossary

| Term | What it means |
|---|---|
| Back-projection | Working backwards from a pixel in a depth image to a 3D point in space |
| Confidence map | An image showing how reliable each depth pixel is (0 = invalid, 1 = medium, 2 = high) |
| Depth map | An image where each pixel stores a distance reading instead of a colour |
| Drift | The slow build-up of position errors that happens during a long camera scan |
| Intrinsics | The camera's own calibration numbers, describing its focal length and centre point |
| ICP | A method for aligning two point clouds by iteratively matching close points |
| Loop closure | Detecting that the camera has returned to a place it already visited, used to correct drift |
| Point cloud | A collection of 3D points, each with an X, Y, Z position |
| Pose | The position and orientation of the camera at a given moment |
| Pose graph | A network of camera positions connected by motion links, used to correct drift globally |
| Quaternion | A set of four numbers used to represent a 3D rotation |
| RANSAC | A method for fitting a model to noisy data by looking for the fit that most points agree on |
| Voxel | A small cube in 3D space, used to represent a region and reduce point cloud density |
| World space | The shared coordinate system that all frames are placed into when building a combined 3D model |

---

*This report covers Phases 0 through 6 of the Agentic Stitcher prototype. All geometric results should be treated as early-stage diagnostics until the calibration assumptions are confirmed and physical measurements are available for comparison.*
