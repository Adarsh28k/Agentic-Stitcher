# Running the complete pipeline

## What this file does

`run_pipeline.py` is a thin orchestrator. It calls the existing phase scripts in order:

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

The phase implementations remain in their own folders. The orchestrator manages the project path, execution order, failure status, and one combined report.

## Run it

From the `Agentic-Stitcher` folder:

```bash
python3 run_pipeline.py
```

To run it against another project folder with the same expected capture structure:

```bash
python3 run_pipeline.py /path/to/project
```

## Expected input structure

The current phase scripts expect these three capture folders:

```text
single_room/
single_scan_floor_only/
single_scan_with_ceiling/
```

Each capture should contain at least:

```text
depth/
confidence/
odometry.csv
camera_matrix.csv
```

## Outputs

Each phase keeps its own outputs. The orchestrator additionally writes:

```text
pipeline_outputs/pipeline_report.json
```

That report contains:

- Overall pipeline status
- Duration and return code for each phase
- Expected phase report paths
- Current limitations

## Smoke-test meaning

A successful smoke test means that all scripts execute in order and produce their expected reports and diagnostic files. It does not mean that the room dimensions are accurate or that the global stitched cloud is ready for production. The Phase 6 README records the current geometric limitations.

