# Jigsaw Puzzle

This repository hosts a small toolkit for experimenting with square jigsaw puzzle assembly, dataset generation, and image thresholding. It combines data preparation utilities, evaluation helpers, and solver prototypes intended for coursework-scale experiments.

## Repository Layout

- `evaluation/` – dataset builder and scoring helpers used by the top-level evaluation script.
- `evaluation.py` – command-line entry point that creates blended test batches and evaluates them.
- `processor.py` – puzzle scoring utilities shared across prototypes.
- `thresholding/` – threshold exploration utilities for preprocessing studies.
- `threshold_calculator.py` – quick script for running the thresholding experiments.
- `.vscode/` – workspace settings for local development (optional).

Folders such as `Tests/`, `solver_data/`, and `solved_puzzles/` are intentionally ignored by version control because they store generated artifacts.

## Requirements

- Python 3.10 or newer
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
- Any additional dependencies listed in individual modules

Install the required packages once before running any scripts:

```sh
python -m pip install --upgrade pip
python -m pip install opencv-python numpy matplotlib
```

## Quick Start

1. Clone or copy the project into your working directory.
2. Ensure the expected folder structure exists:
   - Place raw puzzle images in class-specific folders (for example, `data/2x2`, `data/4x4`, `data/8x8`).
   - Confirm that the `Tests/` directory is writable; it will be populated with generated test batches.
3. Configure the source directories inside `evaluation.py` so that each puzzle size points to the folder containing its images.

## Generating and Evaluating Test Batches

The top-level `evaluation.py` script assembles a mixed dataset based on sampling ratios, then evaluates the generated collection using the helper functions in `evaluation/`.

```sh
python evaluation.py
```

The script will:

1. Create a new test directory inside `Tests/` (for example, `Tests/test_004`).
2. Copy sampled images into `Tests/test_XXX/images/` with a size prefix.
3. Write a `test_ground_truth.csv` manifest and `manifest.json` summarising the run.
4. Invoke `evaluation.evaluate_generated_batch` to compute metrics for the generated dataset.

Edit the `source_folders`, `ratios`, and `total_images` variables in `evaluation.py` to match your dataset layout and desired sampling plan before running the script.

## Thresholding Utilities

Use `threshold_calculator.py` and the helpers under `thresholding/` to profile intensity thresholds across a directory of images. These tools are isolated from the solver pipeline and can be run independently:

```sh
python threshold_calculator.py --images PATH/TO/IMAGES
```