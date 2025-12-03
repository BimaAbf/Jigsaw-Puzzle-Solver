import csv
import json
import os
from datetime import datetime
from typing import Dict, List


def prepare_run_directories(base_output_dir: str, enable_debug: bool = True) -> Dict[str, str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(base_output_dir, exist_ok=True)

    run_dir = os.path.join(base_output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    debug_dir = os.path.join(run_dir, "debug_visuals") if enable_debug else None
    if enable_debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    return {
        "timestamp": timestamp,
        "run_dir": run_dir,
        "debug_dir": debug_dir,
        "csv": os.path.join(run_dir, "results.csv"),
        "descriptors": os.path.join(run_dir, "descriptors.json"),
        "manifest": os.path.join(run_dir, "run_manifest.json"),
    }


def ensure_csv_header(csv_path: str) -> None:
    if os.path.exists(csv_path):
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Filename",
            "Detected_Grid",
            "Score_2x2",
            "Score_4x4",
            "Score_8x8",
        ])


def append_csv_record(csv_path: str, filename: str, detected: int, scores: Dict[str, float]) -> None:
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            filename,
            f"{detected}x{detected}",
            scores["2x2"],
            scores["4x4"],
            scores["8x8"],
        ])


def write_descriptors(path: str, descriptors: List[Dict]) -> None:
    with open(path, "w") as f:
        json.dump(descriptors, f, indent=4)


def write_manifest(path: str, payload: Dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=4)
