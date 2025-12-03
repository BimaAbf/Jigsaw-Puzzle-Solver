import csv
import json
import os
import random
import shutil
from datetime import datetime


def _next_test_directory(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        name
        for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name)) and name.startswith("test_")
    ]

    indices = []
    for name in existing:
        suffix = name[5:]
        if suffix.isdigit():
            indices.append(int(suffix))

    next_index = max(indices) + 1 if indices else 1
    return os.path.join(base_dir, f"test_{next_index:03d}"), next_index


def generate_random_test_batch(source_folders, output_folder, ratios, total_samples):
    """Build a mixed dataset by sampling images from source folders."""
    print(f"--- Generating Random Test Batch ({total_samples} images) ---")

    test_dir, test_index = _next_test_directory(output_folder)
    images_dir = os.path.join(test_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    ground_truth = []

    for label, ratio in ratios.items():
        src_path = source_folders.get(label)
        if not src_path or not os.path.exists(src_path):
            print(f" Warning: Source folder for {label} not found at {src_path}")
            continue

        count = int(total_samples * ratio)
        all_files = [
            f
            for f in os.listdir(src_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if len(all_files) < count:
            print(
                f" Warning: Requested {count} images for {label}, but only found {len(all_files)}. Using all."
            )
            selected_files = all_files
        else:
            selected_files = random.sample(all_files, count)

        print(f"   - Selecting {len(selected_files)} images from {label}...")

        for fname in selected_files:
            new_name = f"{label}_{fname}"
            src_file = os.path.join(src_path, fname)
            dst_file = os.path.join(images_dir, new_name)

            shutil.copy2(src_file, dst_file)
            ground_truth.append([new_name, label])

    csv_path = os.path.join(test_dir, "test_ground_truth.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "True_Label"])
        writer.writerows(ground_truth)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "test_index": test_index,
        "test_directory": test_dir,
        "images_directory": images_dir,
        "total_requested": total_samples,
        "ratios": ratios,
        "source_folders": source_folders,
        "images_written": len(ground_truth),
        "ground_truth_csv": csv_path,
    }

    with open(os.path.join(test_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"✅ Generated batch in '{test_dir}' with CSV.")
    return csv_path
