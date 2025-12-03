import os

import numpy as np

from jigsaw import JigsawExtractor
from .visuals import plot_decision_boundaries


def calculate_global_optimum(folders_dict):
    """Search for the best thresholds that maximize classification accuracy."""
    print("---  Starting Global 3-Class Optimization ---")

    data = []

    for label, folder in folders_dict.items():
        if not os.path.exists(folder):
            print(f" Skipping missing folder: {folder}")
            continue

        print(f" Loading Ground Truth Batch: {label}...")
        extractor = JigsawExtractor(source_path=folder, output_dir="opt_temp")

        for img_path in extractor.image_paths:
            if extractor.load_image(img_path):
                extractor.preprocess()
                extractor.detect_grid_size()

                data.append(
                    {
                        "true_label": label,
                        "s8": extractor.scores["8x8"],
                        "s4": extractor.scores["4x4"],
                    }
                )

    if not data:
        print(" No data found.")
        return

    all_s8 = [d["s8"] for d in data]
    all_s4 = [d["s4"] for d in data]

    steps = 100
    t8_range = np.linspace(min(all_s8), max(all_s8), steps)
    t4_range = np.linspace(min(all_s4), max(all_s4), steps)

    best_stats = {"acc": 0, "t8": 0, "t4": 0}

    print(" Running Grid Search simulation...")

    for t8 in t8_range:
        for t4 in t4_range:
            correct = 0

            for item in data:
                if item["s8"] > t8:
                    pred = "8x8"
                elif item["s4"] > t4:
                    pred = "4x4"
                else:
                    pred = "2x2"

                if pred == item["true_label"]:
                    correct += 1

            acc = correct / len(data)

            if acc > best_stats["acc"]:
                best_stats = {"acc": acc, "t8": t8, "t4": t4}

    best_t8 = best_stats["t8"]
    best_t4 = best_stats["t4"]

    class_stats = {
        "2x2": {"total": 0, "correct": 0},
        "4x4": {"total": 0, "correct": 0},
        "8x8": {"total": 0, "correct": 0},
    }

    for item in data:
        label = item["true_label"]
        class_stats[label]["total"] += 1

        if item["s8"] > best_t8:
            pred = "8x8"
        elif item["s4"] > best_t4:
            pred = "4x4"
        else:
            pred = "2x2"

        if pred == label:
            class_stats[label]["correct"] += 1

    print("\n" + "=" * 50)
    print(" GLOBAL 3-CLASS OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Overall System Accuracy: {best_stats['acc'] * 100:.2f}%")
    print("-" * 30)
    print(f"Optimal Threshold T8:    {best_t8:.2f}")
    print(f"Optimal Threshold T4:    {best_t4:.2f}")
    print("-" * 30)
    print("Per-Class Performance:")
    for cls in ["2x2", "4x4", "8x8"]:
        tot = class_stats[cls]["total"]
        corr = class_stats[cls]["correct"]
        acc = (corr / tot) * 100 if tot > 0 else 0
        print(f"   {cls}: {acc:.1f}% ({corr}/{tot})")
    print("=" * 50)

    plot_decision_boundaries(data, best_t8, best_t4)
