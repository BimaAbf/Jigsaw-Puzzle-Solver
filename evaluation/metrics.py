import csv
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from jigsaw import JigsawExtractor


def evaluate_generated_batch(test_root, csv_path):
    """Run the extractor on the generated folder and print evaluation metrics."""
    print("\n--- Running System Evaluation on Mixed Batch ---")

    truth_map = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            truth_map[row["Filename"]] = row["True_Label"]

    if not truth_map:
        print("❌ CSV is empty.")
        return

    images_dir = os.path.join(test_root, "images")
    if not os.path.isdir(images_dir):
        images_dir = test_root

    extractor_output_root = os.path.join(test_root, "extractor_runs")
    extractor = JigsawExtractor(source_path=images_dir, output_dir=extractor_output_root)

    y_true = []
    y_pred = []
    failed = []

    for filename in truth_map.keys():
        img_path = os.path.join(images_dir, filename)
        expected = truth_map[filename]

        try:
            if extractor.load_image(img_path):
                extractor.preprocess()
                extractor.detect_grid_size()
                extractor.slice_and_save()
                extractor.save_debug_visualization()
                extractor.processed_images.append(filename)

                prediction = f"{extractor.detected_n}x{extractor.detected_n}"

                y_true.append(expected)
                y_pred.append(prediction)
            else:
                print(f" Failed to load image: {filename}")
                failed.append(filename)

        except Exception as exc:
            print(f" Error processing {filename}: {exc}")
            failed.append(filename)

    if not y_true:
        print(" No predictions made.")
        return

    accuracy = accuracy_score(y_true, y_pred)
    unique_labels = sorted(list(set(y_true + y_pred)))

    print("\n" + "=" * 40)
    print(" TEST BATCH REPORT")
    print("=" * 40)
    print(f"Images Tested:    {len(y_true)}")
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print("-" * 40)
    print(classification_report(y_true, y_pred, zero_division=0))
    print("=" * 40)

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=unique_labels,
        yticklabels=unique_labels,
    )
    plt.title(f"Confusion Matrix (Batch Acc: {accuracy * 100:.1f}%)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    plots_dir = os.path.join(test_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close()
    print(f" Confusion matrix saved to {plot_path}")

    report_path = os.path.join(test_root, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("TEST BATCH REPORT\n")
        f.write("=" * 40 + "\n")
        f.write(f"Images Tested: {len(y_true)}\n")
        f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write(classification_report(y_true, y_pred, zero_division=0))
        f.write("\n")
    print(f" Classification report saved to {report_path}")

    if failed:
        print(f" Warning: {len(failed)} files failed to process.")
        failed_path = os.path.join(test_root, "failed_images.txt")
        with open(failed_path, "w") as f:
            for name in failed:
                f.write(f"{name}\n")
        print(f" Failed filenames logged to {failed_path}")

    extractor.finalize_run(len(extractor.processed_images))
