import os

from evaluation import generate_random_test_batch, evaluate_generated_batch


if __name__ == "__main__":
    source_folders = {
        "4x4": os.path.join("data", "4x4"),
        "8x8": os.path.join("data", "8x8"),
        "2x2": os.path.join("data", "2x2"),
    }

    ratios = {
        "2x2": 0.35,
        "4x4": 0.35,
        "8x8": 0.3,
    }

    total_images = 110
    tests_root = "Tests"

    try:
        csv_path = generate_random_test_batch(source_folders, tests_root, ratios, total_images)
        dataset_dir = os.path.dirname(csv_path)
        evaluate_generated_batch(dataset_dir, csv_path)
    except Exception as exc:
        print(f"\n Script failed: {exc}")