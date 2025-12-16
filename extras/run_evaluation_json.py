import os
import json
from jigsaw import JigsawExtractor

def evaluate_folder(folder_name, data_root, output_dir):
    source_dir = os.path.join(data_root, folder_name)
    if not os.path.exists(source_dir):
        print(f"Directory not found: {source_dir}")
        return

    results = {
        "folder": folder_name,
        "total_images": 0,
        "correct_predictions": 0,
        "accuracy": 0.0,
        "details": []
    }

    # Create a temporary output directory for the extractor to avoid cluttering
    # We use a single run directory for the whole batch
    extractor_output_dir = os.path.join("temp_extractor_output", folder_name)
    os.makedirs(extractor_output_dir, exist_ok=True)
    
    # Initialize extractor once for the folder
    # Note: JigsawExtractor might scan the folder in __init__, but we will manually load images
    extractor = JigsawExtractor(source_path=source_dir, output_dir=extractor_output_dir)

    files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    results["total_images"] = len(files)

    print(f"Processing {folder_name}: {len(files)} images")

    for filename in files:
        img_path = os.path.join(source_dir, filename)
        true_label = folder_name # e.g., "2x2"
        
        detail = {
            "filename": filename,
            "true_label": true_label,
            "predicted_label": None,
            "success": False,
            "error": None
        }

        try:
            if extractor.load_image(img_path):
                extractor.preprocess()
                extractor.detect_grid_size()
                # We skip slice_and_save() to save time/space as we only need the classification
                
                prediction = f"{extractor.detected_n}x{extractor.detected_n}"
                detail["predicted_label"] = prediction
                
                if prediction == true_label:
                    results["correct_predictions"] += 1
                    detail["success"] = True
                
            else:
                detail["error"] = "Failed to load image"
        except Exception as e:
            detail["error"] = str(e)
            print(f"Error processing {filename}: {e}")

        results["details"].append(detail)

    if results["total_images"] > 0:
        results["accuracy"] = results["correct_predictions"] / results["total_images"]

    output_json_path = os.path.join(output_dir, f"{folder_name}_results.json")
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Saved results to {output_json_path}")

if __name__ == "__main__":
    data_root = "data"
    output_dir = "preprocessed"
    folders = ["2x2", "4x4", "8x8"]

    for folder in folders:
        evaluate_folder(folder, data_root, output_dir)
