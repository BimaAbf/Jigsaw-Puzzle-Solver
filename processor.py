import os

from jigsaw import JigsawExtractor


if __name__ == "__main__":
    input_path = ""

    if not os.path.exists(input_path):
        os.makedirs(input_path, exist_ok=True)
        print(f" Test folder '{input_path}' created.")

    try:
        extractor = JigsawExtractor(source_path=input_path)
        extractor.run_batch()
        print("\n Grid-based Segmentation Complete!")
    except Exception as e:
        print(f"\n Error: {e}")