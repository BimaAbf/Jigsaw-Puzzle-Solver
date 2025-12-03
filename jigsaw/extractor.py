import os

import cv2
import matplotlib.pyplot as plt

from jigsaw.discovery import gather_image_paths
from jigsaw.metrics import DetectionMetrics, compute_detection_metrics
from jigsaw.output import (
    append_csv_record,
    ensure_csv_header,
    prepare_run_directories,
    write_descriptors,
    write_manifest,
)
from jigsaw.preprocessing import preprocess_image
from jigsaw.visuals import build_debug_figure
from settings import DEBUG_VISUALS


class JigsawExtractor:
    """Pipeline for detecting square grids in jigsaw puzzle images."""

    def __init__(self, source_path, output_dir="output", debug_visuals=None):
        self.source_path = source_path
        self.base_output_dir = output_dir
        self.debug_visuals_enabled = DEBUG_VISUALS if debug_visuals is None else bool(debug_visuals)

        self.image_paths = gather_image_paths(source_path)
        if not self.image_paths:
            print(f" Warning: No images found in {source_path}")

        run_dirs = prepare_run_directories(self.base_output_dir, enable_debug=self.debug_visuals_enabled)
        self.timestamp = run_dirs["timestamp"]
        self.output_dir = run_dirs["run_dir"]
        self.debug_dir = run_dirs["debug_dir"]
        self.csv_file = run_dirs["csv"]
        self.descriptor_path = run_dirs["descriptors"]
        self.manifest_path = run_dirs["manifest"]

        ensure_csv_header(self.csv_file)

        self.current_image_path = None
        self.original_image = None
        self.gray = None
        self.h = 0
        self.w = 0
        self.edges = None
        self.v_projection = None
        self.h_projection = None
        self.detected_n = 2
        self.scores = {"2x2": 0, "4x4": 0, "8x8": 0}
        self.pieces_data = []
        self.global_piece_id = 0
        self.processed_images = []
        self.noise_levels = {"vertical": None, "horizontal": None}
        self.energy_details = {}
        self.cut_positions = {}
        self._metrics = None

        print(
            f" JigsawExtractor initialized. Found {len(self.image_paths)} images."
            f" Run directory: {self.output_dir}"
        )

    def load_image(self, path):
        self.current_image_path = path
        self.original_image = cv2.imread(path)
        if self.original_image is None:
            return False
        self.h, self.w = self.original_image.shape[:2]
        return True

    def preprocess(self):
        self.gray, self.edges = preprocess_image(self.original_image)

    def detect_grid_size(self):
        if self.edges is None:
            raise ValueError("Edges not computed. Call preprocess() first.")

        self._metrics = compute_detection_metrics(self.edges, self.w, self.h)
        self.detected_n = self._metrics.detected_n
        self.scores = self._metrics.scores
        self.v_projection = self._metrics.projections["vertical"]
        self.h_projection = self._metrics.projections["horizontal"]
        self.noise_levels = self._metrics.noise_levels
        self.energy_details = self._metrics.energy_details
        self.cut_positions = self._metrics.cut_positions

        print(
            f"    Scores - 8x8: {self.scores['8x8']}, 4x4: {self.scores['4x4']}, 2x2: {self.scores['2x2']}"
        )
        print(f"    Detected {self.detected_n}x{self.detected_n} Grid")

        append_csv_record(
            self.csv_file,
            os.path.basename(self.current_image_path),
            self.detected_n,
            self.scores,
        )

    def slice_and_save(self):
        n = self.detected_n
        step_h = self.h // n
        step_w = self.w // n

        print(f"--- Step 3: Processing {n * n} pieces ({step_w}x{step_h} px) ---")

        for row in range(n):
            for col in range(n):
                y1 = row * step_h
                y2 = (row + 1) * step_h
                x1 = col * step_w
                x2 = (col + 1) * step_w

                piece_info = {
                    "id": self.global_piece_id,
                    "source_image": os.path.basename(self.current_image_path),
                    "grid_pos": {"row": row, "col": col},
                    "bbox": {"x": x1, "y": y1, "w": step_w, "h": step_h},
                }
                self.pieces_data.append(piece_info)
                self.global_piece_id += 1

    def save_debug_visualization(self):
        if not self.debug_visuals_enabled or not self.debug_dir:
            return

        if self.edges is None or self._metrics is None:
            return

        fig = build_debug_figure(self._metrics, self.original_image, self.detected_n)

        filename = f"debug_{os.path.basename(self.current_image_path)}"
        save_path = os.path.join(self.debug_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)
        print(f"    Saved Enhanced Projection graph to {save_path}")

    def run_batch(self):
        processed_count = 0
        for img_path in self.image_paths:
            print(f"\n Processing: {os.path.basename(img_path)}")
            if self.load_image(img_path):
                self.preprocess()
                self.detect_grid_size()
                self.slice_and_save()
                self.save_debug_visualization()
                self.processed_images.append(os.path.basename(img_path))
                processed_count += 1

        self.finalize_run(processed_count)

    def finalize_run(self, processed_count):
        write_descriptors(self.descriptor_path, self.pieces_data)
        print(f" Saved descriptors to {self.descriptor_path}")
        print(f"\n Batch Analysis stored in: {self.csv_file}")

        manifest = {
            "timestamp": self.timestamp,
            "source_path": self.source_path,
            "output_dir": self.output_dir,
            "images_discovered": len(self.image_paths),
            "images_processed": processed_count,
            "processed_images": self.processed_images,
            "csv_path": self.csv_file,
            "descriptors_path": self.descriptor_path,
            "debug_visuals_dir": self.debug_dir,
            "debug_visuals_enabled": self.debug_visuals_enabled,
        }

        write_manifest(self.manifest_path, manifest)
        print(f" Run artifacts saved in: {self.output_dir}")
