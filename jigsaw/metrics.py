from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np


Thresholds = {"8x8": 13674.0, "4x4": 9722.0}


CutDefinitions = {
    "2x2": {"vertical": [0.5], "horizontal": [0.5]},
    "4x4": {"vertical": [0.25, 0.75], "horizontal": [0.25, 0.75]},
    "8x8": {
        "vertical": [0.125, 0.375, 0.625, 0.875],
        "horizontal": [0.125, 0.375, 0.625, 0.875],
    },
}


@dataclass
class DetectionMetrics:
    detected_n: int
    scores: Dict[str, float]
    projections: Dict[str, np.ndarray]
    noise_levels: Dict[str, float]
    energy_details: Dict[str, Dict[str, float]]
    cut_positions: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)


def enhance_projection(projection: np.ndarray) -> np.ndarray:
    signal = projection.astype(float)
    window_size = max(len(signal) // 16, 1)
    if window_size % 2 == 0:
        window_size += 1
    background = cv2.GaussianBlur(signal.reshape(1, -1), (window_size, 1), 0).flatten()
    return np.maximum(signal - background, 0)


def _get_actual_cuts(width: int, height: int) -> Dict[str, Dict[str, List[float]]]:
    cuts = {}
    for label, axes in CutDefinitions.items():
        cuts[label] = {
            "vertical": [width * frac for frac in axes["vertical"]],
            "horizontal": [height * frac for frac in axes["horizontal"]],
        }
    return cuts


def _peak_energy(projection: np.ndarray, indices: List[float], window: int = 3) -> float:
    energy = 0.0
    count = 0
    for idx in indices:
        idx = int(idx)
        start = max(0, idx - window)
        end = min(len(projection), idx + window + 1)
        if start < end:
            energy += float(np.max(projection[start:end]))
            count += 1
    return energy / count if count else 0.0


def compute_detection_metrics(edges: np.ndarray, width: int, height: int) -> DetectionMetrics:
    raw_v = np.sum(edges, axis=0)
    raw_h = np.sum(edges, axis=1)

    v_projection = enhance_projection(raw_v)
    h_projection = enhance_projection(raw_h)

    v_noise = float(np.median(v_projection) + 1.0)
    h_noise = float(np.median(h_projection) + 1.0)

    cuts = _get_actual_cuts(width, height)

    energy_details = {}
    scores = {}

    for label, axes in cuts.items():
        energy_v = _peak_energy(v_projection, axes["vertical"])
        energy_h = _peak_energy(h_projection, axes["horizontal"])

        ratio_v = energy_v / v_noise if v_noise else 0.0
        ratio_h = energy_h / h_noise if h_noise else 0.0
        score = ratio_v + ratio_h

        energy_details[label] = {
            "vertical_energy": energy_v,
            "horizontal_energy": energy_h,
            "vertical_ratio": ratio_v,
            "horizontal_ratio": ratio_h,
            "score": score,
        }
        scores[label] = round(score, 2)

    if scores["8x8"] > Thresholds["8x8"]:
        detected = 8
    elif scores["4x4"] > Thresholds["4x4"]:
        detected = 4
    else:
        detected = 2

    return DetectionMetrics(
        detected_n=detected,
        scores=scores,
        projections={"vertical": v_projection, "horizontal": h_projection},
        noise_levels={"vertical": v_noise, "horizontal": h_noise},
        energy_details=energy_details,
        cut_positions=cuts,
    )
