import cv2
import matplotlib.pyplot as plt

from jigsaw.metrics import DetectionMetrics


def _add_projection_axis(ax, projection, length, noise_level, cuts, color_map, orientation):
    if orientation == "vertical":
        ax.plot(projection, color="green")
        ax.set_xlim(0, length)
        ax.set_ylim(bottom=0)
        ax.axhline(noise_level, color="orange", linestyle=":", linewidth=1)
    else:
        ax.plot(projection, range(length), color="green")
        ax.invert_yaxis()
        ax.set_ylim(length, 0)
        ax.set_xlim(left=0)
        ax.axvline(noise_level, color="orange", linestyle=":", linewidth=1)

    for label, positions in cuts.items():
        axis_positions = positions[orientation]
        for pos in axis_positions:
            idx = int(min(max(pos, 0), length - 1))
            if orientation == "vertical":
                ax.scatter(pos, projection[idx], color=color_map[label], s=20)
            else:
                ax.scatter(projection[idx], pos, color=color_map[label], s=20)

    ax.axis("off")


def build_debug_figure(metrics: DetectionMetrics, image, detected_n: int):
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)

    color_map = {"2x2": "blue", "4x4": "purple", "8x8": "red"}

    ax_top = fig.add_subplot(gs[0, 0])
    _add_projection_axis(ax_top, metrics.projections["vertical"], width, metrics.noise_levels["vertical"], metrics.cut_positions, color_map, "vertical")
    ax_top.set_title(f"Det: {detected_n}x{detected_n} (8x8 Score: {metrics.scores['8x8']})")

    ax_main = fig.add_subplot(gs[1, 0])
    ax_main.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for i in range(1, detected_n):
        ax_main.axvline(x=i * (width / detected_n), color="red", linestyle="--", linewidth=1)
        ax_main.axhline(y=i * (height / detected_n), color="red", linestyle="--", linewidth=1)
    ax_main.axis("off")

    detail_lines = [
        f"Noise Floor V/H: {metrics.noise_levels['vertical']:.1f} / {metrics.noise_levels['horizontal']:.1f}"
    ]
    for label in ["8x8", "4x4", "2x2"]:
        if label in metrics.energy_details:
            detail = metrics.energy_details[label]
            detail_lines.append(
                f"{label}: score {detail['score']:.2f} | Vratio {detail['vertical_ratio']:.2f} | Hratio {detail['horizontal_ratio']:.2f}"
            )
    ax_main.text(
        0.02,
        0.02,
        "\n".join(detail_lines),
        transform=ax_main.transAxes,
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.6, "pad": 6},
        verticalalignment="bottom",
    )

    ax_right = fig.add_subplot(gs[1, 1])
    _add_projection_axis(ax_right, metrics.projections["horizontal"], height, metrics.noise_levels["horizontal"], metrics.cut_positions, color_map, "horizontal")

    return fig
