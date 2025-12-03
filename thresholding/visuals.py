import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_decision_boundaries(data, t8, t4):
    """Visualize decision regions in Score_4 vs Score_8 space."""
    s4 = [d["s4"] for d in data]
    s8 = [d["s8"] for d in data]
    labels = [d["true_label"] for d in data]

    plt.figure(figsize=(10, 8))

    color_map = {"2x2": "blue", "4x4": "green", "8x8": "red"}
    colors = [color_map[label] for label in labels]

    plt.scatter(s4, s8, c=colors, edgecolors="k", s=50, alpha=0.7, zorder=10)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", label="True 2x2"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green", label="True 4x4"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", label="True 8x8"),
    ]

    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()

    x_max = max(x_max, max(s4) * 1.1)
    y_max = max(y_max, max(s8) * 1.1)

    plt.axhline(y=t8, color="black", linestyle="--", linewidth=2, label=f"T8={t8:.0f}")
    plt.plot([t4, t4], [y_min, t8], "k--", linewidth=2, label=f"T4={t4:.0f}")

    plt.fill_between([x_min, x_max], t8, y_max, color="red", alpha=0.1)
    plt.text(
        x_min + (x_max - x_min) * 0.05,
        t8 + (y_max - t8) * 0.5,
        "PREDICTION: 8x8",
        color="red",
        fontweight="bold",
    )

    plt.fill_between([t4, x_max], y_min, t8, color="green", alpha=0.1)
    plt.text(
        t4 + (x_max - t4) * 0.5,
        y_min + (t8 - y_min) * 0.5,
        "PREDICTION: 4x4",
        color="green",
        fontweight="bold",
    )

    plt.fill_between([x_min, t4], y_min, t8, color="blue", alpha=0.1)
    plt.text(
        x_min + (t4 - x_min) * 0.5,
        y_min + (t8 - y_min) * 0.5,
        "PREDICTION: 2x2",
        color="blue",
        fontweight="bold",
    )

    plt.title("General Optimum: 3-Class Decision Boundaries")
    plt.xlabel("Score 4x4")
    plt.ylabel("Score 8x8")
    plt.legend(handles=legend_elements, loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.show()
