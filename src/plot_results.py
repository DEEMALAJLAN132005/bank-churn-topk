"""Plot Precision@k and Recall@k from saved evaluation results."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from . import config as C


def main():
    data = pd.read_csv(C.RES_DIR / "topk_comparison.csv")

    models = [
        ("logistic_regression", "Logistic Regression", "#496A81"),
        ("random_forest", "Random Forest", "#D17B35"),
        ("mlp", "MLP", "#34846F"),
    ]
    capacities = [0.10, 0.20]
    metrics = [
        ("precision_at_k", "Precision@k"),
        ("recall_at_k", "Recall@k"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    x = np.arange(len(capacities))
    width = 0.24

    for ax, (metric, title) in zip(axes, metrics):
        for position, (model, label, color) in enumerate(models):
            values = []

            for capacity in capacities:
                selected = data[
                    (data["model"] == model)
                    & np.isclose(data["contact_fraction"], capacity)
                ]

                if len(selected) != 1:
                    raise ValueError(
                        f"Expected one row for {model} at {capacity}."
                    )

                value = float(selected.iloc[0][metric])
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid value for {model}: {metric}")

                values.append(value)

            bars = ax.bar(
                x + (position - 1) * width,
                values,
                width,
                label=label,
                color=color,
            )

            ax.bar_label(
                bars,
                labels=[f"{value:.1%}" for value in values],
                padding=4,
                fontsize=9,
            )

        ax.set_title(title)
        ax.set_xticks(x, ["Top 10%", "Top 20%"])
        ax.set_xlabel("Contact capacity")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Model comparison at fixed contact capacities")
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])

    output_dir = C.RES_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_dir / "topk_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "topk_comparison.svg",
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Figures saved in: {output_dir}")


if __name__ == "__main__":
    main()