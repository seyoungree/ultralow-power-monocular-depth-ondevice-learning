from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


plt.rcParams["text.usetex"] = shutil.which("latex") is not None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "Results" / "No dropout_IDSIA finetuning" / "no dropout_IDSIA finetuning.csv"
OUTPUT_PATH = PROJECT_ROOT / "Results" / "No dropout_IDSIA finetuning" / "no dropout_IDSIA finetuning.png"


def load_tensorboard_csv(csv_path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            steps.append(int(row["Step"]))
            values.append(float(row["Value"]))

    if not steps:
        raise ValueError(f"No rows found in {csv_path}")

    return steps, values


def main() -> None:
    steps, values = load_tensorboard_csv(CSV_PATH)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, values, color="tab:blue", linewidth=2.0, marker="o", markersize=4)
    ax.set_xlabel(r"$\mathrm{Epoch}$", fontsize=30)
    ax.set_ylabel(r"$\mathrm{Abs\ Rel}$", fontsize=30)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
