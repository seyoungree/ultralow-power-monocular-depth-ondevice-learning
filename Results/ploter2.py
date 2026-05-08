from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


plt.rcParams["text.usetex"] = shutil.which("latex") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[1]

ABS_REL_PLOTS = [
    {
        "csv_path": PROJECT_ROOT / "Results" / "Dropout_No finetuning" / "abs_rel.csv",
        "output_path": PROJECT_ROOT / "Results" / "Dropout_No finetuning" / "abs_rel.png",
        "show_ylabel": True,
    },
    {
        "csv_path": PROJECT_ROOT / "Results" / "Dropout_IDSIA finetuning" / "abs_rel.csv",
        "output_path": PROJECT_ROOT / "Results" / "Dropout_IDSIA finetuning" / "abs_rel.png",
        "show_ylabel": False,
    },
    {
        "csv_path": PROJECT_ROOT / "Results" / "Dropout_SEC finetuning" / "abs_rel.csv",
        "output_path": PROJECT_ROOT / "Results" / "Dropout_SEC finetuning" / "abs_rel.png",
        "show_ylabel": False,
    },
    {
        "csv_path": PROJECT_ROOT / "Results" / "No dropout_No finetuning" / "No dropout_No finetuning_abs_rel.csv",
        "output_path": PROJECT_ROOT / "Results" / "No dropout_No finetuning" / "abs_rel.png",
        "show_ylabel": True,
    },
    {
        "csv_path": PROJECT_ROOT / "Results" / "No dropout_IDSIA finetuning" / "abs_rel.csv",
        "output_path": PROJECT_ROOT / "Results" / "No dropout_IDSIA finetuning" / "abs_rel.png",
        "show_ylabel": False,
    },
    {
        "csv_path": PROJECT_ROOT / "Results" / "No dropout_SEC finetuning" / "abs_rel.csv",
        "output_path": PROJECT_ROOT / "Results" / "No dropout_SEC finetuning" / "abs_rel.png",
        "show_ylabel": False,
    },
]

D1_PLOTS = [
    {
        "csv_path": PROJECT_ROOT / "Results" / "Dropout_No finetuning" / "d1.csv",
        "output_path": PROJECT_ROOT / "Results" / "Dropout_No finetuning" / "d1.png",
        "show_ylabel": True,
    },
    {
        "csv_path": PROJECT_ROOT / "Results" / "Dropout_IDSIA finetuning" / "d1.csv",
        "output_path": PROJECT_ROOT / "Results" / "Dropout_IDSIA finetuning" / "d1.png",
        "show_ylabel": False,
    },
    {
        "csv_path": PROJECT_ROOT / "Results" / "Dropout_SEC finetuning" / "d1.csv",
        "output_path": PROJECT_ROOT / "Results" / "Dropout_SEC finetuning" / "d1.png",
        "show_ylabel": False,
    },
    {
        "csv_path": PROJECT_ROOT / "Results" / "No dropout_No finetuning" / "d1.csv",
        "output_path": PROJECT_ROOT / "Results" / "No dropout_No finetuning" / "d1.png",
        "show_ylabel": True,
    },
    {
        "csv_path": PROJECT_ROOT / "Results" / "No dropout_IDSIA finetuning" / "d1.csv",
        "output_path": PROJECT_ROOT / "Results" / "No dropout_IDSIA finetuning" / "d1.png",
        "show_ylabel": False,
    },
    {
        "csv_path": PROJECT_ROOT / "Results" / "No dropout_SEC finetuning" / "d1.csv",
        "output_path": PROJECT_ROOT / "Results" / "No dropout_SEC finetuning" / "d1.png",
        "show_ylabel": False,
    },
]


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


def plot_metric(csv_path: Path, output_path: Path, show_ylabel: bool, ylabel: str) -> None:
    steps, values = load_tensorboard_csv(csv_path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, values, color="tab:blue", linewidth=2.0, marker="o", markersize=4)
    ax.set_xlabel(r"$\mathrm{Epoch}$", fontsize=30)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=30)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for spec in ABS_REL_PLOTS:
        plot_metric(**spec, ylabel=r"$\mathrm{Abs\ Rel}$")
        print(f"Wrote {spec['output_path']}")
    for spec in D1_PLOTS:
        plot_metric(**spec, ylabel=r"$\delta_1$")
        print(f"Wrote {spec['output_path']}")


if __name__ == "__main__":
    main()
