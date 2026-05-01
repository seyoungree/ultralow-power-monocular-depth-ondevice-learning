from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "scripts/pre-train-and-odl/train_log_upydnet_det_20260423-173505.txt"
OUTPUT_DIR = PROJECT_ROOT / "Results" / "No dropout_No finetuning"
CSV_PATH = OUTPUT_DIR / "no dropout_No finetuning.csv"
PLOT_PATH = OUTPUT_DIR / "no dropout_No finetuning.png"

EPOCH_PATTERN = re.compile(r">>> EPOCH (\d+) <<<")
LOSS_PATTERN = re.compile(r"LR = ([0-9.eE+-]+), avg_loss = ([0-9.eE+-]+)")
METRIC_PATTERN = re.compile(r"([A-Za-z0-9_]+)\s*=\s*([0-9.eE+-]+)")


def parse_training_log(log_path: Path) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    current_epoch: dict[str, float] | None = None

    for raw_line in log_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        epoch_match = EPOCH_PATTERN.match(line)
        if epoch_match:
            current_epoch = {"epoch": int(epoch_match.group(1))}
            history.append(current_epoch)
            continue

        if current_epoch is None:
            continue

        loss_match = LOSS_PATTERN.match(line)
        if loss_match:
            current_epoch["lr"] = float(loss_match.group(1))
            current_epoch["avg_loss"] = float(loss_match.group(2))
            continue

        metric_match = METRIC_PATTERN.match(line)
        if metric_match:
            current_epoch[metric_match.group(1)] = float(metric_match.group(2))

    if not history:
        raise ValueError(f"No epochs were parsed from {log_path}")

    return history


def write_csv(history: list[dict[str, float]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "lr",
        "avg_loss",
        "abs_rel",
        "sq_rel",
        "rmse",
        "rmse_log",
        "d1",
        "d2",
        "d3",
    ]

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to render the plot. Install it in the active Python "
            "environment, then rerun Results/plotter.py."
        ) from exc

    plt.rcParams["text.usetex"] = shutil.which("latex") is not None

    epochs = [int(row["epoch"]) for row in history]
    metric_specs = [
        ("avg_loss", "Average training loss"),
        ("abs_rel", "Abs Rel"),
        ("rmse", "RMSE"),
        ("d1", r"$\delta_1$ accuracy"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    axes = axes.ravel()

    for axis, (metric, label) in zip(axes, metric_specs):
        values = [row[metric] for row in history]
        axis.plot(epochs, values, marker="o", linewidth=1.8, markersize=3.5)
        axis.set_title(label)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)

    fig.suptitle("No Dropout, No Finetuning")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    history = parse_training_log(LOG_PATH)
    write_csv(history, CSV_PATH)
    plot_history(history, PLOT_PATH)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {PLOT_PATH}")


if __name__ == "__main__":
    main()
