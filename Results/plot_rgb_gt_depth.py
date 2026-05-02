from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["text.usetex"] = shutil.which("latex") is not None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_ODL_DIR = PROJECT_ROOT / "scripts" / "pre-train-and-odl"

if str(PRETRAIN_ODL_DIR) not in sys.path:
    sys.path.insert(0, str(PRETRAIN_ODL_DIR))

import utils.idsia_dataloader as idataloader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Plot RGB image and ground-truth depth for one IDSIA/SEC sample"
    )
    parser.add_argument(
        "--idsiadepth_path",
        type=str,
        default=str(PROJECT_ROOT / "micro_sec_mde"),
        help="Root path to the IDSIA/SEC-style dataset.",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index within the selected split.",
    )
    parser.add_argument(
        "--normalize",
        type=int,
        default=1,
        help="Set to 1 to use normalized RGB values from the dataloader.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(PROJECT_ROOT / "Results" / "micro_sec_mde_val_rgb_gt_depth_0000.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=0,
        help="Set to 1 to display the figure interactively.",
    )
    return parser.parse_args()


def to_display_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.float32 and image.dtype != np.float64:
        image = image.astype(np.float32)

    if image.max() > 1.0:
        image = image / 255.0

    return np.clip(image, 0.0, 1.0)


def main() -> None:
    args = parse_args()

    dataset = idataloader.miniIDSIADepth(
        root_dir=args.idsiadepth_path,
        transform=False,
        set=args.dataset_split,
        normalize=bool(args.normalize),
        flip_horizontally=False,
    )

    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(
            f"Index {args.index} is out of range for split '{args.dataset_split}' "
            f"with {len(dataset)} samples."
        )

    sample = dataset[args.index]

    rgb = sample["img"].permute(1, 2, 0).cpu().numpy()
    rgb = to_display_rgb(rgb)

    depth = sample["depth"].cpu().numpy()
    valid_depth = depth[depth > 0]
    vmax = float(valid_depth.max()) if valid_depth.size else 4.0

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    axes[0].imshow(rgb)
    axes[0].set_title(r"$\mathrm{RGB}$", fontsize=24)
    axes[0].set_xlabel(r"$\mathrm{Width}$", fontsize=18)
    axes[0].set_ylabel(r"$\mathrm{Height}$", fontsize=18)

    depth_im = axes[1].imshow(depth, cmap="plasma", vmin=0.0, vmax=vmax)
    axes[1].set_title(r"$\mathrm{Ground\ Truth\ Depth}$", fontsize=24)
    axes[1].set_xlabel(r"$\mathrm{Width}$", fontsize=18)
    axes[1].set_ylabel(r"$\mathrm{Height}$", fontsize=18)

    for axis in axes:
        axis.tick_params(axis="both", labelsize=12)

    colorbar = fig.colorbar(depth_im, ax=axes[1], fraction=0.046, pad=0.04)
    colorbar.set_label(r"$\mathrm{Depth\ (m)}$", fontsize=18)
    colorbar.ax.tick_params(labelsize=12)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Wrote {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
