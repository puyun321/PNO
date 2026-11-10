"""Plot the a_ij diffusion tensor field (a11, a12, a22, each as a heatmap
over the disk) for every problem_instance_*.mat in a directory -- works for
either schema this project uses:
  - Lipschitz-continuous (matlab/generate_problems_lipschitz.m): a11_grid/
    a12_grid/a22_grid already live on a dense (xg,yg) square grid, so they're
    plotted directly (masked to the disk), no interpolation needed.
  - piecewise-constant-by-quadrant (export_matlab_problem.py): a11_corners/
    a12_corners/a22_corners (4 values) are rasterized onto the same kind of
    square grid first, so the plot shows the sharp quadrant boundaries as-is.

Usage:
    python plot_aij_field.py --problem_dir dataset/lipschitz/matlab_problems/train \
        --out_dir dataset/lipschitz/aij_plots/train
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def quadrant_index(x, y, cx, cy):
    is_right = x >= cx
    is_top = y >= cy
    idx = np.zeros(x.shape, dtype=np.int64)
    idx[is_top & ~is_right] = 0   # TL
    idx[is_top & is_right] = 1    # TR
    idx[~is_top & ~is_right] = 2  # BL
    idx[~is_top & is_right] = 3   # BR
    return idx


def load_a_ij_grid(P, ngrid=101):
    """Returns (xg, yg, a11_grid, a12_grid, a22_grid) on a square grid,
    regardless of which schema the .mat file uses."""
    radius = float(P["radius"])
    cx, cy = float(P["center"][0]), float(P["center"][1])

    if "a11_grid" in P:
        xg = np.asarray(P["xg"], dtype=np.float64).reshape(-1)
        yg = np.asarray(P["yg"], dtype=np.float64).reshape(-1)
        a11_grid = np.asarray(P["a11_grid"], dtype=np.float64)
        a12_grid = np.asarray(P["a12_grid"], dtype=np.float64)
        a22_grid = np.asarray(P["a22_grid"], dtype=np.float64)
    else:
        xg = np.linspace(cx - radius, cx + radius, ngrid)
        yg = np.linspace(cy - radius, cy + radius, ngrid)
        XG, YG = np.meshgrid(xg, yg)
        idx = quadrant_index(XG, YG, cx, cy)
        a11c = np.asarray(P["a11_corners"], dtype=np.float64).reshape(-1)
        a12c = np.asarray(P["a12_corners"], dtype=np.float64).reshape(-1)
        a22c = np.asarray(P["a22_corners"], dtype=np.float64).reshape(-1)
        a11_grid, a12_grid, a22_grid = a11c[idx], a12c[idx], a22c[idx]

    return xg, yg, a11_grid, a12_grid, a22_grid


def plot_one(problem_path, out_path):
    P = loadmat(problem_path, squeeze_me=True)
    radius = float(P["radius"])
    cx, cy = float(P["center"][0]), float(P["center"][1])

    xg, yg, a11_grid, a12_grid, a22_grid = load_a_ij_grid(P)
    XG, YG = np.meshgrid(xg, yg)
    mask = (XG - cx) ** 2 + (YG - cy) ** 2 <= radius ** 2

    theta = np.linspace(0, 2 * np.pi, 200)
    bx, by = cx + radius * np.cos(theta), cy + radius * np.sin(theta)

    fields = [(a11_grid, "$a_{11}$"), (a12_grid, "$a_{12}$"), (a22_grid, "$a_{22}$")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    for ax, (field, name) in zip(axes, fields):
        shown = np.where(mask, field, np.nan)
        im = ax.imshow(shown, extent=[xg[0], xg[-1], yg[0], yg[-1]], origin="lower",
                        cmap="RdBu_r")
        ax.plot(bx, by, color="k", lw=1)
        ax.set_aspect("equal")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=0.85)

    idx = re.search(r"(\d+)\.mat$", problem_path).group(1)
    fig.suptitle(f"problem_instance_{idx}: $a_{{ij}}$ field")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    problem_paths = sorted(glob.glob(os.path.join(args.problem_dir, "problem_instance_*.mat")))
    if not problem_paths:
        raise FileNotFoundError(f"no problem_instance_*.mat files found in {args.problem_dir}")

    for i, problem_path in enumerate(problem_paths):
        idx = re.search(r"(\d+)\.mat$", problem_path).group(1)
        out_path = os.path.join(args.out_dir, f"aij_sample_{idx}.png")
        plot_one(problem_path, out_path)
        if (i + 1) % 100 == 0 or i + 1 == len(problem_paths):
            print(f"[{i + 1}/{len(problem_paths)}] wrote {out_path}")

    print(f"done: wrote {len(problem_paths)} plot(s) to {args.out_dir}")


if __name__ == "__main__":
    main()
