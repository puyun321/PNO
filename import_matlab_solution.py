"""Resample MATLAB reference solutions (matlab/solve_reference_pde_batch.m
output) onto this pipeline's (interior_xy, boundary_xy, t_mesh) sensor points
and write .npz samples that pno.py / levi_operator.py can consume
directly -- with u_interior/u_boundary coming from an independent PDE-Toolbox
finite element solve instead of a manufactured solution. b_1, b_2, c, beta, f,
g, h are each a single constant scalar per sample (export_matlab_problem.py),
not functions of (x,t). a_ij is the exception, and comes in two schemas:
  - piecewise-constant by quadrant (top-left/top-right/bottom-left/
    bottom-right, split at x=center_x/y=center_y,
    export_matlab_problem.py:draw_a_ij_quadrants / problem_instance_*.mat
    with a11_corners/a12_corners/a22_corners) -- each interior_xy/
    boundary_xy point's quadrant is looked up and assigned that quadrant's
    constant matrix.
  - Lipschitz-continuous, given on a fine grid
    (matlab/generate_problems_lipschitz.m, problem_instance_*.mat
    with a11_grid/a12_grid/a22_grid/xg/yg) -- each point's a_ij is
    bilinearly interpolated off that grid instead.
Either way, every downstream consumer (pno.py, levi_operator.py's
Green's-function reconstruction) only ever sees a_ij as a
per-point array, so nothing past this module needs to know which schema
produced it.

Every problem_instance_NNNNN.mat / reference_solution_NNNNN.mat pair is
converted to <out>/sample_NNNNN.npz, on a SHARED sensor mesh (same
interior_xy/boundary_xy for every sample, as required for operator learning).

Usage:
    # run from the repository root, once per dataset/split
    # (dataset in {main, lipschitz}; split in {train, val, test, ood}; main has no val):
    python import_matlab_solution.py --problem_dir dataset/main/matlab_problems/test         --solution_dir dataset/main/matlab_solutions/test --out dataset/main/samples/test
"""

import argparse
import glob
import os
import re

import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.io import loadmat

from qmc_sampling import sample_disk, sample_circle_boundary


def quadrant_index(xy, center):
    """Which quadrant each point falls in, split by x=center_x/y=center_y:
    0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right -- matches the
    corner order in export_matlab_problem.py:draw_a_ij_quadrants."""
    cx, cy = center
    is_right = xy[:, 0] >= cx
    is_top = xy[:, 1] >= cy
    idx = np.empty(len(xy), dtype=np.int64)
    idx[is_top & ~is_right] = 0   # TL
    idx[is_top & is_right] = 1    # TR
    idx[~is_top & ~is_right] = 2  # BL
    idx[~is_top & is_right] = 3   # BR
    return idx


def grid_lookup(xy, xg, yg, grid):
    """Bilinear a_ij lookup off a Lipschitz-continuous grid (matlab/
    generate_problems_lipschitz.m). grid[i, j] <-> (yg[i], xg[j]),
    the same meshgrid(xg,yg) convention used on the MATLAB side, so the
    interpolator's axes are (yg, xg) in that order. Points are clamped to
    the grid's extent first (mirrors solve_reference_pde_batch.m's
    grid_interp) so floating-point roundoff at the disk boundary -- which
    sits exactly on the grid's edge -- can't fall just outside it."""
    interp = RegularGridInterpolator((yg, xg), grid, method="linear",
                                      bounds_error=False, fill_value=None)
    x = np.clip(xy[:, 0], xg.min(), xg.max())
    y = np.clip(xy[:, 1], yg.min(), yg.max())
    return interp(np.stack([y, x], axis=-1))


def import_one(problem_path, solution_path, interior_xy, boundary_xy, n_in, t_mesh, N, M, Mb):
    P = loadmat(problem_path, squeeze_me=True)
    S = loadmat(solution_path, squeeze_me=True)

    radius = float(P["radius"])
    center = np.asarray(P["center"], dtype=np.float64)
    T = float(P["T"])

    if "a11_grid" in P:
        xg = np.asarray(P["xg"], dtype=np.float64).reshape(-1)
        yg = np.asarray(P["yg"], dtype=np.float64).reshape(-1)
        a11_grid = np.asarray(P["a11_grid"], dtype=np.float64)
        a12_grid = np.asarray(P["a12_grid"], dtype=np.float64)
        a22_grid = np.asarray(P["a22_grid"], dtype=np.float64)
        a11_pts = grid_lookup(interior_xy, xg, yg, a11_grid)
        a12_pts = grid_lookup(interior_xy, xg, yg, a12_grid)
        a22_pts = grid_lookup(interior_xy, xg, yg, a22_grid)
        a11b_pts = grid_lookup(boundary_xy, xg, yg, a11_grid)
        a12b_pts = grid_lookup(boundary_xy, xg, yg, a12_grid)
        a22b_pts = grid_lookup(boundary_xy, xg, yg, a22_grid)
    else:
        a11_corners = np.asarray(P["a11_corners"], dtype=np.float64)
        a12_corners = np.asarray(P["a12_corners"], dtype=np.float64)
        a22_corners = np.asarray(P["a22_corners"], dtype=np.float64)

        interior_q = quadrant_index(interior_xy, center)
        boundary_q = quadrant_index(boundary_xy, center)
        a11_pts, a12_pts, a22_pts = a11_corners[interior_q], a12_corners[interior_q], a22_corners[interior_q]  # (M,)
        a11b_pts, a12b_pts, a22b_pts = a11_corners[boundary_q], a12_corners[boundary_q], a22_corners[boundary_q]  # (Mb,)

    nodes = np.asarray(S["nodes"])                 # (2, numNodes)
    tlist = np.asarray(S["tlist"]).reshape(-1)      # (Nt,)
    u_grid = np.asarray(S["u"])                     # (numNodes, Nt)

    def interp_at(xy, t):
        ti = np.argmin(np.abs(tlist - t))
        vals = griddata(nodes.T, u_grid[:, ti], xy, method="linear")
        nearest = griddata(nodes.T, u_grid[:, ti], xy, method="nearest")
        return np.where(np.isnan(vals), nearest, vals)

    u_interior = np.stack([interp_at(interior_xy, t) for t in t_mesh], axis=0)  # (N, M)
    u_boundary = np.stack([interp_at(boundary_xy, t) for t in t_mesh], axis=0)  # (N, Mb)

    b1_val, b2_val, c_val0, f_val, h_val, g_val = (
        float(P["b1"]), float(P["b2"]), float(P["c"]), float(P["f"]), float(P["h"]), float(P["g"]))

    b1 = np.full((N, M), b1_val)
    b2 = np.full((N, M), b2_val)
    c_val = np.full((N, M), c_val0)
    f = np.full((N, M), f_val)
    beta = np.full((N, Mb), float(P["beta"]))
    h = np.full((N, Mb), h_val)
    g = np.full(M, g_val)

    a11_arr = np.tile(a11_pts, (N, 1))
    a12_arr = np.tile(a12_pts, (N, 1))
    a22_arr = np.tile(a22_pts, (N, 1))
    ab11 = np.tile(a11b_pts, (N, 1))
    ab12 = np.tile(a12b_pts, (N, 1))
    ab22 = np.tile(a22b_pts, (N, 1))
    nu1 = ab11 * n_in[:, 0] + ab12 * n_in[:, 1]
    nu2 = ab12 * n_in[:, 0] + ab22 * n_in[:, 1]

    # f_tilde = f - Lg (Eq. 2.19a). g is constant in x, so all its spatial
    # derivatives are 0 and Lg reduces to just the reaction term c*g.
    f_tilde = f - c_val * g_val

    return dict(
        center=center, radius=radius, T=T, c0=np.nan, c1=np.nan,
        interior_xy=interior_xy, boundary_xy=boundary_xy, boundary_normal=n_in, t_mesh=t_mesh,
        a11=a11_arr, a12=a12_arr, a22=a22_arr, b1=b1, b2=b2, c=c_val, f=f, f_tilde=f_tilde,
        g=g, beta=beta, h=h, a11_b=ab11, a12_b=ab12, a22_b=ab22, nu1=nu1, nu2=nu2,
        u_interior=u_interior, u_boundary=u_boundary,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem_dir", type=str, default="dataset/main/matlab_problems/test")
    p.add_argument("--solution_dir", type=str, default="dataset/main/matlab_solutions/test")
    p.add_argument("--M", type=int, default=80, help="# interior sample points")
    p.add_argument("--Mb", type=int, default=80, help="# boundary sample points")
    p.add_argument("--N", type=int, default=10, help="# time levels")
    p.add_argument("--mesh_seed", type=int, default=1_000_000,
                    help="unused on the default Halton-QMC sampling path; kept for API parity")
    p.add_argument("--out", type=str, default="dataset/main/samples/test")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    problem_paths = sorted(glob.glob(os.path.join(args.problem_dir, "problem_instance_*.mat")))
    if not problem_paths:
        raise FileNotFoundError(f"no problem_instance_*.mat files found in {args.problem_dir}")

    # one shared sensor mesh for every sample (required for operator learning);
    # geometry comes from the first problem instance (radius/center are the
    # same for all of them).
    P0 = loadmat(problem_paths[0], squeeze_me=True)
    radius, center = float(P0["radius"]), np.asarray(P0["center"], dtype=np.float64)
    T = float(P0["T"])
    rng = np.random.default_rng(args.mesh_seed)
    interior_xy = sample_disk(rng, args.M, center, radius)
    boundary_xy, n_in = sample_circle_boundary(rng, args.Mb, center, radius)
    t_mesh = T * (np.arange(1, args.N + 1) / (args.N + 1))

    n_written = 0
    for problem_path in problem_paths:
        idx = re.search(r"(\d+)\.mat$", problem_path).group(1)
        solution_path = os.path.join(args.solution_dir, f"reference_solution_{idx}.mat")
        if not os.path.exists(solution_path):
            print(f"skip {problem_path}: no matching {solution_path}")
            continue
        sample = import_one(problem_path, solution_path, interior_xy, boundary_xy, n_in, t_mesh,
                             args.N, args.M, args.Mb)
        out_path = os.path.join(args.out, f"sample_{idx}.npz")
        np.savez_compressed(out_path, **sample)
        print(f"wrote {out_path}")
        n_written += 1

    print(f"done: {n_written}/{len(problem_paths)} samples written to {args.out}")


if __name__ == "__main__":
    main()
