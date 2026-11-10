"""Export one or many self-contained instances of the parabolic IBVP
(Eq. 1.4, p.4) to .mat files for matlab/solve_reference_pde_batch.m to solve
independently in MATLAB (PDE Toolbox).

Every coefficient/data function -- b_1, b_2, c, beta, f, g, h -- is constant
in (x,t), and g is fixed to 0 (zero initial condition, Eq. 1.4b): a single random scalar per instance, not a function of position or
time. The paper only requires b_i, c, beta, f, h to be Hölder continuous
(Eq. 1.3b-c); it does not prescribe how to construct a concrete random
instance. Using plain constants is the simplest such instance, and keeps this
pipeline (FEM reference vs PNO, see pno.py) easy to reason about.

a_ij is the exception: it is a piecewise-constant field, not one constant for
the whole domain. The disk is split into 4 quadrants by the two lines
x=center_x, y=center_y (top-left/top-right/bottom-left/bottom-right), and
each quadrant gets its own independent random constant matrix (symmetric/
elliptic with eigenvalues in [eig_low, eig_high], Eq. 1.1, Eq. draw_a_ij) --
see draw_a_ij_quadrants. Since each quadrant's matrix is a single draw_a_ij
draw (no blending/interpolation across quadrants), every quadrant's a_ij
satisfies the eigenvalue constraint exactly, the same as the old single-
constant field did.

These values are NOT back-computed from a manufactured exact solution u:
there is no analytic ground truth here. The "exact" solution is whatever
MATLAB's PDE Toolbox parabolic solver produces on a fine finite-element mesh,
treated as the trusted numerical reference.

Usage:
    # run from the repository root (the folder containing this file):
    python export_matlab_problem.py --n_samples 1000 --seed 2000 --out dataset/main/matlab_problems/train
    python export_matlab_problem.py --n_samples 100  --seed 5000 --out dataset/main/matlab_problems/test
"""

import argparse
import os

import numpy as np
from scipy.io import savemat


def draw_constant(rng, amplitude=1.0):
    """A single random scalar, constant in (x,t)."""
    return float(rng.normal(0.0, 1.0) * amplitude)


def draw_a_ij(rng, eig_low, eig_high):
    """A constant, symmetric, elliptic 2x2 matrix with eigenvalues in
    [eig_low, eig_high] (Eq. 1.1)."""
    lam1 = rng.uniform(eig_low, eig_high)
    lam2 = rng.uniform(eig_low, eig_high)
    th = rng.uniform(0.0, np.pi)
    ct, st = np.cos(th), np.sin(th)
    a11 = lam1 * ct ** 2 + lam2 * st ** 2
    a22 = lam1 * st ** 2 + lam2 * ct ** 2
    a12 = (lam1 - lam2) * st * ct
    return a11, a12, a22, lam1, lam2


def draw_a_ij_quadrants(rng, eig_low, eig_high):
    """Draw 4 independent constant, symmetric, elliptic a_ij matrices (Eq. 1.1
    eigenvalues in [eig_low, eig_high]) -- one for each quadrant of the disk,
    split by the two lines x=center_x, y=center_y (top-left, top-right,
    bottom-left, bottom-right). Each is a single draw_a_ij draw, so every
    quadrant's a_ij is exactly elliptic with eigenvalues in
    [eig_low, eig_high] -- no blending across quadrants, no extra clipping.
    """
    tl = draw_a_ij(rng, eig_low, eig_high)
    tr = draw_a_ij(rng, eig_low, eig_high)
    bl = draw_a_ij(rng, eig_low, eig_high)
    br = draw_a_ij(rng, eig_low, eig_high)
    return dict(
        a11_corners=np.array([tl[0], tr[0], bl[0], br[0]]),  # order: TL,TR,BL,BR
        a12_corners=np.array([tl[1], tr[1], bl[1], br[1]]),
        a22_corners=np.array([tl[2], tr[2], bl[2], br[2]]),
    )


def draw_problem_instance(rng, radius, center, T, eig_low, eig_high):
    """Draw one random (a_ij quadrants, b_i, c, beta, f, g, h) instance --
    b_i,c,beta,f,g,h are each a single constant scalar, while a_ij is
    piecewise-constant by quadrant (see draw_a_ij_quadrants) -- returned as a
    dict ready for scipy.io.savemat.
    """
    corners = draw_a_ij_quadrants(rng, eig_low, eig_high)

    b1 = draw_constant(rng, amplitude=0.6)
    b2 = draw_constant(rng, amplitude=0.6)
    c = draw_constant(rng, amplitude=0.6)
    beta = draw_constant(rng, amplitude=0.8)
    f = draw_constant(rng, amplitude=1.0)
    h = draw_constant(rng, amplitude=1.0)
    # Zero initial condition u(x,0) = g = 0 (Eq. 1.4b). The draw is still
    # consumed (and discarded) so the RNG stream -- and hence every other
    # coefficient for a given seed -- matches the earlier random-g data.
    _ = draw_constant(rng, amplitude=1.0)
    g = 0.0

    data = dict(
        radius=radius, center=np.asarray(center, dtype=np.float64), T=T,
        b1=b1, b2=b2, c=c, beta=beta, f=f, h=h, g=g,
        **corners,
    )
    return data, corners


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=0, help="seed for sample 0; sample i uses seed+i")
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--center", type=float, nargs=2, default=(0.0, 0.0))
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--eig_low", type=float, default=1.0, help="a_ij eigenvalue lower bound")
    p.add_argument("--eig_high", type=float, default=2.0, help="a_ij eigenvalue upper bound")
    p.add_argument("--out", type=str, default="dataset/main/matlab_problems/train", help="output directory")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    corner_names = ("TL", "TR", "BL", "BR")
    for i in range(args.n_samples):
        rng = np.random.default_rng(args.seed + i)
        data, corners = draw_problem_instance(
            rng, args.radius, args.center, args.T, args.eig_low, args.eig_high)
        path = os.path.join(args.out, f"problem_instance_{i:05d}.mat")
        savemat(path, data)
        a11c, a12c, a22c = corners["a11_corners"], corners["a12_corners"], corners["a22_corners"]
        corner_str = "  ".join(
            f"{name}=[[{a11c[k]:.3f},{a12c[k]:.3f}],[{a12c[k]:.3f},{a22c[k]:.3f}]]"
            for k, name in enumerate(corner_names))
        print(f"[{i + 1}/{args.n_samples}] wrote {path}\n"
              f"    a_ij corners: {corner_str}\n"
              f"    b1={data['b1']:.4f} b2={data['b2']:.4f} c={data['c']:.4f} "
              f"beta={data['beta']:.4f} f={data['f']:.4f} h={data['h']:.4f} g={data['g']:.4f}")


if __name__ == "__main__":
    main()
