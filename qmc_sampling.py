"""Halton quasi-Monte-Carlo sampling utilities for the disk domain D and its
boundary (dD) (paper Eq. 3.2-3.3, Definition 3.1). Shared by the active
FEM / Green's-function / 2D-CNN comparison pipeline (import_matlab_solution.py)
so it doesn't depend on the legacy manufactured-solution pipeline in
previous/generate_samples.py.
"""

import numpy as np


def halton_radical_inverse(m, base):
    """phi_b(m), Definition 3.1 (p.18): the b-adic radical inverse function."""
    result, f = 0.0, 1.0 / base
    while m > 0:
        result += f * (m % base)
        m //= base
        f /= base
    return result


def halton_sequence(n_points, base, start=1):
    """1D Halton/van der Corput sequence in base `base`, Definition 3.1/Example 3.2.
    start=1 skips phi_b(0)=0 (the paper's own examples start at m=0, but m=0 maps to a
    degenerate point at the sampling boundary, e.g. theta=0 or r=0, so we skip it)."""
    return np.array([halton_radical_inverse(m, base) for m in range(start, start + n_points)])


def sample_disk(rng, n_points, center, radius, use_qmc=True):
    """Eq. 3.3a: D_mesh_T's spatial part is S_1(H_{2,M}), a 2D Halton sequence
    mapped onto D via the area-preserving polar transform r=radius*sqrt(u),
    theta=2*pi*v (rather than the paper's generic affine-map-then-intersect,
    since for a disk this transform lands exactly on D with no rejected points)."""
    if use_qmc:
        u, v = halton_sequence(n_points, 2), halton_sequence(n_points, 3)
    else:
        u, v = rng.uniform(0.0, 1.0, n_points), rng.uniform(0.0, 1.0, n_points)
    r = radius * np.sqrt(u)
    theta = 2 * np.pi * v
    return np.stack([center[0] + r * np.cos(theta), center[1] + r * np.sin(theta)], axis=1)


def sample_circle_boundary(rng, n_points, center, radius, use_qmc=True):
    """Eq. 3.3b: (dD)_mesh_T's spatial part is S_2(H_{1,M}) = gamma(2*pi*H_{1,M}),
    a 1D Halton sequence mapped through the boundary parametrization."""
    v = halton_sequence(n_points, 2) if use_qmc else rng.uniform(0.0, 1.0, n_points)
    theta = 2 * np.pi * v
    xy = np.stack([center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta)], axis=1)
    inward_normal = -np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return xy, inward_normal
