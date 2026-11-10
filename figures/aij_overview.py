import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
os.chdir(ROOT); sys.path.insert(0, ROOT)                                # every path below is relative to it
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.io import loadmat
from plot_aij_field import load_a_ij_grid
ids = [0, 1, 2, 3]
sets = [("Discontinuous", "dataset/main/matlab_problems/test"), ("Lipschitz", "dataset/lipschitz/matlab_problems/test")]
names = ["$a_{11}$", "$a_{12}$", "$a_{22}$"]
fig, axes = plt.subplots(len(sets) * len(ids), 3, figsize=(9, 2.6 * len(sets) * len(ids)), constrained_layout=True)
r = 0
for lab, d in sets:
    for i in ids:
        P = loadmat(f"{d}/problem_instance_{i:05d}.mat", squeeze_me=True)
        R = float(P["radius"]); cx, cy = map(float, P["center"])
        xg, yg, *F = load_a_ij_grid(P, ngrid=201)
        XG, YG = np.meshgrid(xg, yg); m = (XG-cx)**2 + (YG-cy)**2 <= R**2
        th = np.linspace(0, 2*np.pi, 200)
        for j, (f, nm) in enumerate(zip(F, names)):
            ax = axes[r, j]
            im = ax.imshow(np.where(m, f, np.nan), extent=[xg[0], xg[-1], yg[0], yg[-1]], origin="lower", cmap="RdBu_r")
            ax.plot(cx + R*np.cos(th), cy + R*np.sin(th), "k", lw=0.8); ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.8)
            if r == 0: ax.set_title(nm)
            if j == 0: ax.set_ylabel(f"{lab}\ntest {i:05d}")
        r += 1
fig.savefig("results/aij/aij_overview.png", dpi=120)
print("saved results/aij/aij_overview.png")
