import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
os.chdir(ROOT); sys.path.insert(0, ROOT)                                # every path below is relative to it
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pno

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load("runs_pno/main/pno.pt", map_location=dev, weights_only=False)
geo = pno.Geometry(np.load(ck["splits"]["test"][0]), dev)
model = pno.PNO(K=ck["K"]).to(dev); model.load_state_dict(ck["state"])
paths = [os.path.join("dataset", "main", "samples", "test", f"sample_{i:05d}.npz") for i in (1, 78)]
errs, preds = pno.score(model, geo, paths, dev, return_preds=True)

N, M, R = geo.N, geo.M, geo.radius
xy = geo.interior_xy
gx = np.linspace(-R, R, 200); GX, GY = np.meshgrid(gx, gx); inside = GX**2 + GY**2 <= R**2
def interp(v):
    g = griddata(xy, v, (GX, GY), method="linear")
    g = np.where(np.isnan(g), griddata(xy, v, (GX, GY), method="nearest"), g)
    return np.where(inside, g, np.nan)
th = np.linspace(0, 2*np.pi, 300)
idxs = np.linspace(0, N - 1, 4).astype(int)

fig = plt.figure(figsize=(11, 15.5), layout="constrained")
subs = fig.subfigures(2, 1, hspace=0.0)
fig.get_layout_engine().set(h_pad=0.01, hspace=0.0)
for sf, p, e, up, tag in zip(subs, paths, errs, preds, "ab"):
    uf = np.load(p)["u_interior"].reshape(N, M); up = np.asarray(up).reshape(N, M); er = up - uf
    vmin, vmax, emax = uf.min(), uf.max(), np.abs(er).max()
    rows = [(uf, r"$u_{\mathrm{FEM}}$", vmin, vmax), (up, r"$\hat u$ (PNO)", vmin, vmax),
            (er, r"$\hat u - u_{\mathrm{FEM}}$", -emax, emax)]
    axes = sf.subplots(3, 4)
    for ri, (F, nm, lo, hi) in enumerate(rows):
        for ci, ti in enumerate(idxs):
            ax = axes[ri, ci]
            im = ax.imshow(interp(F[ti]), extent=[-R, R, -R, R], origin="lower", cmap="RdBu_r", vmin=lo, vmax=hi)
            ax.plot(R*np.cos(th), R*np.sin(th), "k", lw=0.8); ax.set_xticks([]); ax.set_yticks([])
            if ri == 0: ax.set_title(f"$t = {(ti + 1) / (N + 1):.2f}$", fontsize=12)
        axes[ri, 0].set_ylabel(nm, fontsize=13)
        sf.colorbar(im, ax=axes[ri, :].tolist(), shrink=0.9, pad=0.02)
    name = os.path.splitext(os.path.basename(p))[0]
    sf.suptitle(f"({tag})", fontsize=24, fontweight="bold", x=0.01, ha="left")
out = os.path.join("results", "pno", "paper figure"); os.makedirs(out, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out, f"main_test_samples_00001_00078.{ext}"), dpi=300, bbox_inches="tight")
print(errs, "saved")
