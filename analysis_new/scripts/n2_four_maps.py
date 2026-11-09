"""4 maps: {raster, traversal(serpentine)} x {peak amplitude (max E), peak-to-peak}. No fitting."""
import os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = r"C:\Users\cojyi\Desktop\beamsize hw"; A = os.path.join(R, "analysis_new")
d = np.load(os.path.join(A, "cache", "raw.npy")); cube = d.reshape(151, 151, 1000)
q = {"peak amplitude (max E)": cube.max(2), "peak-to-peak (max-min)": cube.max(2)-cube.min(2)}
x0, y0, s = 63.04, 39.63, 0.1; ext = [x0-s/2, x0+150*s+s/2, y0-s/2, y0+150*s+s/2]
rough = lambda m: np.mean(np.abs(np.diff(m, axis=0))[40:110, 40:110])
fig, ax = plt.subplots(2, 2, figsize=(13, 11))
for i, (qn, m) in enumerate(q.items()):
    print(qn, "range %.0f .. %.0f nA, 1-99%%: %.0f .. %.0f" % (m.min(), m.max(), *np.percentile(m, [1, 99])))
    for j, (hn, mm) in enumerate([("raster", m), ("traversal", np.where((np.arange(151) % 2 == 1)[:, None], m[:, ::-1], m))]):
        a = ax[i, j]; im = a.imshow(mm, origin="lower", extent=ext, cmap="viridis", interpolation="nearest")
        a.set_title(f"{hn} | {qn}\nrow roughness {rough(mm):.1f} nA, range {mm.min():.0f}-{mm.max():.0f}")
        a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]"); plt.colorbar(im, ax=a)
plt.tight_layout(); plt.savefig(os.path.join(A, "figs", "n2_four_maps.png"), dpi=100)
