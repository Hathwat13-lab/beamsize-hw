"""2P-1 profiles at 1 THz: along x at central row, along y at central column (no fitting)."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = r"C:\Users\cojyi\Desktop\beamsize hw"; A = os.path.join(R, "analysis_new")
df = pd.read_csv(os.path.join(A, "csv", "chip1_P_ij_f_20261109_1559.csv"))
j = list(df.columns[4:]).index("1.00"); P = df.iloc[:, 4+j].values.reshape(151, 151)   # [iy, ix]
x = 63.04 + 0.1*np.arange(151); y = 39.63 + 0.1*np.arange(151); iy0, ix0 = 75, 80
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
for d in range(-10, 11, 5):
    if d: ax[0].plot(x, 2*P[iy0+d]-1, c="gray", lw=.6, alpha=.6); ax[1].plot(y, 2*P[:, ix0+d]-1, c="gray", lw=.6, alpha=.6)
ax[0].plot(x, 2*P[iy0]-1, "o-", ms=2.5, lw=1, c="C0", label="y = %.2f mm (iy=%d)" % (y[iy0], iy0))
ax[1].plot(y, 2*P[:, ix0]-1, "o-", ms=2.5, lw=1, c="C3", label="x = %.2f mm (ix=%d)" % (x[ix0], ix0))
ax[0].plot([], [], c="gray", lw=.6, label="rows iy=65,70,80,85,..."); ax[1].plot([], [], c="gray", lw=.6, label="cols ix=70,75,85,90")
for a, xl in zip(ax, ["x_b [mm]", "y_b [mm]"]):
    a.axhline(1, c="k", ls=":", lw=.6); a.axhline(-1, c="k", ls=":", lw=.6); a.axhline(0, c="k", ls=":", lw=.4)
    a.set(xlabel=xl, ylabel="2P - 1", title="1.00 THz: 2P-1 vs " + xl.split()[0], ylim=(-1.4, 1.4)); a.grid(alpha=.3); a.legend(fontsize=8, loc="lower center")
plt.tight_layout(); plt.savefig(os.path.join(A, "figs", "n10_%s_2P1_1THz.png" % "20261109_1605"), dpi=100)
r = 2*P[iy0]-1; c = 2*P[:, ix0]-1
print("row min/max: %.3f %.3f ; col min/max: %.3f %.3f" % (r.min(), r.max(), c.min(), c.max()))
