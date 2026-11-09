"""Ellipses (semi-axes = averaged w_x, w_y) for every frequency of chip1_w4_allfreq, numerically failed fits filtered out."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
R = r"C:\Users\cojyi\Desktop\beamsize hw"; A = os.path.join(R, "analysis_new")
t = pd.read_csv(os.path.join(A, "csv", "chip1_w4_allfreq_20261109_1630.csv"))
E = ["xleft", "xright", "ylow", "yhigh"]
ok = np.ones(len(t), bool)
for e in E:
    ok &= np.isfinite(t["w_"+e]) & (t["nok_"+e] >= 20) & (t["rms_"+e] < 0.1) & (t["w_"+e] > 0.25) & (t["w_"+e] < 8)
ok &= np.isfinite(t.w_x_avg) & np.isfinite(t.w_y_avg) & (t.f_THz > 0)
print("filter: all 4 edges finite, >=20/31 lines OK, mean rms<0.1, 0.25<w<8 mm, f>0")
print("kept %d of %d bins; f range of kept: %.2f-%.2f THz" % (ok.sum(), len(t), t.f_THz[ok].min(), t.f_THz[ok].max()))
for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 3), (3, 4), (4, 6), (6, 10.01)]:
    m = (t.f_THz >= lo) & (t.f_THz < hi); print("  %.1f-%.1f THz: kept %3d / %3d" % (lo, hi, (ok & m).sum(), m.sum()))
d = t[ok]; fig, ax = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [1, 1]})
import matplotlib.colors as mc
cmap = plt.cm.turbo; nrm = mc.Normalize(d.f_THz.min(), d.f_THz.max())
for a, lim in zip(ax, [None, 2.6]):
    for _, r in d.iterrows():
        a.add_patch(Ellipse((0, 0), 2*r.w_x_avg, 2*r.w_y_avg, fill=False, lw=.8, color=cmap(nrm(r.f_THz)), alpha=.9))
    a.axhline(0, c="k", lw=.4); a.axvline(0, c="k", lw=.4); a.set_aspect("equal"); a.grid(alpha=.3)
    L = lim or max(d.w_x_avg.max(), d.w_y_avg.max())*1.05; a.set(xlim=(-L, L), ylim=(-L, L), xlabel="x [mm] (x-intercept = avg w_x)", ylabel="y [mm] (y-intercept = avg w_y)")
    a.set_title("all kept bins (%d)" % len(d) if lim is None else "zoom (+-%.1f mm)" % lim)
sm = plt.cm.ScalarMappable(norm=nrm, cmap=cmap); fig.colorbar(sm, ax=ax, label="frequency [THz]", shrink=.8)
fig.suptitle("Effective beam ellipses (1/e^2 radii) per frequency, from chip1_w4_allfreq_20261109_1630.csv (numerical-failure bins removed)", fontsize=12)
plt.savefig(os.path.join(A, "figs", "n16_%s_ellipses_allfreq.png" % "20261109_1637"), dpi=100, bbox_inches="tight")
