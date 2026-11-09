"""Ellipses at equally spaced (0.222 THz, antinode) frequencies inside the trusted range 0.556-3.667 THz."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.colors as mc
from matplotlib.patches import Ellipse
R = r"C:\Users\cojyi\Desktop\beamsize hw"; A = os.path.join(R, "analysis_new")
t = pd.read_csv(os.path.join(A, "csv", "chip1_w4_allfreq_20261109_1630.csv"))
ks = [k for k in range(0, 30) if 0.556-1e-3 <= (2*k+1)/9 <= 3.667+1e-3]
d = t.iloc[[int(round(((2*k+1)/9)/0.02)) for k in ks]].copy(); d["f_target"] = [(2*k+1)/9 for k in ks]
d.to_csv(os.path.join(A, "csv", "n17_%s_equispaced_w.csv" % "20261109_1639"), index=False, float_format="%.5g")
print(d[["f_target", "f_THz", "w_x_avg", "w_y_avg"]].round(3).to_string(index=False)); print("N =", len(d))
cmap = plt.cm.turbo; nrm = mc.Normalize(d.f_target.min(), d.f_target.max())
fig, ax = plt.subplots(1, 2, figsize=(17, 7.5), gridspec_kw={"width_ratios": [1.15, 1]})
a = ax[0]
for _, r in d.iterrows():
    c = cmap(nrm(r.f_target)); a.add_patch(Ellipse((0, 0), 2*r.w_x_avg, 2*r.w_y_avg, fill=False, lw=1.8, color=c))
    a.annotate("%.2f" % r.f_target, (0, r.w_y_avg), fontsize=6.5, color=c, ha="center", va="bottom") if r.f_target > 1.2 else None
a.axhline(0, c="k", lw=.4); a.axvline(0, c="k", lw=.4); a.set_aspect("equal"); a.grid(alpha=.3)
a.set(xlim=(-2.4, 2.4), ylim=(-2.4, 2.4), xlabel="x [mm] (intercept = avg w_x)", ylabel="y [mm] (intercept = avg w_y)", title="beam ellipses, f = (2k+1)/9 THz, 0.556-3.667 THz (N=%d)" % len(d))
sm = plt.cm.ScalarMappable(norm=nrm, cmap=cmap); fig.colorbar(sm, ax=a, label="frequency [THz]", shrink=.75)
b = ax[1]
for e, mk, lab in [("xleft", "<", "x left edge"), ("xright", ">", "x right edge"), ("ylow", "v", "y low edge"), ("yhigh", "^", "y high edge")]:
    b.plot(d.f_target, d["w_"+e], mk, ms=5, alpha=.7, label=lab)
b.plot(d.f_target, d.w_x_avg, "-", c="C0", lw=1.5, label="avg w_x"); b.plot(d.f_target, d.w_y_avg, "-", c="C3", lw=1.5, label="avg w_y")
b.set(xlabel="frequency [THz]", ylabel="w [mm]", title="widths at the same frequencies"); b.grid(alpha=.3); b.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(A, "figs", "n17_%s_ellipses_equispaced.png" % "20261109_1639"), dpi=100)
