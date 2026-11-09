"""1 THz beam visualisation: ellipse from averaged widths + the four edge widths as points on the axes."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
R = r"C:\Users\cojyi\Desktop\beamsize hw"; A = os.path.join(R, "analysis_new")
t = pd.read_csv(os.path.join(A, "csv", "chip1_w4_allfreq_20261109_1630.csv")); r = t.iloc[int(round(1.0/0.02))]
wxl, wxr, wyl, wyh = r.w_xleft, r.w_xright, r.w_ylow, r.w_yhigh; wx, wy = (wxl+wxr)/2, (wyl+wyh)/2
fig, ax = plt.subplots(figsize=(8.5, 7.5))
ax.add_patch(Ellipse((0, 0), 2*wx, 2*wy, fill=False, lw=2, color="C0", label="ellipse from averaged widths\n(1/e$^2$ radii w_x=%.3f, w_y=%.3f mm)" % (wx, wy)))
pts = [(+wxl, 0, "xleft edge -> beam right part\nw=%.3f" % wxl, "C1"), (-wxr, 0, "xright edge -> beam left part\nw=%.3f" % wxr, "C3"),
       (0, +wyl, "ylow edge -> beam +y part\nw=%.3f" % wyl, "C2"), (0, -wyh, "yhigh edge -> beam -y part\nw=%.3f" % wyh, "C4")]
for px, py, lab, c in pts:
    ax.plot(px, py, "o", ms=9, c=c); ax.annotate(lab, (px, py), textcoords="offset points", fontsize=9, color=c,
        xytext=(8, 8) if px > 0 else (-8, 8) if px < 0 else (8, 10 if py > 0 else -26), ha="left" if px >= 0 else "right")
ax.axhline(0, c="k", lw=.5); ax.axvline(0, c="k", lw=.5); ax.plot(0, 0, "k+", ms=10)
ax.set(xlim=(-2.4, 2.4), ylim=(-2.0, 2.0), aspect="equal", xlabel="x relative to beam centre [mm]", ylabel="y relative to beam centre [mm]",
       title="1.00 THz effective beam (1/e$^2$ intensity radius)"); ax.grid(alpha=.3); ax.legend(loc="lower left", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(A, "figs", "n14_%s_beam_1THz.png" % "20261109_1632"), dpi=110)
print("wx_avg %.3f wy_avg %.3f ratio %.3f | asym x (left-right) %.3f, y (low-high) %.3f" % (wx, wy, wx/wy, wxl-wxr, wyl-wyh))
