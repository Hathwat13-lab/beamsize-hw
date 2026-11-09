"""2D 피팅 결과(table_wf_2d_fast.csv): 신뢰도 등급, w(f) 곡선, 0.1–4 THz 빔 타원 플롯."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from common import *
d = pd.read_csv(os.path.join(ROOT, "analysis", "table_wf_2d_fast.csv"))
rel = (d.rel_d > 1.0) & (d.snr_d > 100) & (d.rms < 0.15) & ~d.at_bound & (d.A > 0.8) & (d.A < 1.2)
mar = (d.rel_d > 0.5) & (d.snr_d > 50) & (d.rms < 0.3) & ~d.at_bound & ~rel
d["tier"] = np.where(rel, "reliable", np.where(mar, "marginal", "unreliable"))
d["ex"] = np.maximum(d.wx_hi-d.wx, d.wx-d.wx_lo); d["ey"] = np.maximum(d.wy_hi-d.wy, d.wy-d.wy_lo)
d.to_csv(os.path.join(ROOT, "analysis", "table_wf_2d_tiers.csv"), index=False)
print(d.tier.value_counts().to_string())
r = d[d.tier == "reliable"]; print("reliable f range: %.2f-%.2f THz; bins=%d" % (r.f.min(), r.f.max(), len(r)))
print("reliable f list:", np.round(r.f.values, 2).tolist())
for c, nm in [("wx", "x"), ("wy", "y")]:
    rr = r[(r.f >= 0.3) & (r.f <= 1.6)]; p, c0 = np.polyfit(np.log(rr.f), np.log(rr[c]), 1); lam = 0.29979/rr.f
    print("w_%s: f^%.2f, w(1THz)=%.2f mm; w/λ = %.2f ± %.2f (0.3-1.6 THz, reliable)" % (nm, p, np.exp(c0), (rr[c]/lam).mean(), (rr[c]/lam).std()))
# ---- w(f) 곡선 ----
col = {"reliable": "C0", "marginal": "C1", "unreliable": "0.75"}
fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
for t_ in ["unreliable", "marginal", "reliable"]:
    s = d[d.tier == t_]
    for a, c, e, mk in [(ax[0], "wx", "ex", "o"), (ax[1], "wy", "ey", "s")]:
        a.errorbar(s.f, s[c], s[e], fmt=mk, ms=3.5, color=col[t_], label=t_, lw=.8)
    lam = 0.29979/s.f; ax[2].plot(s.f, s.wx/lam, "o", ms=3.5, color=col[t_]); ax[2].plot(s.f, s.wy/lam, "s", ms=3.5, color=col[t_], mfc="none")
for a, t_ in zip(ax, ["w_x [mm] (1/e² radius, effective)", "w_y [mm]", "w/λ  (filled = w_x, open = w_y)"]):
    a.set_xlabel("f [THz]"); a.set_title(t_); a.grid(alpha=.25)
ax[0].set_ylim(0, 8); ax[1].set_ylim(0, 8); ax[2].set_ylim(0, 12); ax[0].legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "s7_wf_2d.png"), dpi=125)
# ---- 타원 갤러리 (0.1 THz 간격, 패널별 축, 1/e² 윤곽) ----
sel = d[(d.f*10).round().sub(d.f*10).abs() < 1e-6].copy(); sel = sel[(sel.f >= 0.099)].reset_index(drop=True)
nc = 8; nr = int(np.ceil(len(sel)/nc)); fig, ax = plt.subplots(nr, nc, figsize=(2.9*nc, 3.1*nr)); ax = ax.ravel()
for a, (_, s) in zip(ax, sel.iterrows()):
    L = 1.35*max(s.wx, s.wy); g = np.linspace(-L, L, 200); X, Y = np.meshgrid(g, g); I = np.exp(-2*(X**2/s.wx**2+Y**2/s.wy**2))
    a.imshow(I, extent=[-L, L, -L, L], origin="lower", cmap="viridis", vmin=0, vmax=1, alpha=1 if s.tier == "reliable" else .45)
    a.add_patch(Ellipse((0, 0), 2*s.wx, 2*s.wy, fill=False, ec="w" if s.tier != "unreliable" else "k", lw=1.4, ls="-" if s.tier == "reliable" else "--"))
    a.set_aspect("equal"); a.set_title("%.1f THz  %.2f×%.2f%s" % (s.f, s.wx, s.wy, "" if s.tier == "reliable" else ("  (marg.)" if s.tier == "marginal" else "  (unrel.)")), fontsize=8.5, color="k" if s.tier == "reliable" else "0.4")
    a.tick_params(labelsize=7); a.set_xlabel("x [mm]", fontsize=7); a.set_ylabel("y [mm]", fontsize=7)
for a in ax[len(sel):]: a.set_visible(False)
fig.suptitle("Effective beam (1/e² intensity radius ellipse, semi-axes w_x × w_y [mm]); per-panel axes; dashed/faded = not reliable", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "s7_ellipse_gallery_0p1_4THz.png"), dpi=105)
# ---- 겹쳐 그린 타원(공통 축, 신뢰 주파수만) ----
rr = r[(r.f >= 0.3) & (r.f <= 1.6)]; fs = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.5]
fig, ax = plt.subplots(figsize=(7.2, 6.4)); cm = plt.cm.viridis
for i, f in enumerate(fs):
    s = d.iloc[int(np.argmin(abs(d.f-f)))]; ax.add_patch(Ellipse((0, 0), 2*s.wx, 2*s.wy, fill=False, ec=cm(i/(len(fs)-1)), lw=2, label="%.1f THz  (%.2f×%.2f)" % (s.f, s.wx, s.wy), ls="-" if s.tier == "reliable" else "--"))
ax.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), aspect="equal", xlabel="x [mm]", ylabel="y [mm]", title="1/e² beam ellipses, common axes (dashed = marginal/unreliable)"); ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "s7_ellipse_overlay.png"), dpi=125)
print(d[d.f.round(3).isin([0.1, 0.3, 0.5, 1.0, 1.488, 1.5, 3.0, 4.0])][["f", "wx", "wy", "A", "rms", "tier"]].to_string(index=False, float_format=lambda v: "%.3g" % v))
