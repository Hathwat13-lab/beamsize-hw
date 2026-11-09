"""반마디(|R1-R2| 최대, f=(2k+1)/(2*4.5ps)) 빈만 사용한 w(f) 와 타원. 톱니(체계 편향)의 영향이 가장 작은 지점."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from common import *
d = pd.read_csv(os.path.join(ROOT, "analysis", "table_wf_2d_tiers.csv"))
fa = np.array([(2*k+1)/(2*4.5e-3*1e3)/1e0 for k in range(0, 18)])*1.0   # THz: (2k+1)/(2*4.5 ps)
fa = np.array([(2*k+1)/(2*4.5)*1.0 for k in range(18)])/1.0              # (2k+1)/9 ps^-1 -> THz
rows = []
for f in fa:
    if f < 0.1 or f > 4.0: continue
    i = int(np.argmin(abs(d.f-f))); s = d.iloc[i]
    rows.append(dict(f_anti=f, f_bin=s.f, wx=s.wx, wy=s.wy, ex=s.ex, ey=s.ey, A=s.A, rms=s.rms, rel_d=s.rel_d, snr_d=s.snr_d, at_bound=s.at_bound))
a = pd.DataFrame(rows); a["ok"] = (a.rel_d > 1.0) & (a.snr_d > 50) & ~a.at_bound & (a.rms < 0.25)
pd.set_option("display.width", 200); print(a.to_string(index=False, float_format=lambda v: "%.3g" % v)); a.to_csv(os.path.join(ROOT, "analysis", "table_wf_antinode.csv"), index=False)
g = a[a.ok & (a.f_anti >= 0.3) & (a.f_anti <= 3.2)]
fits = {}
for c in ["wx", "wy"]:
    p, c0 = np.polyfit(np.log(g.f_anti), np.log(g[c]), 1); res = np.log(g[c])-(p*np.log(g.f_anti)+c0); fits[c] = (p, np.exp(c0))
    print("%s: f^%.2f  w(1THz)=%.2f mm  residual %.1f%%  w/lam=%.2f+-%.2f" % (c, p, np.exp(c0), 100*res.std(), (g[c]/(0.29979/g.f_anti)).mean(), (g[c]/(0.29979/g.f_anti)).std()))
fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
ff = np.linspace(0.3, 3.2, 100)
for c, mk, col in [("wx", "o", "C0"), ("wy", "s", "C1")]:
    ax[0].errorbar(g.f_anti, g[c], g["e"+c[-1]], fmt=mk, color=col, ms=5, label=c); ax[0].plot(ff, fits[c][1]*ff**fits[c][0], color=col, lw=1, alpha=.7)
    ax[1].plot(g.f_anti, g[c]/(0.29979/g.f_anti), mk, color=col, label=c)
ax[0].set(xlabel="f [THz]", ylabel="w [mm]", title="w(f) at half-node bins (power-law fit lines)"); ax[0].legend(); ax[0].grid(alpha=.25)
ax[1].set(xlabel="f [THz]", ylabel="w/λ", title="w/λ", ylim=(0, 8)); ax[1].legend(); ax[1].grid(alpha=.25)
cm = plt.cm.viridis; gg = g[(g.f_anti <= 2.4)]
for i, (_, s) in enumerate(gg.iterrows()):
    ax[2].add_patch(Ellipse((0, 0), 2*s.wx, 2*s.wy, fill=False, ec=cm(i/max(len(gg)-1, 1)), lw=2, label="%.2f THz (%.2f×%.2f)" % (s.f_anti, s.wx, s.wy)))
ax[2].set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), aspect="equal", xlabel="x [mm]", ylabel="y [mm]", title="1/e² effective beam ellipses (common axes)"); ax[2].legend(fontsize=7, loc="upper right"); ax[2].grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "s7c_antinode_wf_ellipse.png"), dpi=120)
# 톱니 크기: 각 마디 구간 안에서 w_x의 최대/최소 비 (tier=reliable 빈 기준)
r = d[d.tier == "reliable"].copy(); r["seg"] = np.floor((r.f*4.5e-3*1e3)/1.0 + 0.5)   # 마디 번호
sg = r[(r.f > 0.5) & (r.f < 2.0)].groupby("seg").agg(f0=("f", "mean"), wx_ratio=("wx", lambda v: v.max()/v.min()), wy_ratio=("wy", lambda v: v.max()/v.min()), n=("f", "size"))
print(sg.to_string(float_format=lambda v: "%.2f" % v))
