"""최종 요약: 신뢰 빈(|d|/|R2|>1, 0.35-1.5 THz)만 사용, w_x/w_y, w/λ 거듭제곱 피팅, LSF 가우시안성, Rayleigh."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from edgefit import *
df = pd.read_csv(os.path.join(ROOT, "analysis", "table_wf_edges.csv"))
g = df[(df.rel_d > 1.0) & (df.f >= 0.35) & (df.f <= 1.5) & (df.snr_d > 50)].copy()
piv = g.pivot(index="f", columns="edge", values="w").dropna()
rng_ = g.assign(r=(g.rmax-g.rmin)/2).pivot(index="f", columns="edge", values="r").loc[piv.index]
piv["w_x"] = piv[["left", "right"]].mean(1); piv["w_y"] = piv[["top", "bottom"]].mean(1)
piv["sx"] = np.sqrt((piv.left-piv.right)**2/2 + (rng_[["left", "right"]]**2).mean(1)); piv["sy"] = np.sqrt((piv.top-piv.bottom)**2/2 + (rng_[["top", "bottom"]]**2).mean(1))
piv["lam"] = 0.29979/piv.index
# 거듭제곱 피팅 log w = log c + p log f
for c, nm in [("w_x", "x"), ("w_y", "y")]:
    p, c0 = np.polyfit(np.log(piv.index), np.log(piv[c]), 1); print("w_%s ∝ f^%.2f (w(1THz)=%.2f mm);  w/λ mean %.2f ± %.2f" % (nm, p, np.exp(c0), (piv[c]/piv.lam).mean(), (piv[c]/piv.lam).std()))
sel = piv.loc[[f for f in piv.index if min(abs(f-np.array([0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4]))) < 0.011]]
print(sel[["left", "right", "top", "bottom", "w_x", "sx", "w_y", "sy"]].to_string(float_format=lambda v: "%.2f" % v))
piv.to_csv(os.path.join(ROOT, "analysis", "table_wf_final.csv"))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].errorbar(piv.index, piv.w_x, piv.sx, fmt="o", ms=3, label="w_x (L,R avg)"); ax[0].errorbar(piv.index, piv.w_y, piv.sy, fmt="s", ms=3, label="w_y (T,B avg)")
ff = np.linspace(0.35, 1.5, 100); 
for c, col in [("w_x", "C0"), ("w_y", "C1")]:
    p, c0 = np.polyfit(np.log(piv.index), np.log(piv[c]), 1); ax[0].plot(ff, np.exp(c0)*ff**p, col+"-", lw=1)
ax[0].set_xlabel("f [THz]"); ax[0].set_ylabel("w [mm]"); ax[0].legend(); ax[0].set_title("w(f) (reliable bins)")
ax[1].plot(piv.index, piv.w_x/piv.lam, "o", ms=3); ax[1].plot(piv.index, piv.w_y/piv.lam, "s", ms=3); ax[1].set_xlabel("f [THz]"); ax[1].set_ylabel("w/λ"); ax[1].set_title("w/λ")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "s5_wf_final.png"), dpi=130)
# LSF 가우시안성: a_tmpl 정규화 평균 프로파일(가장 신뢰 에지, 라인 정규화) 미분의 첨도
Gt = np.load(os.path.join(CACHE, "maps_tmpl.npz")); XC, YC = x[int(Gt["ix"])], y[int(Gt["iy"])]
tb = pd.read_csv(os.path.join(ROOT, "analysis", "table_broadband_w.csv"))
for edge in ["left", "right", "top", "bottom"]:
    lines = get_lines(edge, Gt["a"], XC, YC); J = joint_fit(lines, +1, 4.0, 2.5)
    cosf = 1/np.sqrt(1+J["m"]**2); P, V = [], []
    for j, i in enumerate(J["idx"]):
        sk, q, v = lines[i]; A_, B_ = J["AB"][j]; P.append((q-(J["c0"]+J["m"]*(sk-J["sbar"])))*cosf); V.append((v-B_)/A_)
    P = np.concatenate(P); V = np.concatenate(V); o = np.argsort(P); P, V = P[o], V[o]
    bins = np.arange(-3, 3.01, 0.1); idx = np.digitize(P, bins); mp = np.array([V[idx == k].mean() if (idx == k).any() else np.nan for k in range(1, len(bins))]); xc = 0.5*(bins[1:]+bins[:-1]); okb = np.isfinite(mp); mp, xc = mp[okb], xc[okb]
    lsf = np.gradient(mp, xc); lsf = np.clip(lsf, 0, None); lsf /= lsf.sum()
    mu = (xc*lsf).sum(); var = ((xc-mu)**2*lsf).sum(); k4 = (((xc-mu)**4)*lsf).sum()/var**2
    print("LSF %s: σ=%.2f mm (w=2σ→%.2f; erf-fit w=%.2f), 첨도=%.2f (가우시안=3)" % (edge, np.sqrt(var), 2*np.sqrt(var), J["w"], k4))
lam = 0.29979/np.array([0.5, 1.0]); 
for f, w in [(0.5, 2.4), (1.0, 1.3)]:
    l = 0.29979/f; print("f=%.1f THz: z_R=πw²/λ=%.1f mm  (vs 0.675 mm Si 두께: 비 %.3f)" % (f, np.pi*w**2/l, 0.675/(np.pi*w**2/l)))
