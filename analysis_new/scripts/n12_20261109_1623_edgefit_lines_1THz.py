"""1 THz: per-line erf fits along each of the 4 edges (lines parallel to scan axis), w vs position along the edge."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
TS = "20261109_1623"
R = r"C:\Users\cojyi\Desktop\beamsize hw"; A = os.path.join(R, "analysis_new")
df = pd.read_csv(os.path.join(A, "csv", "chip1_P_ij_f_20261109_1559.csv"))
j = list(df.columns[4:]).index("1.00"); P = df.iloc[:, 4+j].values.reshape(151, 151); V = 2*P - 1       # [iy, ix]
x = 63.04 + 0.1*np.arange(151); y = 39.63 + 0.1*np.arange(151)
HW, STRIP, FAR = 2.0, 1.5, 2.5                  # fit half-window [mm]; used strip half-length along edge [mm]; plotted half-length
def model(u, a, b, u0, w, sg): return sg*a*erf(np.sqrt(2)*(u-u0)/w) + b
def crossing(u, v, sg):
    s = [i for i in np.where(np.diff(np.sign(v)) != 0)[0] if (v[i+1]-v[i])*sg > 0]
    if not s: return None
    i = s[0] if sg > 0 else s[-1]; return u[i] - v[i]*(u[i+1]-u[i])/(v[i+1]-v[i])
def fit(u, v, sg, hw=HW):
    u0 = crossing(u, v, sg)
    if u0 is None: return None
    m = np.abs(u-u0) <= hw
    try: p, cov = curve_fit(lambda uu, a, b, c, w: model(uu, a, b, c, w, sg), u[m], v[m], p0=[1, 0, u0, 1.4], maxfev=5000)
    except Exception: return None
    return p, np.sqrt(np.diag(cov)), np.sqrt(np.mean((v[m]-model(u[m], *p, sg))**2))
# edges: name, axis of profile, sign, which lines (index along edge), center index of the strip, coordinate of lines
edges = [("x1 (left)", "x", +1, 75, y), ("x2 (right)", "x", -1, 75, y), ("y1 (low-y)", "y", +1, 80, x), ("y2 (high-y)", "y", -1, 80, x)]
rows = []; res = {}
for nm, ax_, sg, c, pos in edges:
    u = x if ax_ == "x" else y; rec = []
    for k in range(c-int(FAR*10), c+int(FAR*10)+1):
        v = V[k] if ax_ == "x" else V[:, k]; r = fit(u, v, sg)
        if r is None: continue
        p, e, rms = r; rec.append((pos[k]-pos[c], p[3], e[3], p[2], p[0], p[1], rms, k))
        rows.append([nm, k, pos[k], p[3], e[3], p[2], p[0], p[1], rms])
    res[nm] = np.array(rec)
pd.DataFrame(rows, columns=["edge", "line_idx", "line_pos_mm", "w_mm", "w_sigma", "u0_mm", "A", "B", "rms"]).to_csv(os.path.join(A, "csv", "n12_%s_edgefit_lines_1THz.csv" % TS), index=False, float_format="%.5f")
fig, ax = plt.subplots(3, 4, figsize=(18, 12)); summ = {}
for a_, (nm, ax_, sg, c, pos) in enumerate(edges):
    r = res[nm]; used = np.abs(r[:, 0]) <= STRIP + 1e-9; wu = r[used, 1]
    summ[nm] = (wu.mean(), wu.std(), np.median(wu), used.sum())
    # variants of window on the mean
    u = x if ax_ == "x" else y; vv = []
    for hw in (1.5, 2.5, 3.0):
        ws = [fit(u, V[int(k)] if ax_ == "x" else V[:, int(k)], sg, hw) for k in r[used, 7]]; vv.append(np.mean([w[0][3] for w in ws if w]))
    summ[nm] += (min(vv + [wu.mean()]), max(vv + [wu.mean()]))
    print("%-12s lines %d  w mean %.3f  std %.3f  median %.3f | mean over windows 1.5/2.5/3: %s -> range %.3f-%.3f" % (nm, used.sum(), *summ[nm][:3], np.round(vv, 3), summ[nm][4], summ[nm][5]))
    cm = plt.cm.coolwarm; o = ax[0, a_]
    for rr in r[used][::3]:
        k = int(rr[7]); v = V[k] if ax_ == "x" else V[:, k]; uu = np.linspace(rr[3]-3, rr[3]+3, 200); m = np.abs(u-rr[3]) <= 3
        col = cm((rr[0]+STRIP)/(2*STRIP)); o.plot(u[m]-rr[3], v[m], ".", ms=2, c=col); o.plot(uu-rr[3], model(uu, rr[4], rr[5], rr[3], rr[1], sg), c=col, lw=.8)
    o.set(title="%s: lines in strip +-%.1f mm\n(aligned at each line's u0)" % (nm, STRIP), xlabel="u - u0 [mm]", ylabel="2P-1", ylim=(-1.3, 1.3)); o.grid(alpha=.3)
    o = ax[1, a_]; o.plot(r[:, 0], r[:, 1], "o-", ms=3, lw=.8, c="C%d" % a_); o.axvspan(-STRIP, STRIP, alpha=.1)
    o.axhline(summ[nm][0], c="k", ls="--", lw=.8); o.set(title="w = %.3f +- %.3f mm (std over %d lines)\nmedian %.3f, window range %.2f-%.2f" % (summ[nm][0], summ[nm][1], summ[nm][3], summ[nm][2], summ[nm][4], summ[nm][5]),
        xlabel="position along edge, rel. to center [mm]", ylabel="w [mm]"); o.grid(alpha=.3)
    o = ax[2, a_]; o.plot(r[:, 0], r[:, 3], "o-", ms=3, lw=.8, c="C%d" % a_); o.axvspan(-STRIP, STRIP, alpha=.1)
    sl = np.polyfit(r[used, 0], r[used, 3], 1)[0]; o.set(title="edge position u0, slope %.4f -> %.2f deg" % (sl, np.degrees(np.arctan(sl))), xlabel="position along edge [mm]", ylabel="u0 [mm]"); o.grid(alpha=.3)
wx = np.mean([summ["x1 (left)"][0], summ["x2 (right)"][0]]); wy = np.mean([summ["y1 (low-y)"][0], summ["y2 (high-y)"][0]])
fig.suptitle("1.00 THz, per-line erf fits (window +-%.1f mm): w_x = %.2f mm (x1 %.2f, x2 %.2f) | w_y = %.2f mm (y1 %.2f, y2 %.2f)" % (HW, wx, summ["x1 (left)"][0], summ["x2 (right)"][0], wy, summ["y1 (low-y)"][0], summ["y2 (high-y)"][0]), fontsize=14)
plt.tight_layout(rect=(0, 0, 1, .96)); plt.savefig(os.path.join(A, "figs", "n12_%s_edgefit_lines_1THz.png" % TS), dpi=100)
print("w_x %.3f  w_y %.3f" % (wx, wy))
