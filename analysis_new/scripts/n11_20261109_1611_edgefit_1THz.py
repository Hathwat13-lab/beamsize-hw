"""Demo: fit A*erf(sqrt2 (u-u0)/w)+B to each of the 4 edges of the central x-row / y-column of 2P-1 at 1 THz."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
R = r"C:\Users\cojyi\Desktop\beamsize hw"; A = os.path.join(R, "analysis_new")
df = pd.read_csv(os.path.join(A, "csv", "chip1_P_ij_f_20261109_1559.csv"))
j = list(df.columns[4:]).index("1.00"); P = df.iloc[:, 4+j].values.reshape(151, 151)
x = 63.04 + 0.1*np.arange(151); y = 39.63 + 0.1*np.arange(151)
prof = {"x": (x, 2*P[75]-1), "y": (y, 2*P[:, 80]-1)}
edges = [("x1 (left)", "x", +1), ("x2 (right)", "x", -1), ("y1 (low-y)", "y", +1), ("y2 (high-y)", "y", -1)]
def model(u, a, b, u0, w, sg): return sg*a*erf(np.sqrt(2)*(u-u0)/w) + b
def crossing(u, v, sg):                                   # zero crossing of the profile in the chip-edge direction
    s = np.where(np.diff(np.sign(v)) != 0)[0]; s = [i for i in s if (v[i+1]-v[i])*sg > 0]; i = s[0] if sg > 0 else s[-1]
    return u[i] - v[i]*(u[i+1]-u[i])/(v[i+1]-v[i])
def fit(u, v, sg, hw, u0):
    m = np.abs(u-u0) <= hw
    p, cov = curve_fit(lambda uu, a, b, c, w: model(uu, a, b, c, w, sg), u[m], v[m], p0=[1, 0, u0, 1.5])
    res = v[m] - model(u[m], *p, sg); dof = m.sum()-4; s2 = (res**2).sum()/dof
    return p, np.sqrt(np.diag(cov)), res, m, np.sqrt(s2)
DEF = 2.0; VAR = [1.5, 2.0, 2.5, 3.0]; out = {}
fig = plt.figure(figsize=(17, 12)); gs = fig.add_gridspec(3, 4, height_ratios=[1.2, 1.2, .7])
for a, (nm, ax_, sg) in enumerate(edges):
    u, v = prof[ax_]; u0 = crossing(u, v, sg); p, e, res, m, rms = fit(u, v, sg, DEF, u0)
    ws = [fit(u, v, sg, hw, u0)[0][3] for hw in VAR]; out[nm] = (p, e, rms, ws, u0)
    print("%-12s u0=%.3f  A=%.3f B=%.3f  w=%.3f +- %.3f mm  rms %.4f | w(window +-1.5/2/2.5/3 mm) = %s" % (nm, p[2], p[0], p[1], p[3], e[3], rms, np.round(ws, 3)))
for k, ax_ in enumerate(["x", "y"]):
    a = fig.add_subplot(gs[0, 2*k:2*k+2]); u, v = prof[ax_]; a.plot(u, v, "o", ms=2.5, c="gray", label="data 2P-1")
    for nm, ax2, sg in edges:
        if ax2 != ax_: continue
        p, e, rms, ws, u0 = out[nm]; uu = np.linspace(u[0], u[-1], 800); mm = np.abs(uu-u0) <= DEF
        a.plot(uu[mm], model(uu[mm], *p, sg), lw=2.2, label="%s: w=%.2f mm" % (nm, p[3])); a.axvspan(u0-DEF, u0+DEF, alpha=.08)
    a.set(title="central %s with fit windows (+-%.1f mm around each edge)" % ("row (vs x)" if ax_ == "x" else "column (vs y)", DEF), xlabel=ax_+" [mm]", ylabel="2P-1", ylim=(-1.4, 1.4)); a.grid(alpha=.3); a.legend(fontsize=8)
for a_, (nm, ax_, sg) in enumerate(edges):
    u, v = prof[ax_]; p, e, rms, ws, u0 = out[nm]; z = fig.add_subplot(gs[1, a_]); r = fig.add_subplot(gs[2, a_])
    mm = np.abs(u-u0) <= 3.0; z.plot(u[mm], v[mm], "o", ms=3, c="gray"); uu = np.linspace(u0-3, u0+3, 400)
    z.plot(uu, model(uu, *p, sg), c="C%d" % a_, lw=2); z.axvline(u0, c="k", ls=":", lw=.6)
    z.set(title="%s: w = %.3f +- %.3f mm\n(window +-1.5..3: %.2f - %.2f)" % (nm, p[3], e[3], min(ws), max(ws)), ylim=(-1.3, 1.3)); z.grid(alpha=.3)
    z.text(.03, .05, "A=%.3f B=%.3f\nu0=%.3f mm\nrms=%.4f" % (p[0], p[1], p[2], rms), transform=z.transAxes, fontsize=8, va="bottom")
    m2 = np.abs(u-u0) <= DEF; r.plot(u[m2], v[m2]-model(u[m2], *p, sg), ".-", c="C%d" % a_); r.axhline(0, c="k", lw=.5); r.set(xlabel=ax_+" [mm]", ylabel="residual", ylim=(-.1, .1)); r.grid(alpha=.3)
wx = [out["x1 (left)"][0][3], out["x2 (right)"][0][3]]; wy = [out["y1 (low-y)"][0][3], out["y2 (high-y)"][0][3]]
fig.suptitle("1.00 THz  |  w_x = %.2f, %.2f mm (mean %.2f)  |  w_y = %.2f, %.2f mm (mean %.2f)" % (*wx, np.mean(wx), *wy, np.mean(wy)), fontsize=14)
plt.tight_layout(rect=(0, 0, 1, .97)); plt.savefig(os.path.join(A, "figs", "n11_%s_edgefit_1THz.png" % "20261109_1611"), dpi=100)
print("w_x mean %.3f  w_y mean %.3f" % (np.mean(wx), np.mean(wy)))
