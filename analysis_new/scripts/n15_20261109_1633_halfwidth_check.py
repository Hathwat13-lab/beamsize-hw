"""Check: does a symmetric-erf width mix the two beam halves? Split model: halves w_t (toward chip) and w_a (away from chip)."""
import os, numpy as np, pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf, erfc
R = r"C:\Users\cojyi\Desktop\beamsize hw"; A = os.path.join(R, "analysis_new")
S2 = np.sqrt(2)
def split(s, A_, B_, wt, wa):          # s: signed distance of beam centre from edge, positive = beam centre inside chip
    Pm = np.where(s <= 0, wt/(wt+wa)*erfc(S2*np.abs(s)/wt), 1 - wa/(wt+wa)*erfc(S2*np.abs(s)/wa)); return A_*(2*Pm-1) + B_
def sym(s, A_, B_, s0, w): return A_*erf(S2*(s-s0)/w) + B_
# (1) simulation: beam with halves (toward chip 1.6, away 1.2) -> symmetric fit with free edge position
s = np.arange(-2.5, 2.5001, 0.1); v = split(s, 1, 0, 1.6, 1.2)
p, _ = curve_fit(sym, s, v, p0=[1, 0, 0, 1.4]); print("SIM  halves (toward 1.60, away 1.20): symmetric-erf w = %.3f (mean of halves 1.40), fitted edge offset s0=%.3f mm" % (p[3], p[2]))
# (2) real data at 1 THz: per-line split fits, strip +-1.5 mm
P = pd.read_csv(os.path.join(A, "csv", "chip1_P_ij_f_20261109_1559.csv")); j = list(P.columns[4:]).index("1.00")
V = 2*P.iloc[:, 4+j].values.reshape(151, 151) - 1
g = pd.read_csv(os.path.join(A, "csv", "n12_20261109_1623_edgefit_lines_1THz.csv"))
x = 63.04 + 0.1*np.arange(151); y = 39.63 + 0.1*np.arange(151)
E = [("x1 left  (chip at +x)", "x1 (left)", "x", +1, 75, "w_t = beam +x half", "w_a = beam -x half"),
     ("x2 right (chip at -x)", "x2 (right)", "x", -1, 75, "w_t = beam -x half", "w_a = beam +x half"),
     ("y1 low   (chip at +y)", "y1 (low-y)", "y", +1, 80, "w_t = beam +y half", "w_a = beam -y half"),
     ("y2 high  (chip at -y)", "y2 (high-y)", "y", -1, 80, "w_t = beam -y half", "w_a = beam +y half")]
res = {}
for lab, key, ax_, sg, c, tl, al in E:
    u = x if ax_ == "x" else y; gg = g[g.edge == key].set_index("line_idx").u0_mm; W = []; Ws = []
    for k in range(c-15, c+16):
        v = V[k] if ax_ == "x" else V[:, k]; u0 = gg[k]; m = np.abs(u-u0) <= 2.5; ss = sg*(u[m]-u0)
        try:
            q, _ = curve_fit(split, ss, v[m], p0=[.97, 0, 1.4, 1.4], bounds=([.5, -.5, .3, .3], [1.5, .5, 6, 6])); W.append(q[2:])
            q2, _ = curve_fit(sym, ss, v[m], p0=[.97, 0, 0, 1.4]); Ws.append(q2[3])
        except Exception: pass
    W = np.array(W); res[key] = (W[:, 0].mean(), W[:, 1].mean())
    print("%-22s lines %2d | %s = %.3f +- %.3f | %s = %.3f +- %.3f | symmetric-erf w = %.3f" % (lab, len(W), tl, W[:, 0].mean(), W[:, 0].std(), al, W[:, 1].mean(), W[:, 1].std(), np.mean(Ws)))
print("--- cross-check: the same beam half seen from two different edges")
print("beam +x half: x1 outer(w_t) %.3f  vs  x2 inner(w_a) %.3f" % (res["x1 (left)"][0], res["x2 (right)"][1]))
print("beam -x half: x2 outer(w_t) %.3f  vs  x1 inner(w_a) %.3f" % (res["x2 (right)"][0], res["x1 (left)"][1]))
print("beam +y half: y1 outer(w_t) %.3f  vs  y2 inner(w_a) %.3f" % (res["y1 (low-y)"][0], res["y2 (high-y)"][1]))
print("beam -y half: y2 outer(w_t) %.3f  vs  y1 inner(w_a) %.3f" % (res["y2 (high-y)"][0], res["y1 (low-y)"][1]))
