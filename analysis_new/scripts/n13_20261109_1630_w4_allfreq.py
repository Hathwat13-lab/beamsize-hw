"""Per-edge widths at all frequencies: per-line erf fits (strip +-1.5 mm along each edge), window +-2 mm (variants 1.5/2.5/3) -> CSV."""
import os, sys, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import curve_fit
from scipy.special import erf
TS = "20261109_1630"
R = r"C:\Users\cojyi\Desktop\beamsize hw"; A = os.path.join(R, "analysis_new")
x = 63.04 + 0.1*np.arange(151); y = 39.63 + 0.1*np.arange(151)
EDGES = [("xleft", "x", +1, 75), ("xright", "x", -1, 75), ("ylow", "y", +1, 80), ("yhigh", "y", -1, 80)]   # name, profile axis, sign, strip-centre line
HWS = (2.0, 1.5, 2.5, 3.0); STRIP = 15      # strip +-1.5 mm = +-15 lines
def model(u, a, b, u0, w, sg): return sg*a*erf(np.sqrt(2)*(u-u0)/w) + b
G = {}
def init():
    P = pd.read_csv(os.path.join(A, "csv", "chip1_P_ij_f_20261109_1559.csv"))
    G["f"] = np.array([float(c) for c in P.columns[4:]]); G["V"] = 2*P.iloc[:, 4:].values.reshape(151, 151, -1) - 1
    g = pd.read_csv(os.path.join(A, "csv", "n12_20261109_1623_edgefit_lines_1THz.csv"))   # edge geometry (u0 per line) from the 1 THz fits
    nm2 = {"xleft": "x1 (left)", "xright": "x2 (right)", "ylow": "y1 (low-y)", "yhigh": "y2 (high-y)"}
    G["u0"] = {k: g[g.edge == v].set_index("line_idx").u0_mm for k, v in nm2.items()}
def fit1(u, v, sg, u0c, hw, w0):
    m = np.abs(u-u0c) <= hw
    try:
        p, _ = curve_fit(lambda uu, a, b, c, w: model(uu, a, b, c, w, sg), u[m], v[m], p0=[1, 0, u0c, w0],
                         bounds=([0.2, -1.5, u0c-0.6, 0.2], [3, 1.5, u0c+0.6, 12]), maxfev=2000)
        rms = np.sqrt(np.mean((v[m]-model(u[m], *p, sg))**2)); atb = (p[3] < 0.21) or (p[3] > 11.9) or abs(p[2]-u0c) > 0.59
        return p[3], p[0], p[1], rms, atb
    except Exception: return None
def work(jf):
    f = G["f"][jf]; w0 = float(np.clip(1.4/max(f, 0.05), 0.3, 6.0)); out = {"f_THz": f}
    for nm, ax_, sg, c in EDGES:
        u = x if ax_ == "x" else y
        for hi, hw in enumerate(HWS):
            ws = []; As = []; Bs = []; rs = []
            for k in range(c-STRIP, c+STRIP+1):
                v = G["V"][k, :, jf] if ax_ == "x" else G["V"][:, k, jf]
                r = fit1(u, v, sg, G["u0"][nm][k], hw, w0)
                if r and not r[4] and r[3] < 0.2: ws.append(r[0]); As.append(r[1]); Bs.append(r[2]); rs.append(r[3])
            if hi == 0:
                out["w_"+nm] = np.mean(ws) if ws else np.nan; out["std_"+nm] = np.std(ws) if ws else np.nan; out["nok_"+nm] = len(ws)
                out["A_"+nm] = np.mean(As) if ws else np.nan; out["B_"+nm] = np.mean(Bs) if ws else np.nan; out["rms_"+nm] = np.mean(rs) if ws else np.nan
            else: out["w_%s_hw%.1f" % (nm, hw)] = np.mean(ws) if ws else np.nan
    return out
if __name__ == "__main__":
    init(); n = len(G["f"]); print("freqs", n, "cpu", os.cpu_count(), flush=True)
    with Pool(max(1, (os.cpu_count() or 2)-1), initializer=init) as pool: rows = pool.map(work, range(n), chunksize=4)
    t = pd.DataFrame(rows); tab = pd.read_csv(os.path.join(A, "csv", "chip1_P_ij_f_20261109_1559_R1R2_table.csv"))
    t.insert(1, "rel_d", tab["|R1-R2|/|R2|"].values)             # chip-vs-stage contrast |R1-R2|/|R2| (P is ill-conditioned where small)
    t["w_x_avg"] = t[["w_xleft", "w_xright"]].mean(axis=1); t["w_y_avg"] = t[["w_ylow", "w_yhigh"]].mean(axis=1)
    out = os.path.join(A, "csv", "chip1_w4_allfreq_%s.csv" % TS); t.to_csv(out, index=False, float_format="%.5g"); print(out, t.shape)
    for fv in (0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0):
        r = t.iloc[int(round(fv/0.02))]; print("f=%.2f rel_d=%.2f | w xleft %.3f xright %.3f ylow %.3f yhigh %.3f | nok %d %d %d %d" % (fv, r.rel_d, r.w_xleft, r.w_xright, r.w_ylow, r.w_yhigh, r.nok_xleft, r.nok_xright, r.nok_ylow, r.nok_yhigh))
