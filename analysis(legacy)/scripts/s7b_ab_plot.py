import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
d = pd.read_csv(os.path.join(ROOT, "analysis", "table_wf_ab.csv"))
ok = (d.sinphi > 0.5) & ~d.bound_a & ~d.bound_b & (d.rms_a < .15) & (d.rms_b < .15)
d["ok"] = ok; print("ok bins:", ok.sum(), "of", len(d))
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
for c, (cx, cy, ttl) in enumerate([("wax", "way", "chip path (a)"), ("wbx", "wby", "open path (b)")]):
    for r, (cc, nm) in enumerate([(cx, "w_x"), (cy, "w_y")]):
        a = ax[r, c]; a.plot(d.f[~ok], d[cc][~ok], ".", color="0.75", ms=4); a.plot(d.f[ok], d[cc][ok], "o", ms=3, color="C0" if r == 0 else "C1")
        a.set(ylim=(0, 6), xlabel="f [THz]", ylabel=nm+" [mm]", title="%s  %s" % (nm, ttl)); a.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "s7b_ab_wf.png"), dpi=120)
g = d[ok]; lam = 0.29979/g.f
print(g[g.f.round(2).isin([0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.4, 3.0, 3.5])][["f", "sinphi", "wax", "wbx", "way", "wby"]].to_string(index=False, float_format=lambda v: "%.2f" % v))
for c in ["wax", "wbx", "way", "wby"]:
    s = g[(g.f >= .3) & (g.f <= 3.0)]; p, c0 = np.polyfit(np.log(s.f), np.log(s[c]), 1); res = np.log(s[c])-(p*np.log(s.f)+c0)
    print(c, "f^%.2f  w(1THz)=%.2f  log-resid scatter %.3f  w/lam=%.2f+-%.2f" % (p, np.exp(c0), res.std(), (s[c]/(0.29979/s.f)).mean(), (s[c]/(0.29979/s.f)).std()))
