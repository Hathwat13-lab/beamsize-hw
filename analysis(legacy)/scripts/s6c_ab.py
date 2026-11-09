"""복소 신호를 두 실수 성분으로 분리: E = a R1 + b R2 (a=칩 경로, b=개방 경로). a,b 맵을 각각 2D 모델로 피팅 -> w_chip(f), w_open(f)."""
import numpy as np, pandas as pd, time
from scipy.signal.windows import tukey
import s6b_2d_fast as m
from common import *
D = np.asarray(load(), dtype=np.float64); Gt = np.load(os.path.join(CACHE, "maps_tmpl.npz")); ix, iy = int(Gt["ix"]), int(Gt["iy"])
E = np.fft.rfft(D*tukey(NT, 0.2), axis=2); ff = np.fft.rfftfreq(NT, DT); del D
fr = 0.1+0.02*np.arange(196); j = np.round((fr-ff[0])/(ff[1]-ff[0])).astype(int)
R1 = E[iy-5:iy+6, ix-5:ix+6].mean((0, 1)); nn = 20
R2 = np.concatenate([E[:nn, :nn].reshape(-1, E.shape[2]), E[:nn, -nn:].reshape(-1, E.shape[2]), E[-nn:, :nn].reshape(-1, E.shape[2]), E[-nn:, -nn:].reshape(-1, E.shape[2])]).mean(0)
blk = np.concatenate([E[:nn, :nn], E[-nn:, -nn:]]); dd = blk[:, 1:]-blk[:, :-1]; sig = np.sqrt((abs(dd)**2).mean((0, 1))/4)
m.init(); rows = []; t0 = time.time()
for q, jj in enumerate(j):
    r1, r2, Ez = R1[jj], R2[jj], E[:, :, jj]
    det = r1.real*r2.imag-r2.real*r1.imag
    a = (Ez.real*r2.imag-r2.real*Ez.imag)/det; b = (r1.real*Ez.imag-r1.imag*Ez.real)/det
    sinphi = abs(det)/(abs(r1)*abs(r2)); amp = np.sqrt((r2.imag**2+r2.real**2+r1.real**2+r1.imag**2))/abs(det)*sig[jj]   # a,b 잡음 크기 근사
    out = {}
    for nm, mp in (("a", a), ("b", b)):
        res = {v: m.fit(mp, m.roi(*m.VAR[v]), [1.5, 1.2]) for v in m.VAR}
        W = np.array([res[v][0] for v in m.VAR]); wb, ab, rms, edge = res["base"]
        out.update({"w%sx" % nm: wb[0], "w%sy" % nm: wb[1], "e%sx" % nm: max(W[:, 0].max()-wb[0], wb[0]-W[:, 0].min()), "e%sy" % nm: max(W[:, 1].max()-wb[1], wb[1]-W[:, 1].min()), "A_%s" % nm: ab[0], "rms_%s" % nm: rms, "bound_%s" % nm: edge})
    rows.append(dict(f=fr[q], sinphi=sinphi, noise_ab=amp, **out))
    if q % 20 == 0: print(q, "%.0f s" % (time.time()-t0), flush=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "analysis", "table_wf_ab.csv"), index=False); print("DONE %.0f s" % (time.time()-t0))
