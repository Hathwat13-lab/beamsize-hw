"""s6의 수정판: w를 [0.2, 12] mm로 제한(발산 방지), 격자 0.05 mm, 빈 단위 병렬, 진행률 출력.
기하(geom2d.npy)와 F 맵(F_0p1_4THz.npy)은 s6 결과를 재사용. 출력: table_wf_2d_fast.csv"""
import numpy as np, pandas as pd, time, sys
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from common import *
H = 0.05; SUB = 2; WLO, WHI = 0.2, 12.0
def init():
    global FM, geom, M0, PXX, PYY, XC0, YC0
    FM = np.load(os.path.join(CACHE, "F_0p1_4THz.npy"), mmap_mode="r"); g = np.load(os.path.join(CACHE, "geom2d.npy")); geom = g[:5]
    xf = X0+H*np.arange((NX-1)*SUB+1); yf = Y0+H*np.arange((NY-1)*SUB+1); XF, YF = np.meshgrid(xf, yf)
    xc, yc, Lx, Ly, th = geom
    u = (XF-xc)*np.cos(th)+(YF-yc)*np.sin(th); v = -(XF-xc)*np.sin(th)+(YF-yc)*np.cos(th)
    M0 = np.clip((Lx/2-abs(u))/H+.5, 0, 1)*np.clip((Ly/2-abs(v))/H+.5, 0, 1)
    PXX, PYY = np.meshgrid(x, y); XC0, YC0 = xc, yc
def roi(R, hole):
    m = (abs(PXX-XC0) < R) & (abs(PYY-YC0) < R)
    if hole: m &= ~((abs(PXX-XC0) < 1.5) & (abs(PYY-YC0) < 1.5))
    return m
def fit(data, m, w0):
    def r(p):
        M = gaussian_filter1d(M0, p[0]/2/H, axis=1, mode="constant", truncate=4); M = gaussian_filter1d(M, p[1]/2/H, axis=0, mode="constant", truncate=4)[::SUB, ::SUB]
        B = np.stack([M[m], np.ones(m.sum())], 1); ab, *_ = np.linalg.lstsq(B, data[m], rcond=None); r.ab = ab; return data[m]-B@ab
    s = least_squares(r, np.clip(w0, WLO*1.5, WHI*.8), bounds=([WLO, WLO], [WHI, WHI]), x_scale=[.1, .1], max_nfev=40)
    res = r(s.x); return s.x, r.ab, np.sqrt(np.mean(res**2)), bool(np.any(np.isclose(s.x, [WLO, WLO])|np.isclose(s.x, [WHI, WHI])))
VAR = {"base": (7.0, False), "R5.5": (5.5, False), "hole": (7.0, True)}
def work(k):
    data = np.asarray(FM[k], dtype=float); out = {}
    for nm, (R, h) in VAR.items(): out[nm] = fit(data, roi(R, h), [1.5, 1.2])
    W = np.array([out[v][0] for v in VAR]); wb, ab, rms, edge = out["base"]
    return k, dict(wx=wb[0], wy=wb[1], wx_lo=W[:, 0].min(), wx_hi=W[:, 0].max(), wy_lo=W[:, 1].min(), wy_hi=W[:, 1].max(), A=ab[0], B=ab[1], rms=rms, at_bound=edge)
if __name__ == "__main__":
    n = np.load(os.path.join(CACHE, "F_0p1_4THz.npy"), mmap_mode="r").shape[0]; fr = 0.1+0.02*np.arange(n)
    # 스펙트럼 지표(신뢰도 표시용): s4/s6과 같은 정의로 재계산
    from scipy.signal.windows import tukey
    Gt = np.load(os.path.join(CACHE, "maps_tmpl.npz")); ix, iy = int(Gt["ix"]), int(Gt["iy"]); D = np.asarray(load(), dtype=np.float64)
    E = np.fft.rfft(D*tukey(NT, 0.2), axis=2); ff = np.fft.rfftfreq(NT, DT); j = np.round((fr-ff[0])/ (ff[1]-ff[0])).astype(int)
    R1 = E[iy-5:iy+6, ix-5:ix+6].mean((0, 1))[j]; nn = 20
    R2 = np.concatenate([E[:nn, :nn].reshape(-1, E.shape[2]), E[:nn, -nn:].reshape(-1, E.shape[2]), E[-nn:, :nn].reshape(-1, E.shape[2]), E[-nn:, -nn:].reshape(-1, E.shape[2])]).mean(0)[j]
    blk = np.concatenate([E[:nn, :nn], E[-nn:, -nn:]]); dd = blk[:, 1:]-blk[:, :-1]; sig = np.sqrt((abs(dd)**2).mean((0, 1))/4)[j]; del E, D
    d = R1-R2; rel = abs(d)/abs(R2); snr = abs(d)/sig
    t0 = time.time(); rows = {}
    with Pool(max(1, (os.cpu_count() or 2)-1), initializer=init) as P:
        for i, (k, r) in enumerate(P.imap_unordered(work, range(n))):
            rows[k] = r
            if (i+1) % 10 == 0:
                print("%d/%d bins, %.0f s" % (i+1, n, time.time()-t0), flush=True)
                pd.DataFrame([dict(f=fr[q], rel_d=rel[q], snr_d=snr[q], absR2=abs(R2[q]), noise=sig[q], **rows[q]) for q in sorted(rows)]).to_csv(os.path.join(ROOT, "analysis", "table_wf_2d_fast.csv"), index=False)
    pd.DataFrame([dict(f=fr[q], rel_d=rel[q], snr_d=snr[q], absR2=abs(R2[q]), noise=sig[q], **rows[q]) for q in sorted(rows)]).to_csv(os.path.join(ROOT, "analysis", "table_wf_2d_fast.csv"), index=False)
    print("DONE %.0f s" % (time.time()-t0), flush=True)
