"""2D 전방 모델 피팅: 회전한 직사각형 칩 마스크 ⊛ 이방 가우시안(exp(-2(dx²/wx²+dy²/wy²))).
F_ij(f) 맵(복소 스펙트럼 선형 혼합 사영, s4와 동일 정의)을 주파수별로 피팅. 0.1–4 THz."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal.windows import tukey
from common import *

D = np.asarray(load(), dtype=np.float64)
Gt = np.load(os.path.join(CACHE, "maps_tmpl.npz")); ix, iy = int(Gt["ix"]), int(Gt["iy"])
E = np.fft.rfft(D*tukey(NT, 0.2), axis=2); fr = np.fft.rfftfreq(NT, DT); del D
R1 = E[iy-5:iy+6, ix-5:ix+6].mean((0, 1)); n = 20
corn = np.concatenate([E[:n, :n].reshape(-1, len(fr)), E[:n, -n:].reshape(-1, len(fr)), E[-n:, :n].reshape(-1, len(fr)), E[-n:, -n:].reshape(-1, len(fr))])
R2 = corn.mean(0); blk = np.concatenate([E[:n, :n], E[-n:, -n:]]); dd = blk[:, 1:]-blk[:, :-1]
sig = np.sqrt((abs(dd)**2).mean((0, 1))/4); d = R1-R2; snr_d = abs(d)/sig
fsel = np.where((fr >= 0.099) & (fr <= 4.001))[0]
FM = np.empty((len(fsel), NY, NX), np.float32)
for k, j in enumerate(fsel): FM[k] = ((E[:, :, j]-R2[j])*np.conj(d[j])).real/abs(d[j])**2
np.save(os.path.join(CACHE, "F_0p1_4THz.npy"), FM)

# ---- 전방 모델 ----
H = 0.025; SUB = 4                     # 미세 격자 0.025 mm, 픽셀 = 4칸마다
xf = X0 + H*np.arange((NX-1)*SUB+1); yf = Y0 + H*np.arange((NY-1)*SUB+1)
XF, YF = np.meshgrid(xf, yf)
def mask(xc, yc, Lx, Ly, th):
    u = (XF-xc)*np.cos(th) + (YF-yc)*np.sin(th); v = -(XF-xc)*np.sin(th) + (YF-yc)*np.cos(th)
    return np.clip((Lx/2-abs(u))/H+.5, 0, 1)*np.clip((Ly/2-abs(v))/H+.5, 0, 1)
def blur(M, wx, wy):
    M = gaussian_filter1d(M, wx/2/H, axis=1, mode="constant", truncate=5)
    return gaussian_filter1d(M, wy/2/H, axis=0, mode="constant", truncate=5)
def pix(Mf): return Mf[::SUB, ::SUB]
XC0, YC0 = x[ix], y[iy]
PXX, PYY = np.meshgrid(x, y)
def roi(R, hole):
    m = (abs(PXX-XC0) < R) & (abs(PYY-YC0) < R)
    if hole: m &= ~((abs(PXX-XC0) < 1.5) & (abs(PYY-YC0) < 1.5))
    return m
def solve_AB(Mp, data, m):
    Bm = np.stack([Mp[m], np.ones(m.sum())], 1); ab, *_ = np.linalg.lstsq(Bm, data[m], rcond=None)
    return ab, data[m]-Bm@ab

def fit_geom(data, p0, m):
    def r(p):
        xc, yc, Lx, Ly, th, wx, wy = p; Mp = pix(blur(mask(xc, yc, Lx, Ly, th), abs(wx), abs(wy))); return solve_AB(Mp, data, m)[1]
    s = least_squares(r, p0, x_scale=[.1, .1, .1, .1, .01, .1, .1]); return s.x, np.sqrt(np.mean(s.fun**2))
def fit_w(data, geom, w0, m):
    xc, yc, Lx, Ly, th = geom; M0 = mask(xc, yc, Lx, Ly, th)
    def r(p): return solve_AB(pix(blur(M0, abs(p[0]), abs(p[1]))), data, m)[1]
    s = least_squares(r, w0, x_scale=[.1, .1]); ab, res = solve_AB(pix(blur(M0, abs(s.x[0]), abs(s.x[1]))), data, m)
    return np.abs(s.x), ab, np.sqrt(np.mean(res**2))

# ---- 1단계: 기하 (신뢰 빈에서 7변수 자유 피팅의 중앙값) ----
mbase = roi(7.0, False)
okb = [k for k, j in enumerate(fsel) if 0.4 <= fr[j] <= 1.3 and abs(d[j])/abs(R2[j]) > 1.0 and snr_d[j] > 50]
G = []
for k in okb:
    p, rms = fit_geom(FM[k].astype(float), [XC0, YC0, 6.0, 6.0, -0.04, 1.5, 1.2], mbase); G.append(p)
G = np.array(G); gmed = np.median(G, 0)
names = ["xc", "yc", "Lx", "Ly", "theta[rad]", "wx", "wy"]
print("기하 빈 수", len(okb))
for nm, med, sd in zip(names, gmed, G.std(0)): print("  %-10s %.4f ± %.4f" % (nm, med, sd))
print("  theta = %.2f deg" % np.degrees(gmed[4]))
geom = gmed[:5]; np.save(os.path.join(CACHE, "geom2d.npy"), gmed)

# ---- 2단계: 주파수별 wx, wy (+변형으로 불확실도) ----
variants = {"base": (7.0, False), "R5.5": (5.5, False), "R8.5": (8.5, False), "hole": (7.0, True)}
rows = []; w0 = [gmed[5], gmed[6]]
for k, j in enumerate(fsel):
    data = FM[k].astype(float); res = {}
    for nm, (R, hole) in variants.items():
        w, ab, rms = fit_w(data, geom, w0, roi(R, hole)); res[nm] = (w, ab, rms)
    wb = res["base"][0]; W = np.array([res[v][0] for v in variants])
    rows.append(dict(f=fr[j], wx=wb[0], wy=wb[1], wx_lo=W[:, 0].min(), wx_hi=W[:, 0].max(), wy_lo=W[:, 1].min(), wy_hi=W[:, 1].max(),
                     A=res["base"][1][0], B=res["base"][1][1], rms=res["base"][2], rel_d=abs(d[j])/abs(R2[j]), snr_d=snr_d[j], absR2=abs(R2[j]), noise=sig[j]))
    if res["base"][2] < 0.2 and 0.3 < wb[0] < 6 and 0.3 < wb[1] < 6: w0 = list(wb)    # 이웃 빈으로 초기값 이어받기
df = pd.DataFrame(rows); df.to_csv(os.path.join(ROOT, "analysis", "table_wf_2d.csv"), index=False)
print(df[df.f.round(2).isin([0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0])].to_string(index=False, float_format=lambda v: "%.3g" % v))

# ---- 진단 그림: 데이터 / 모델 / 잔차 (f≈0.5, 1.0, 1.5 THz) ----
fig, ax = plt.subplots(3, 3, figsize=(13, 11)); ext = [x[0], x[-1], y[-1], y[0]]
for r, ft in enumerate([0.5, 1.0, 1.5]):
    k = int(np.argmin(abs(fr[fsel]-ft))); row = df.iloc[k]; Mp = pix(blur(mask(*geom[:2], *geom[2:]), row.wx, row.wy)); mod = row.A*Mp+row.B
    for c, (im, tt, kw) in enumerate([(FM[k], "data F", dict(vmin=-.1, vmax=1.1)), (mod, "model", dict(vmin=-.1, vmax=1.1)), (FM[k]-mod, "residual", dict(vmin=-.2, vmax=.2, cmap="RdBu_r"))]):
        h = ax[r, c].imshow(im, extent=ext, **({"cmap": "viridis"} | kw)); ax[r, c].set_title("%s  %.2f THz  (wx=%.2f, wy=%.2f)" % (tt, fr[fsel][k], row.wx, row.wy), fontsize=9); plt.colorbar(h, ax=ax[r, c], fraction=.046)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "s6_2d_fit_diagnostic.png"), dpi=110)
