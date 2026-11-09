"""템플릿 선형 최소제곱: E_ij(t) = a T1(t) + b T2(t) + c  (픽셀별). T1=칩 중앙, T2=개방 코너."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
D = np.asarray(load(), dtype=np.float64)
G = np.load(os.path.join(CACHE, "maps_gate.npz"))
ag = G["a"]
# 칩 중심: 게이트 a 맵의 무게중심
w = np.clip(ag - 0.5*ag.max(), 0, None)
yy, xx = np.indices(ag.shape)
cy = (w*yy).sum()/w.sum(); cx = (w*xx).sum()/w.sum()
iy, ix = int(round(cy)), int(round(cx))
print("chip centre px", cx, cy, "-> x,y mm", x[ix], y[iy])
T1 = D[iy-5:iy+6, ix-5:ix+6].mean((0, 1))
n = 20
corners = np.concatenate([D[:n, :n].reshape(-1, NT), D[:n, -n:].reshape(-1, NT),
                          D[-n:, :n].reshape(-1, NT), D[-n:, -n:].reshape(-1, NT)])
T2 = corners.mean(0)
print("T1 t_max-T0", t[T1.argmax()]-T0, "T2 t_max-T0", t[T2.argmax()]-T0, "shift [ps]", t[T2.argmax()]-t[T1.argmax()])
# 개방 코너 간 균일성 (4코너 평균 파형의 진폭 변화)
print("corner pp:", [float(np.ptp(D[sl].mean((0, 1)))) for sl in [(slice(0, n), slice(0, n)), (slice(0, n), slice(-n, None)), (slice(-n, None), slice(0, n)), (slice(-n, None), slice(-n, None))]])
Bm = np.stack([T1, T2, np.ones(NT)], 1)           # (1000,3)
coef, *_ = np.linalg.lstsq(Bm, D.reshape(-1, NT).T, rcond=None)   # (3, N)
coef = coef.reshape(3, NY, NX)
res = D.reshape(-1, NT).T - Bm @ coef.reshape(3, -1)
rms = np.sqrt((res**2).mean(0)).reshape(NY, NX)
print("template a range", coef[0].min(), coef[0].max(), " b range", coef[1].min(), coef[1].max())
print("resid rms: median %.3f nA, max %.3f nA, (pp~1180)" % (np.median(rms), rms.max()))
cond = np.linalg.cond(Bm); print("cond(basis)", cond)
np.savez(os.path.join(CACHE, "maps_tmpl.npz"), a=coef[0], b=coef[1], c=coef[2], rms=rms, T1=T1, T2=T2, ix=ix, iy=iy)
ext = [x[0]-.05, x[-1]+.05, y[-1]+.05, y[0]-.05]
fig, ax = plt.subplots(1, 4, figsize=(19, 4.5))
for A, (m, nm) in zip(ax, [(coef[0], "a (template)"), (coef[1], "b (template)"), (rms, "residual rms [nA]"), (coef[0]+coef[1], "a+b")]):
    im = A.imshow(m, extent=ext, cmap="viridis"); A.set_title(nm); A.set_xlabel("x [mm]"); A.set_ylabel("y [mm]"); plt.colorbar(im, ax=A)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "s2_template_maps.png"), dpi=130)
fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(t-T0, T1, label="T1 chip centre"); ax.plot(t-T0, T2, label="T2 open"); ax.set_xlabel("t - 1053 [ps]"); ax.set_ylabel("nA"); ax.legend(); ax.set_xlim(0, 50)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "s2_templates.png"), dpi=130)
