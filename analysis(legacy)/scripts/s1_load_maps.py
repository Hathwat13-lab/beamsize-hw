import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
D = load()
D = np.asarray(D)
print(D.shape, D.dtype)
pp = D.max(2) - D.min(2)
tmax = t[D.argmax(2)] - T0
# 게이팅 a,b (early [8.4,11.4], late [12.9,15.9] ps)
def gate(lo, hi):
    m = (t-T0 >= lo) & (t-T0 <= hi)
    s = D[:, :, m]; return s.max(2) - s.min(2)
a = gate(8.4, 11.4); b = gate(12.9, 15.9)
np.savez(os.path.join(CACHE, "maps_gate.npz"), a=a, b=b, pp=pp, tmax=tmax)
# 방향 검증: 인접 행/열 상관 (X-fast 확인용)
def corr(u, v): return np.corrcoef(u.ravel(), v.ravel())[0, 1]
print("row-adjacent corr", corr(pp[:-1], pp[1:]), " col-adjacent", corr(pp[:, :-1], pp[:, 1:]))
# 칩 대칭성: a 맵에서 칩 영역 (a 큰 영역) 외형
mask = a > 0.5*np.nanmax(a)
yy, xx = np.nonzero(mask)
print("chip bbox rows", yy.min(), yy.max(), "cols", xx.min(), xx.max())
print("chip centroid x,y [mm]", x[xx].mean(), y[yy].mean())
# 중앙 resonator: 칩 내부에서 a가 낮은/다른 영역
fig, ax = plt.subplots(1, 4, figsize=(19, 4.5))
ext = [x[0]-.05, x[-1]+.05, y[-1]+.05, y[0]-.05]
for A, n in zip(ax, [(pp, "pp [nA]"), (a, "a: early gate pp"), (b, "b: late gate pp"), (tmax, "t_max - 1053 [ps]")]):
    im = A.imshow(n[0], extent=ext, cmap="viridis"); A.set_title(n[1]); A.set_xlabel("x [mm]"); A.set_ylabel("row-index y [mm]")
    plt.colorbar(im, ax=A)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "s1_maps.png"), dpi=130)
