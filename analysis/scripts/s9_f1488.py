"""1.488 THz (resonator 공명 대역) 근방 T 맵. 선배 코드 로더 + 정확한 1.488 THz DFT(0.02 THz 격자 밖이므로 직접 DFT)."""
import importlib.util, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
spec = importlib.util.spec_from_file_location("jr", os.path.join(ROOT, "imaging_analysis_JaydenRho.py")); jr = importlib.util.module_from_spec(spec); spec.loader.exec_module(jr)
data = jr.ImagingData.load(os.path.join(ROOT, "chip1.tim.csv"), os.path.join(ROOT, "ref.pulse.csv"))
W = data.waveforms - data.waveforms.mean(-1, keepdims=True); g = data.gold - data.gold.mean()   # 도구와 동일: 평균 제거, rectangular
tt = np.arange(NT)*DT
def Tmap(f):
    e = np.exp(-2j*np.pi*f*tt); return (abs(W @ e)/abs(g @ e))**2
f0 = 1.488; T0m = Tmap(f0)
fl = [1.40, 1.44, 1.47, f0, 1.50, 1.53, 1.56, 1.60]; Tm = {f: Tmap(f) for f in fl}
ext = [x[0]-.05, x[-1]+.05, y[-1]+.05, y[0]-.05]; zoom = dict(xlim=(66.5, 75.7), ylim=(51.8, 42.5))
fig, ax = plt.subplots(2, 4, figsize=(19, 8.6)); ax = ax.ravel()
for a, f in zip(ax, fl):
    v = Tm[f]; sub = v[(y > 43) & (y < 51.5)][:, (x > 67) & (x < 75.5)]; lo, hi = np.percentile(sub, [1, 99.5])
    im = a.imshow(v, extent=ext, cmap="viridis", vmin=lo, vmax=hi); a.set(xlim=zoom["xlim"], ylim=zoom["ylim"], xlabel="x [mm]", ylabel="y [mm]")
    a.set_title("%.3f THz%s" % (f, "  <- resonance band" if f == f0 else ""), fontsize=10, color="C3" if f == f0 else "k"); plt.colorbar(im, ax=a, fraction=.046)
    cx, cy = XC = 71.04, 47.13; a.plot([cx-1, cx+1, cx+1, cx-1, cx-1], [cy-1, cy-1, cy+1, cy+1, cy-1], "w--", lw=.8)   # 중앙 2x2 mm
fig.suptitle("chip1 T map near 1.488 THz (zoom on chip; white dashed = central 2×2 mm resonator)  per-panel color range", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "s9_T_1p488_neighborhood.png"), dpi=110)
# 확대 단일 맵 + 대표 스펙트럼
iy, ix = 75, 80; off = 18
pts = {"chip center (resonator)": [(iy, ix)], "chip inner ring (±1.8 mm, 4 pts)": [(iy+off, ix), (iy-off, ix), (iy, ix+off), (iy, ix-off)], "open corner": [(8, 8)]}
fq = np.arange(0.9, 2.1, 0.004); E = np.exp(-2j*np.pi*np.outer(fq, tt)); G = abs(E @ g)
fig, ax = plt.subplots(1, 2, figsize=(15, 5.8))
v = T0m; lo, hi = np.percentile(v[(y > 43) & (y < 51.5)][:, (x > 67) & (x < 75.5)], [1, 99.5])
im = ax[0].imshow(v, extent=ext, cmap="viridis", vmin=lo, vmax=hi); ax[0].set(xlim=zoom["xlim"], ylim=zoom["ylim"], xlabel="x [mm]", ylabel="y [mm]", title="T at 1.488 THz"); plt.colorbar(im, ax=ax[0], fraction=.046)
ax[0].plot([70.04, 72.04, 72.04, 70.04, 70.04], [46.13, 46.13, 48.13, 48.13, 46.13], "w--", lw=.8)
for nm, pl in pts.items():
    T = np.mean([(abs(E @ W[r, c])/G)**2 for r, c in pl], 0); ax[1].plot(fq, T, label=nm, lw=1.2)
ax[1].axvline(f0, color="C3", ls="--", lw=.8); ax[1].set(xlabel="f [THz]", ylabel="T = |sample/Gold|²", title="T spectra", yscale="log", ylim=(0.02, 2)); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "s9_T_1p488_zoom_spectra.png"), dpi=115)
c = T0m[iy-5:iy+6, ix-5:ix+6].mean(); r = np.mean([T0m[a, b] for a, b in pts["chip inner ring (±1.8 mm, 4 pts)"]]); o = T0m[:20, :20].mean()
print("T(1.488) center %.3f | inner ring %.3f | open %.3f" % (c, r, o))
for f in [1.40, 1.45, 1.488, 1.52, 1.58]:
    e = np.exp(-2j*np.pi*f*tt); gg = abs(g @ e)
    print(f, "center/ring ratio of T:", ((abs(W[iy-5:iy+6, ix-5:ix+6] @ e)/gg)**2).mean()/np.mean([(abs(W[a, b] @ e)/gg)**2 for a, b in pts["chip inner ring (±1.8 mm, 4 pts)"]]))
