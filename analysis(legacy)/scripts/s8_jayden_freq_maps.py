"""선배 코드(imaging_analysis_JaydenRho.py, 수정 없이 import)로 chip1 처리 → 주파수별 T 맵 갤러리."""
import importlib.util, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *
spec = importlib.util.spec_from_file_location("jr", os.path.join(ROOT, "imaging_analysis_JaydenRho.py")); jr = importlib.util.module_from_spec(spec); spec.loader.exec_module(jr)
sample, gold = os.path.join(ROOT, "chip1.tim.csv"), os.path.join(ROOT, "ref.pulse.csv")
data = jr.ImagingData.load(sample, gold)
fft = data.fft("rectangular")                                  # 도구 기본값: 직사각 윈도우, 전체 창
res = jr.extract(fft, 0.58, 1.5, 2.0, 5, "raster", "rectangular")   # 기본 설정 (floor 5 %, raster)
out = os.path.join(ROOT, "analysis", "jayden_out"); jr.export(out, data, res, sample, gold)   # 도구 자체 내보내기(imaging_maps.png 등)
print("valid band:", res.freq[res.valid][0], "-", res.freq[res.valid][-1], "THz;", res.valid.sum(), "bins")
T = res.spectra; fq = res.freq
want = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]
fl = [f for f in want if res.valid[int(abs(fq-f).argmin())]]
print("plotted:", fl, " skipped(below gold floor):", [f for f in want if f not in fl])
ext = [x[0]-.05, x[-1]+.05, y[-1]+.05, y[0]-.05]
nc = 4; nr = int(np.ceil(len(fl)/nc)); fig, ax = plt.subplots(nr, nc, figsize=(4.3*nc, 3.8*nr)); ax = np.atleast_1d(ax).ravel()
for a, f in zip(ax, fl):
    k = int(abs(fq-f).argmin()); v = T[..., k]; lo, hi = np.nanpercentile(v, [1, 99.5])
    im = a.imshow(v, extent=ext, cmap="viridis", vmin=lo, vmax=hi); a.set_title("%.2f THz  (T %.2g–%.2g)" % (fq[k], np.nanmin(v), np.nanmax(v)), fontsize=9)
    a.set_xlabel("x [mm]", fontsize=8); a.set_ylabel("y [mm]", fontsize=8); a.tick_params(labelsize=7); plt.colorbar(im, ax=a, fraction=.046)
for a in ax[len(fl):]: a.set_visible(False)
fig.suptitle("chip1: T = |FFT(sample)/FFT(Gold)|²  (senior code, rectangular window, per-panel color range)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "s8_jayden_T_by_freq.png"), dpi=110)
# 칩 중앙 / 개방 / 에지 대표 픽셀의 T 스펙트럼
fig, a = plt.subplots(figsize=(8, 4.5)); ys, xs = {"chip center": (75, 80), "open corner": (10, 10), "left edge": (75, 52)}, None
for nm, (r, c) in ys.items(): a.plot(fq, T[r, c], label="%s (row %d, col %d)" % (nm, r, c), lw=1)
a.set_xlim(0, 3); a.set_yscale("log"); a.set_xlabel("f [THz]"); a.set_ylabel("T"); a.legend(); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "s8_jayden_T_spectra.png"), dpi=120)
