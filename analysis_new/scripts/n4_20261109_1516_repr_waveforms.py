"""Representative TDS waveforms from chip1.tim.csv (raster order, rows = y, cols = x). No windowing/FFT."""
import os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = r"C:\Users\cojyi\Desktop\beamsize hw"; A = os.path.join(R, "analysis_new")
cube = np.load(os.path.join(A, "cache", "raw.npy")).reshape(151, 151, 1000)
t = 1053 + 0.05*np.arange(1000); x = 63.04 + 0.1*np.arange(151); y = 39.63 + 0.1*np.arange(151)
pp = cube.max(2) - cube.min(2)
pts = {"chip center (71.0,47.1)": (75, 80), "open corner (64,41)": (15, 10),
       "left edge (67.8,47.1)": (75, 48), "right edge (73.9,47.1)": (75, 109),
       "top edge (71.0,44.2)": (46, 80), "bottom edge (71.0,50.0)": (104, 80)}
def avg(iy, ix, h=1): return cube[iy-h:iy+h+1, ix-h:ix+h+1].mean((0, 1))   # 3x3 mean
fig = plt.figure(figsize=(16, 13)); gs = fig.add_gridspec(3, 2)
a0 = fig.add_subplot(gs[0, 0]); a1 = fig.add_subplot(gs[0, 1])
for k, (iy, ix) in pts.items():
    a0.plot(t, avg(iy, ix), lw=.9, label=k); a1.plot(t, avg(iy, ix), lw=.9)
a0.set(title="3x3-mean waveforms, full 50 ps", xlabel="t [ps]", ylabel="nA"); a0.legend(fontsize=8)
a1.set(title="zoom 1058-1075 ps", xlabel="t [ps]", xlim=(1058, 1075)); a1.grid(alpha=.3)
a2 = fig.add_subplot(gs[1, 0]); a3 = fig.add_subplot(gs[1, 1])
for k in ["chip center (71.0,47.1)", "open corner (64,41)"]:
    iy, ix = pts[k]; a2.semilogy(t, np.abs(avg(iy, ix)) + 1e-3, lw=.8, label=k)
a2.set(title="|E| log scale (tail / noise floor)", xlabel="t [ps]", ylim=(1e-2, 1e3)); a2.legend(fontsize=8); a2.grid(alpha=.3)
n = np.std(cube[:, :, :150], axis=2)   # pre-pulse noise (1053-1060.5 ps)
a3.hist(n.ravel(), bins=60); a3.set(title=f"pre-pulse rms per pixel (1053-1060.5 ps), median {np.median(n):.3f} nA", xlabel="nA")
# waterfall along x through chip center row, and along y through center column
for j, (lab, img, ext, xl) in enumerate([("x-line at y=47.1", cube[75], [x[0], x[-1]], "x [mm]"), ("y-line at x=71.0", cube[:, 80], [y[0], y[-1]], "y [mm]")]):
    a = fig.add_subplot(gs[2, j])
    im = a.imshow(img.T, origin="lower", aspect="auto", extent=[ext[0], ext[1], t[0], t[-1]], cmap="RdBu_r", vmin=-600, vmax=600)
    a.set(title="waterfall " + lab, xlabel=xl, ylabel="t [ps]", ylim=(1058, 1075)); plt.colorbar(im, ax=a, label="nA")
plt.tight_layout(); plt.savefig(os.path.join(A, "figs", "n4_20261109_1516_repr_waveforms.png"), dpi=100)
for k, (iy, ix) in pts.items():
    w = avg(iy, ix); print("%-26s pp %.0f  t_max %.2f  t_min %.2f  late-peak(|E|>20 last) %.2f ps" % (k, w.max()-w.min(), t[w.argmax()], t[w.argmin()], t[np.where(np.abs(w) > 20)[0].max()]))
print("noise median %.3f nA" % np.median(n))
