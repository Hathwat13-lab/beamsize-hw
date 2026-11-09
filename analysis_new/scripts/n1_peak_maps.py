"""Raw chip1.tim.csv -> peak-amplitude maps under two scan-order hypotheses (no fitting)."""
import os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = r"C:\Users\cojyi\Desktop\beamsize hw"; A = os.path.join(R, "analysis_new")
raw = os.path.join(R, "chip1.tim.csv"); cache = os.path.join(A, "cache", "raw.npy")
with open(raw) as f:
    h = [next(f) for _ in range(3)]
print("header:", [s.strip() for s in h])
NY, NX, dy, dx, _, NT, dt = [float(v) for v in h[1].split(",")]; NY, NX, NT = int(NY), int(NX), int(NT)
if not os.path.exists(cache):
    d = np.loadtxt(raw, delimiter=",", skiprows=3, dtype=np.float32)
    print("data rows x cols:", d.shape); np.save(cache, d)
d = np.load(cache); assert d.shape == (NY*NX, NT), d.shape
cube = d.reshape(NY, NX, NT)                 # line k = row k, as stored
peak = cube.max(2)                           # peak amplitude (positive max)
x0, y0 = 63.04, 39.63
ext = [x0-dx/2, x0+(NX-1)*dx+dx/2, y0-dy/2, y0+(NY-1)*dy+dy/2]
raster = peak.copy()
zig = peak.copy(); zig[1::2] = zig[1::2, ::-1]
rough = lambda m: np.mean(np.abs(np.diff(m, axis=0))[40:110, 40:110])
fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
vmin, vmax = np.percentile(peak, [1, 99])
for a, m, t in [(ax[0], raster, "H1 raster: all rows left->right"), (ax[1], zig, "H2 zigzag: odd rows reversed")]:
    im = a.imshow(m, origin="lower", extent=ext, vmin=vmin, vmax=vmax, cmap="viridis", interpolation="nearest")
    a.set_title(f"{t}\nmean |row diff| = {rough(m):.1f} nA"); a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]"); plt.colorbar(im, ax=a)
dm = zig-raster
im = ax[2].imshow(dm, origin="lower", extent=ext, cmap="RdBu_r", vmin=-200, vmax=200, interpolation="nearest")
ax[2].set_title("H2 - H1"); ax[2].set_xlabel("x [mm]"); plt.colorbar(im, ax=ax[2])
plt.tight_layout(); plt.savefig(os.path.join(A, "figs", "n1_peak_raster_vs_zigzag.png"), dpi=110)
# zoom on chip (x 66-76, y 42-52)
fig, ax = plt.subplots(1, 2, figsize=(12, 5.6))
for a, m, t in [(ax[0], raster, "H1 raster"), (ax[1], zig, "H2 zigzag")]:
    im = a.imshow(m, origin="lower", extent=ext, vmin=vmin, vmax=vmax, cmap="viridis", interpolation="nearest")
    a.set_xlim(66, 76); a.set_ylim(42, 52); a.set_title(t + " (zoom)"); a.set_xlabel("x [mm]"); a.set_ylabel("y [mm]")
plt.tight_layout(); plt.savefig(os.path.join(A, "figs", "n1_peak_zoom.png"), dpi=110)
print("roughness raster %.2f zigzag %.2f" % (rough(raster), rough(zig)))
