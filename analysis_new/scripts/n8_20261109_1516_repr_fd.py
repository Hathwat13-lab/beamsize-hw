"""Representative pixels in frequency domain, read back from the exported CSVs (validates the files)."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
CS = "20261109_1513"
R = r"C:\Users\cojyi\Desktop\beamsize hw"; A = os.path.join(R, "analysis_new")
amp = pd.read_csv(os.path.join(A, "csv", "chip1_fft_amplitude_nAps_%s.csv" % CS)); ph = pd.read_csv(os.path.join(A, "csv", "chip1_fft_phase_rad_%s.csv" % CS))
f = np.array([float(c) for c in amp.columns[4:]]); Z = amp.iloc[:, 4:].values*np.exp(1j*ph.iloc[:, 4:].values)   # (22801,501) complex
Z = Z.reshape(151, 151, -1)
pts = {"chip center (71.0,47.1)": (75, 80), "open corner (64,41)": (15, 10), "left edge (67.8,47.1)": (75, 48), "right edge (73.9,47.1)": (75, 109),
       "top edge (71.0,44.2)": (46, 80), "bottom edge (71.0,50.0)": (104, 80)}
S = {k: Z[iy-1:iy+2, ix-1:ix+2].mean((0, 1)) for k, (iy, ix) in pts.items()}   # 3x3 complex mean, same as TDS figure
ref = S["open corner (64,41)"]
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
for k, s in S.items():
    ax[0, 0].plot(f, 20*np.log10(abs(s)), lw=.9, label=k); ax[0, 1].plot(f, abs(s), lw=.9)
    ax[1, 0].plot(f, np.unwrap(np.angle(s)), lw=.9); ax[1, 1].plot(f, np.unwrap(np.angle(s/ref)), lw=.9)
ax[0, 0].set(title="amplitude [dB re 1 nA*ps], 3x3 complex mean", xlim=(0, 4), ylim=(-10, 60), ylabel="dB"); ax[0, 0].legend(fontsize=8)
ax[0, 1].set(title="amplitude (linear)", xlim=(0, 2.5), ylabel="nA*ps")
ax[1, 0].set(title="phase (unwrapped, origin 1053 ps)", xlim=(0, 4), ylabel="rad")
ax[1, 1].set(title="phase relative to open corner (unwrapped)", xlim=(0, 4), ylabel="rad")
for a in ax.ravel(): a.set_xlabel("f [THz]"); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(A, "figs", "n8_%s_repr_fd.png" % "20261109_1516"), dpi=100)
# consistency check vs earlier n6 (R1 11x11 mean spectrum, no window)
n6 = np.load(os.path.join(A, "cache", "n6_20261109_1510_fft3_nopad.npz")); R1 = Z[70:81, 75:86].mean((0, 1))
print("max |R1(csv) - R1(n6 none)| / |R1|: %.2e" % np.max(abs(R1 - n6["none_X1"])/abs(n6["none_X1"])[0:501].clip(1e-9)))
for k, s in S.items(): print("%-26s |X| at 0.5/1/2 THz: %s nA*ps" % (k, [round(abs(s[int(round(v/.02))]), 1) for v in (.5, 1, 2)]))
