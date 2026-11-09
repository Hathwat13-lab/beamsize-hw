"""FFT of representative chip1 waveforms with 3 window versions. No fitting."""
import os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
TS = "20261109_1458"
R = r"C:\Users\cojyi\Desktop\beamsize hw"; A = os.path.join(R, "analysis_new")
cube = np.load(os.path.join(A, "cache", "raw.npy")).reshape(151, 151, 1000).astype(float)
t = 1053 + 0.05*np.arange(1000); dt = 0.05; NFFT = 16384
R1 = cube[70:81, 75:86].mean((0, 1))                      # chip center 11x11 (iy 70-80, ix 75-85)
R2 = np.mean([cube[:20, :20], cube[:20, -20:], cube[-20:, :20], cube[-20:, -20:]], axis=(0, 1, 2))  # 4 open corners 20x20
def win(kind):
    w = np.zeros(1000)
    if kind == "none": w[:] = 1
    elif kind == "tukey": w[:] = tukey(1000, 0.2)
    elif kind == "crop": m = (t >= 1059) & (t <= 1074); w[m] = tukey(m.sum(), 0.2)
    return w
def spec(x, kind):
    X = np.fft.rfft(x*win(kind), NFFT)*dt
    return np.fft.rfftfreq(NFFT, dt), X                      # f [THz], X in nA*ps (time origin = 1053 ps for all versions)
labels = {"none": "no window (50 ps)", "tukey": "Tukey a=0.2 (50 ps)", "crop": "crop 1059-1074 ps + Tukey a=0.2"}
col = {"none": "C0", "tukey": "C1", "crop": "C3"}
out = {}
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
for k in labels:
    f, X1 = spec(R1, k); _, X2 = spec(R2, k); d = X1 - X2
    out[k+"_R1"] = X1; out[k+"_R2"] = X2
    ax[0, 0].plot(f, 20*np.log10(np.abs(X1)), c=col[k], lw=.9, label=labels[k])
    ax[0, 1].plot(f, 20*np.log10(np.abs(X2)), c=col[k], lw=.9, label=labels[k])
    ax[1, 0].plot(f, np.abs(d)/np.abs(X2), c=col[k], lw=.9, label=labels[k])
    ax[1, 1].plot(f, np.unwrap(np.angle(X1/X2)), c=col[k], lw=.9, label=labels[k])
np.savez(os.path.join(A, "cache", "n5_%s_fft3.npz" % TS), f=f, **out)
for a, ti, yl in zip(ax.ravel(), ["R1 chip center (11x11 mean): amplitude", "R2 open corners (4x 20x20 mean): amplitude", "|R1-R2| / |R2| (chip vs open contrast)", "phase of R1/R2 (unwrapped)"], ["dB (nA*ps)", "dB (nA*ps)", "ratio", "rad"]):
    a.set(title=ti, xlabel="f [THz]", ylabel=yl, xlim=(0, 4)); a.grid(alpha=.3); a.legend(fontsize=8)
ax[0, 0].set_ylim(-60, 60); ax[0, 1].set_ylim(-60, 60); ax[1, 0].set_ylim(0, 6)
plt.tight_layout(); plt.savefig(os.path.join(A, "figs", "n5_%s_fft3.png" % TS), dpi=100)
# numbers: peak freq, 0.5/1/2/3 THz values, and nodes of |d|
for k in labels:
    X1, X2 = out[k+"_R1"], out[k+"_R2"]; g = lambda fr: np.argmin(abs(f-fr))
    print(k, "peak f R1 %.3f R2 %.3f THz" % (f[np.abs(X1).argmax()], f[np.abs(X2).argmax()]),
          "| |R1|dB at .5/1/2/3 THz:", [round(20*np.log10(abs(X1[g(v)])), 1) for v in (.5, 1, 2, 3)])
