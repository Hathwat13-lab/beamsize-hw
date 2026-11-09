"""FFT of representative chip1 waveforms, 3 window versions, NO zero padding (native grids). No fitting."""
import os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
TS = "20261109_1510"
R = r"C:\Users\cojyi\Desktop\beamsize hw"; A = os.path.join(R, "analysis_new")
cube = np.load(os.path.join(A, "cache", "raw.npy")).reshape(151, 151, 1000).astype(float)
t = 1053 + 0.05*np.arange(1000); dt = 0.05
R1 = cube[70:81, 75:86].mean((0, 1))
R2 = np.mean([cube[:20, :20], cube[:20, -20:], cube[-20:, :20], cube[-20:, -20:]], axis=(0, 1, 2))
mc = (t >= 1059) & (t <= 1074)                   # crop mask
def dft(x, f, tt):                               # exact DTFT at frequencies f [THz]; time origin 1053 ps
    return (x[None, :]*np.exp(-2j*np.pi*f[:, None]*(tt[None, :]-1053))).sum(1)*dt
ver = {"none":  dict(tt=t,     w=np.ones(1000),         lab="no window (50 ps, df=0.02)"),
       "tukey": dict(tt=t,     w=tukey(1000, 0.2),      lab="Tukey a=0.2 (50 ps, df=0.02)"),
       "crop":  dict(tt=t[mc], w=tukey(mc.sum(), 0.2),  lab="crop 1059-1074 + Tukey a=0.2 (df=%.4f)" % (1/(mc.sum()*dt)))}
col = {"none": "C0", "tukey": "C1", "crop": "C3"}
S = {}
for k, v in ver.items():
    N = len(v["tt"]); f = np.arange(N//2+1)/(N*dt)      # native rfft grid, no padding
    sel = mc if k == "crop" else slice(None)
    S[k] = dict(f=f, X1=dft(R1[sel]*v["w"], f, v["tt"]), X2=dft(R2[sel]*v["w"], f, v["tt"]))
    # check against np.fft.rfft (native length, phase factor for start time)
    chk = np.fft.rfft(R1[sel]*v["w"])*dt*np.exp(-2j*np.pi*f*(v["tt"][0]-1053)); assert np.allclose(chk, S[k]["X1"], atol=1e-6*abs(chk).max())
np.savez(os.path.join(A, "cache", "n6_%s_fft3_nopad.npz" % TS), **{k+"_"+n: S[k][n] for k in S for n in S[k]})
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
for k, v in ver.items():
    f, X1, X2 = S[k]["f"], S[k]["X1"], S[k]["X2"]; mk = "o" if k == "crop" else None
    kw = dict(c=col[k], lw=.9, ms=3, marker=mk, label=v["lab"])
    ax[0, 0].plot(f, 20*np.log10(abs(X1)), **kw); ax[0, 1].plot(f, 20*np.log10(abs(X2)), **kw)
    ax[1, 0].plot(f, abs(X1-X2)/abs(X2), **kw); ax[1, 1].plot(f, np.unwrap(np.angle(X1/X2)), **kw)
for a, ti, yl in zip(ax.ravel(), ["R1 chip center: amplitude", "R2 open corners: amplitude", "|R1-R2|/|R2|", "phase R1/R2 (unwrapped)"], ["dB (nA*ps)", "dB (nA*ps)", "ratio", "rad"]):
    a.set(title=ti, xlabel="f [THz]", ylabel=yl, xlim=(0, 4)); a.grid(alpha=.3); a.legend(fontsize=8)
ax[0, 0].set_ylim(-10, 60); ax[0, 1].set_ylim(-10, 60); ax[1, 0].set_ylim(0, 3.2)
plt.tight_layout(); plt.savefig(os.path.join(A, "figs", "n6_%s_fft3_nopad.png" % TS), dpi=100)
# quantitative: 'none' vs 'tukey' on same grid; 'crop' vs 'none' evaluated by exact DTFT of 'none' at crop freqs
fc = S["crop"]["f"]; Xn1 = dft(R1, fc, t); Xn2 = dft(R2, fc, t)
def rep(name, Xa, Xb, f, lo, hi):
    m = (f >= lo) & (f <= hi); r = np.abs(Xa[m]-Xb[m])/np.abs(Xb[m])
    print("%-22s %.1f-%.1f THz: median |dX|/|X| = %.4f, max = %.4f" % (name, lo, hi, np.median(r), r.max()))
for lo, hi in [(0.2, 3.5), (0.3, 1.6), (3.5, 4.0)]:
    rep("R1 tukey vs none", S["tukey"]["X1"], S["none"]["X1"], S["none"]["f"], lo, hi)
    rep("R2 tukey vs none", S["tukey"]["X2"], S["none"]["X2"], S["none"]["f"], lo, hi)
    rep("R1 crop vs none", S["crop"]["X1"], Xn1, fc, lo, hi); rep("R2 crop vs none", S["crop"]["X2"], Xn2, fc, lo, hi)
