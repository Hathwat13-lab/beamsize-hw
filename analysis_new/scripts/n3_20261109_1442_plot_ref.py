"""Plot ref.pulse.csv and rerfe.spectr.csv (raw, as stored)."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = r"C:\Users\cojyi\Desktop\beamsize hw"; A = os.path.join(R, "analysis_new")
p = pd.read_csv(os.path.join(R, "ref.pulse.csv")); s = pd.read_csv(os.path.join(R, "rerfe.spectr.csv"))
print(p.describe().T[["min", "max"]]); print(list(s.columns))
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
t = p.iloc[:, 0]
ax[0, 0].plot(t, p["Signal_1/nA"], label="Signal_1"); ax[0, 0].plot(t, p["Signal_2/nA"], label="Signal_2")
ax[0, 0].plot(t, p["Reference_1/nA"], "--", lw=.8, label="Reference_1"); ax[0, 0].plot(t, p["Reference_2/nA"], "--", lw=.8, label="Reference_2")
ax[0, 0].set(title="ref.pulse.csv (full, 1000 pts, dt 0.05 ps)", xlabel="t [ps]", ylabel="current [nA]"); ax[0, 0].legend()
m = (t > 1058) & (t < 1075)
ax[0, 1].plot(t[m], p["Signal_1/nA"][m], ".-", ms=3); ax[0, 1].set(title="Signal_1 zoom 1058-1075 ps", xlabel="t [ps]", ylabel="nA")
ax[1, 0].plot(s["Frequency_1/THz"], s["Power_Spectrum_1/dB"], label="Power_1"); ax[1, 0].plot(s["Frequency_2/THz"], s["Power_Spectrum_2/dB"], label="Power_2")
ax[1, 0].plot(s["Frequency_1/THz"], s["Reference_Spectrum_1/dB"], "--", lw=.8, label="Ref_1")
ax[1, 0].set(title="rerfe.spectr.csv power spectrum", xlabel="f [THz]", ylabel="dB", ylim=(-130, 0)); ax[1, 0].legend()
ph = [c for c in s.columns if c.startswith("Phase")]
for c in ph: ax[1, 1].plot(s["Frequency_1/THz"], s[c], label=c)
ax[1, 1].set(title="phase (unwrapped)", xlabel="f [THz]", ylabel="rad"); ax[1, 1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(A, "figs", "n3_20261109_1442_ref_pulse_spectrum.png"), dpi=100)
