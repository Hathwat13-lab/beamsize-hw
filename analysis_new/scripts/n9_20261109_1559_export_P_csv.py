"""P_ij(f) by least-squares projection of E_ij(f) onto the R1-R2 line, all pixels, all 501 bins (no window, no padding)."""
import os, numpy as np
TS = "20261109_1559"
R = r"C:\Users\cojyi\Desktop\beamsize hw"; A = os.path.join(R, "analysis_new")
d = np.load(os.path.join(A, "cache", "raw.npy")).astype(np.float64)       # (22801, 1000) raster order
dt = 0.05; X = np.fft.rfft(d, axis=1)*dt; f = np.fft.rfftfreq(1000, dt)   # same convention as the amplitude/phase CSVs
Xc = X.reshape(151, 151, -1)
R1 = Xc[70:81, 75:86].mean((0, 1))                                        # chip center 11x11 (iy 70-80, ix 75-85), complex mean
R2 = np.mean([Xc[:20, :20], Xc[:20, -20:], Xc[-20:, :20], Xc[-20:, -20:]], axis=(0, 1, 2))   # 4 corners 20x20
D = R1 - R2
P = np.real((X - R2[None, :])*np.conj(D)[None, :])/np.abs(D)[None, :]**2  # least-squares P (real)
k = np.arange(d.shape[0]); iy, ix = k//151, k%151
idx = np.column_stack([iy, ix, 63.04+0.1*ix, 39.63+0.1*iy])
hdr = "iy,ix,x_mm,y_mm," + ",".join("%.2f" % v for v in f)
out = os.path.join(A, "csv", "chip1_P_ij_f_%s.csv" % TS)
with open(out, "w") as fh:
    fh.write(hdr + "\n")
    for s in range(0, len(k), 2000):
        np.savetxt(fh, np.column_stack([idx[s:s+2000], P[s:s+2000]]), delimiter=",", fmt=["%d", "%d", "%.2f", "%.2f"] + ["%.5f"]*P.shape[1])
print(out, os.path.getsize(out)/1e6, "MB")
# per-frequency companion table: R1, R2 and the separation |R1-R2| (P is ill-conditioned where |d| is small)
cmp = np.column_stack([f, np.abs(R1), np.abs(R2), np.abs(D), np.abs(D)/np.abs(R2), np.angle(R1), np.angle(R2)])
np.savetxt(os.path.join(A, "csv", "chip1_P_ij_f_%s_R1R2_table.csv" % TS), cmp, delimiter=",", fmt="%.6g",
           header="f_THz,|R1|,|R2|,|R1-R2|,|R1-R2|/|R2|,phase_R1_rad,phase_R2_rad", comments="")
# sanity: P in the R1 / R2 reference regions and spread outside [-0.2, 1.2]
Pc = P.reshape(151, 151, -1)
for fv in (0.5, 1.0, 1.5, 2.0):
    j = int(round(fv/0.02)); p = Pc[:, :, j]; cen = p[70:81, 75:86]; opn = np.concatenate([p[:20, :20].ravel(), p[-20:, -20:].ravel()])
    print("f=%.1f THz: P(center 11x11) mean %.3f | P(open corners) mean %.3f | frac pixels outside [-0.2,1.2]: %.3f | |d|/|R2| %.2f" % (fv, cen.mean(), opn.mean(), np.mean((p < -0.2) | (p > 1.2)), abs(D[j])/abs(R2[j])))
