"""FFT of ALL 22801 chip1 rows (raster order), no window, no zero padding -> amplitude / phase CSV."""
import os, numpy as np
TS = "20261109_1513"
R = r"C:\Users\cojyi\Desktop\beamsize hw"; A = os.path.join(R, "analysis_new"); os.makedirs(os.path.join(A, "csv"), exist_ok=True)
d = np.load(os.path.join(A, "cache", "raw.npy")).astype(np.float64)       # (22801, 1000) as stored
dt = 0.05; N = d.shape[1]
X = np.fft.rfft(d, axis=1)*dt                      # time origin = first sample (1053 ps); units nA*ps
f = np.fft.rfftfreq(N, dt)                          # THz, 0..10, step 0.02, 501 bins
amp, ph = np.abs(X), np.angle(X)                    # phase wrapped to (-pi, pi]
k = np.arange(d.shape[0]); iy, ix = k//151, k%151
idx = np.column_stack([iy, ix, 63.04+0.1*ix, 39.63+0.1*iy])
hdr = "iy,ix,x_mm,y_mm," + ",".join("%.2f" % v for v in f)
for name, M, fmt in [("amplitude_nAps", amp, "%.6g"), ("phase_rad", ph, "%.5f")]:
    out = os.path.join(A, "csv", "chip1_fft_%s_%s.csv" % (name, TS))
    with open(out, "w") as fh:
        fh.write(hdr + "\n")
        for s in range(0, len(k), 2000):
            np.savetxt(fh, np.column_stack([idx[s:s+2000], M[s:s+2000]]), delimiter=",",
                       fmt=["%d", "%d", "%.2f", "%.2f"] + [fmt]*M.shape[1])
    print(out, os.path.getsize(out)/1e6, "MB")
print("shape", amp.shape, "f0..f1", f[0], f[1], "f_end", f[-1])
