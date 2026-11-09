import numpy as np, os
ROOT = r"C:\Users\cojyi\Desktop\beamsize hw"
CACHE = os.path.join(ROOT, "analysis", "cache")
FIG = os.path.join(ROOT, "analysis", "figs")
NY = NX = 151; NT = 1000; DT = 0.05; T0 = 1053.0; STEP = 0.1
X0, Y0 = 63.04, 39.63
x = X0 + STEP*np.arange(NX); y = Y0 + STEP*np.arange(NY)
t = T0 + DT*np.arange(NT)

def load():
    p = os.path.join(CACHE, "chip1.npy")
    if os.path.exists(p):
        return np.load(p, mmap_mode="r")
    raw = os.path.join(ROOT, "chip1.tim.csv")
    a = np.loadtxt(raw, skiprows=3, delimiter=",", dtype=np.float32)
    assert a.shape == (NY*NX, NT), a.shape
    a = a.reshape(NY, NX, NT)
    np.save(p, a)
    return a
