"""주파수별 w(f): 복소 스펙트럼 선형 혼합 모델 E_ij(f)=F_ij(f) R1(f) + (1-F_ij(f)) R2(f)
R1=칩 중앙 평균 스펙트럼, R2=개방 코너 평균 스펙트럼. 게이팅 불필요(모든 처리가 선형이라 윈도우 무관).
픽셀·주파수별로 F=Re[(E-R2) conj(d)]/|d|^2, d=R1-R2 (최소제곱 사영) -> F 맵에 erf 피팅."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from edgefit import *
rng = np.random.default_rng(1)
D = np.asarray(load(), dtype=np.float64)
Gt = np.load(os.path.join(CACHE, "maps_tmpl.npz")); ix, iy = int(Gt["ix"]), int(Gt["iy"])
XC, YC = x[ix], y[iy]
win = tukey(NT, 0.2)
E = np.fft.rfft(D*win, axis=2); fr = np.fft.rfftfreq(NT, DT)       # THz
R1 = E[iy-5:iy+6, ix-5:ix+6].mean((0, 1))
n = 20; corn = np.concatenate([E[:n, :n].reshape(-1, len(fr)), E[:n, -n:].reshape(-1, len(fr)), E[-n:, :n].reshape(-1, len(fr)), E[-n:, -n:].reshape(-1, len(fr))])
R2 = corn.mean(0)
blk = np.concatenate([E[:n, :n], E[-n:, -n:]]); df_ = blk[:, 1:] - blk[:, :-1]
sig = np.sqrt((abs(df_)**2).mean((0, 1))/4)   # 인접 픽셀 차이로 추정한 성분당 잡음(공간 구조 제외)
d = R1-R2
snr_d = abs(d)/np.maximum(sig, 1e-12)
fsel = np.where((fr >= 0.10) & (fr <= 2.50))[0]
FM = np.empty((len(fsel), NY, NX), np.float32); PERP = np.empty_like(FM)
for k, j in enumerate(fsel):
    z = (E[:, :, j]-R2[j])*np.conj(d[j]); FM[k] = z.real/abs(d[j])**2; PERP[k] = z.imag/abs(d[j])**2
np.savez(os.path.join(CACHE, "maps_freq.npz"), F=FM, perp=PERP, f=fr[fsel], R1=R1, R2=R2, sig=sig)
# 모델 검증: 에지 영역에서 E가 R1-R2 직선 위에 있는가 (수직 성분 vs 잡음)
ar = FM[:, :, :]
edge_zone = (abs(FM[len(fsel)//4]) > 0.1) & (abs(FM[len(fsel)//4]-1) > 0.1)
rows_chk = []
for k in range(0, len(fsel), 8):
    m = (FM[k] > 0.15) & (FM[k] < 0.85)
    rows_chk.append((fr[fsel][k], abs(d[fsel][k]), snr_d[fsel][k], np.sqrt((PERP[k][m]**2).mean())*abs(d[fsel][k]), sig[fsel][k]))
chk = pd.DataFrame(rows_chk, columns=["f_THz", "|d|", "SNR_d", "perp_resid_edge[nA·ps]", "noise/comp"])
print(chk.to_string(index=False, float_format=lambda v: "%.3g" % v))

EDGES = ["left", "right", "top", "bottom"]
good = (abs(d)/abs(R2) > 0.4) & (snr_d > 15)
fgood = [k for k, j in enumerate(fsel) if good[j] and 0.25 <= fr[j] <= 1.3]
print("geometry bins:", len(fgood))
geo = {}
for edge in EDGES:
    cs, ms = [], []
    for k in fgood:
        J = joint_fit(get_lines(edge, FM[k].astype(float), XC, YC), +1, 4.0, 2.5); cs.append(J["c0"]); ms.append(J["m"])
    geo[edge] = (np.median(cs), np.median(ms)); print(edge, "c0 %.3f±%.3f  slope %.4f±%.4f" % (geo[edge][0], np.std(cs), geo[edge][1], np.std(ms)))

rows = []
ranges = [(4.0, 2.5), (3.0, 2.0), (5.0, 3.0)]
for k, j in enumerate(fsel):
    for edge in EDGES:
        lines = get_lines(edge, FM[k].astype(float), XC, YC)
        ws = [joint_fit(lines, +1, Lr, Ur, fix=geo[edge])["w"] for Lr, Ur in ranges]
        J = joint_fit(lines, +1, 4.0, 2.5, fix=geo[edge])
        bs = [joint_fit(lines, +1, 4.0, 2.5, idx=rng.integers(0, len(lines), len(lines)), fix=geo[edge])["w"] for _ in range(20)]
        rows.append(dict(f=fr[j], edge=edge, w=J["w"], boot=np.std(bs), rmin=min(ws), rmax=max(ws), rms=J["rms"], A=J["AB"][:, 0].mean(), snr_d=snr_d[j], rel_d=abs(d[j])/abs(R2[j]), ok=bool(good[j])))
df = pd.DataFrame(rows); df.to_csv(os.path.join(ROOT, "analysis", "table_wf_edges.csv"), index=False)
print(df[df.f.round(2).isin([0.3, 0.5, 0.7, 1.0, 1.5])].to_string(index=False, float_format=lambda v: "%.3f" % v))

# ---- 요약/그림: 사용 대역 = ok 빈 (SNR_d>15, |d|/|R2|>0.4) ----
fig, ax = plt.subplots(1, 3, figsize=(19, 5))
col = dict(left="C0", right="C1", top="C2", bottom="C3")
for e in EDGES:
    s_ = df[(df.edge == e) & df.ok]
    err = np.sqrt(s_.boot**2 + ((s_.rmax-s_.rmin)/2)**2)
    ax[0].errorbar(s_.f, s_.w, err, fmt="o", ms=3, color=col[e], label=e)
ax[0].set_xlabel("f [THz]"); ax[0].set_ylabel("w [mm]"); ax[0].legend(); ax[0].set_title("w(f) per edge (ok bins)"); ax[0].set_ylim(0, 6)
fq = np.linspace(0.2, 2.0, 50); 
for e in EDGES:
    s_ = df[(df.edge == e) & df.ok]; lam = 0.2998/s_.f
    ax[1].plot(s_.f, s_.w/lam, "o", ms=3, color=col[e], label=e)
ax[1].set_xlabel("f [THz]"); ax[1].set_ylabel("w/λ"); ax[1].set_title("w/λ (flat if w ∝ λ)"); ax[1].legend(); ax[1].set_ylim(0, 10)
ax[2].semilogy(fr[fsel], abs(R2[fsel]), label="|R2| open"); ax[2].semilogy(fr[fsel], abs(d[fsel]), label="|R1-R2|"); ax[2].semilogy(fr[fsel], sig[fsel], label="noise/component")
ax[2].set_xlabel("f [THz]"); ax[2].legend(); ax[2].set_title("spectra")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "s4_wf.png"), dpi=130)
piv = df[df.ok].assign(fb=lambda q: (q.f/0.2).round()*0.2).groupby(["fb", "edge"]).w.median().unstack()
piv["w_x"] = piv[["left", "right"]].mean(1); piv["w_y"] = piv[["top", "bottom"]].mean(1)
print(piv.to_string(float_format=lambda v: "%.2f" % v)); piv.to_csv(os.path.join(ROOT, "analysis", "table_wf_binned.csv"))
