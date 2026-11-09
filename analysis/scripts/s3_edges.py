"""4변 에지 피팅: 라인별(분산) + 공동(변수분리) 피팅, 범위 의존성, 부트스트랩, 4변 기하 일관성."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from edgefit import *
rng = np.random.default_rng(0)
Gt = np.load(os.path.join(CACHE, "maps_tmpl.npz")); Gg = np.load(os.path.join(CACHE, "maps_gate.npz"))
MAPS = {"a_tmpl": Gt["a"], "b_tmpl": Gt["b"], "a_gate": Gg["a"], "b_gate": Gg["b"]}
XC, YC = x[int(Gt["ix"])], y[int(Gt["iy"])]
EDGES = ["left", "right", "top", "bottom"]
Lgrid = [3.0, 4.0, 5.0]; Ugrid = [2.0, 2.5, 3.0]
rows = []; geo = {}
fig, axs = plt.subplots(2, 4, figsize=(19, 8))
for edge in EDGES:
    for key, M in MAPS.items():
        sg = +1 if key.startswith("a") else -1
        lines = get_lines(edge, M, XC, YC)
        ws = [joint_fit(lines, sg, Lr, Ur)["w"] for Lr in Lgrid for Ur in Ugrid]
        J = joint_fit(lines, sg, 4.0, 2.5)
        ang = np.degrees(np.arctan(J["m"]))
        wl = J["per_line"][:, 2]/np.sqrt(1+J["m"]**2)
        bs = [joint_fit(lines, sg, 4.0, 2.5, idx=rng.integers(0, len(lines), len(lines)))["w"] for _ in range(100)]
        # 절대 기울기: LR은 dx/dy, TB는 dy/dx  (right/bottom은 q=-axis이므로 부호 반전)
        slope_abs = J["m"] if edge in ("left", "top") else -J["m"]
        c_abs = J["c0"] if edge in ("left", "top") else -J["c0"]
        rows.append(dict(edge=edge, key=key, w=J["w"], boot_std=np.std(bs), line_mean=np.nanmean(wl), line_std=np.nanstd(wl),
                         range_min=min(ws), range_max=max(ws), slope=slope_abs, angle_deg=np.degrees(np.arctan(slope_abs)),
                         pos_mm=c_abs+J["m"]*0 if edge in ("left", "top") else c_abs, rms=J["rms"], A_mean=J["AB"][:, 0].mean()))
        if key in ("a_tmpl", "b_tmpl"):
            ax = axs[0 if key == "a_tmpl" else 1, EDGES.index(edge)]
            cosf = 1/np.sqrt(1+J["m"]**2)
            for j, i in enumerate(J["idx"]):
                sk, q, v = lines[i]; mk = J["sel"][j]; A_, B_ = J["AB"][j]
                perp = (q[mk]-(J["c0"]+J["m"]*(sk-J["sbar"])))*cosf
                ax.plot(perp, (v[mk]-B_)/A_, ".", ms=1.5, alpha=.3, color="C0")
            uu = np.linspace(-4, 2.5, 300); ax.plot(uu, F(uu, J["w"], 1), "r", label="w=%.2f mm" % J["w"])
            ax.set_title("%s %s  tilt %.2f°" % (edge, key, ang)); ax.legend(); ax.set_xlabel("perp. dist. inward [mm]")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "s3_edge_fits.png"), dpi=130)
df = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.float_format", "{:.3f}".format)
print(df.to_string(index=False)); df.to_csv(os.path.join(ROOT, "analysis", "table_broadband_w.csv"), index=False)
