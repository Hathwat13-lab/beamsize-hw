"""공통 에지 피팅 도구: 라인 추출, 라인별 erf, 변수분리(variable projection) 공동 피팅."""
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.special import erf
from common import *
HALF = 1.5
S2 = np.sqrt(2)
def F(perp, w, sg=1): return 0.5*(1 + sg*erf(S2*perp/w))

def get_lines(edge, M, XC, YC, half=HALF):
    """각 라인: (s [mm along], q [mm, 안쪽으로 증가하는 좌표, 칩 중심 쪽 절반만], 값)."""
    out = []
    if edge in ("left", "right"):
        for r in np.where(abs(y-YC) <= half+1e-9)[0]:
            ax, v, c = x, M[r, :], XC; s = y[r]
            sel = ax <= c if edge == "left" else ax >= c
            q = ax if edge == "left" else -ax
            out.append((s, q[sel], v[sel]))
    else:
        for c_ in np.where(abs(x-XC) <= half+1e-9)[0]:
            ax, v, c = y, M[:, c_], YC; s = x[c_]
            sel = ax <= c if edge == "top" else ax >= c
            q = ax if edge == "top" else -ax
            out.append((s, q[sel], v[sel]))
    return out

def line_fit(q, v, sg, Lr, Ur):
    lvl = v.min() + 0.5*np.ptp(v)
    k = np.argmax((v-lvl) > 0) if sg > 0 else np.argmax((v-lvl) < 0)
    q0 = q[k]; m = (q >= q0-Lr) & (q <= q0+Ur)
    f = lambda u, A, u0, w, B: A*F(u-u0, abs(w), sg) + B
    try:
        p, _ = curve_fit(f, q[m], v[m], p0=[np.ptp(v), q0, 2.0, v.min()], maxfev=20000)
        p[2] = abs(p[2]); return p            # A,u0(abs q),w,B
    except Exception:
        return np.full(4, np.nan)

def joint_fit(lines, sg, Lr, Ur, idx=None, init=None, fix=None):
    """공유 (c0, 기울기 m, w), 라인별 (A_k,B_k)는 선형으로 소거. fix=(c0,m)이면 w만 피팅."""
    idx = np.arange(len(lines)) if idx is None else np.asarray(idx)
    s = np.array([lines[i][0] for i in idx]); sbar = np.mean([l[0] for l in lines])
    if fix is None:
        pl = np.array([line_fit(lines[i][1], lines[i][2], sg, Lr, Ur) for i in idx]); ok = np.isfinite(pl[:, 0])
        m0, c00 = np.polyfit(s[ok]-sbar, pl[ok, 1], 1); w0 = float(np.nanmedian(pl[:, 2]))
    else:
        pl = None; c00, m0 = fix; w0 = 2.0 if init is None else init
    sel = []
    for i in idx:
        sk, q, v = lines[i]; perp = (q-(c00+m0*(sk-sbar)))/np.sqrt(1+m0**2)
        sel.append((perp >= -Lr) & (perp <= Ur))
    def resid(th, ret=False):
        c0, m, w = th; w = abs(w); r = []; AB = []
        cosf = 1/np.sqrt(1+m**2)
        for j, i in enumerate(idx):
            sk, q, v = lines[i]; mk = sel[j]
            perp = (q[mk]-(c0+m*(sk-sbar)))*cosf; Bm = np.stack([F(perp, w, 1), np.ones_like(perp)], 1)
            ab, *_ = np.linalg.lstsq(Bm, v[mk], rcond=None); r.append(v[mk]-Bm@ab); AB.append(ab)
        return (np.concatenate(r), np.array(AB)) if ret else np.concatenate(r)
    if fix is None:
        sol = least_squares(resid, [c00, m0, w0], x_scale=[0.1, 0.01, 0.1]); th = sol.x
    else:
        sol = least_squares(lambda p: resid([c00, m0, p[0]]), [w0], x_scale=[0.1]); th = np.array([c00, m0, sol.x[0]])
    r, AB = resid(th, True)
    return dict(c0=th[0], m=th[1], w=abs(th[2]), AB=AB, rms=np.sqrt(np.mean(r**2)), s=s, sbar=sbar, sel=sel, per_line=pl, idx=idx)
