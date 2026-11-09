import time, numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from common import *
FM = np.load(os.path.join(CACHE, "F_0p1_4THz.npy")); g = np.load(os.path.join(CACHE, "geom2d.npy")); geom = g[:5]
fr = np.arange(0.1, 4.001, 0.02)[:FM.shape[0]]
H=0.025; SUB=4
xf = X0+H*np.arange((NX-1)*SUB+1); yf = Y0+H*np.arange((NY-1)*SUB+1); XF, YF = np.meshgrid(xf, yf)
xc,yc,Lx,Ly,th = geom
u=(XF-xc)*np.cos(th)+(YF-yc)*np.sin(th); v=-(XF-xc)*np.sin(th)+(YF-yc)*np.cos(th)
M0=np.clip((Lx/2-abs(u))/H+.5,0,1)*np.clip((Ly/2-abs(v))/H+.5,0,1)
PXX,PYY=np.meshgrid(x,y); m=(abs(PXX-xc)<7)&(abs(PYY-yc)<7)
cnt=[0]
def r(p,data):
    cnt[0]+=1
    M=gaussian_filter1d(M0,abs(p[0])/2/H,axis=1,mode="constant",truncate=5); M=gaussian_filter1d(M,abs(p[1])/2/H,axis=0,mode="constant",truncate=5)[::SUB,::SUB]
    B=np.stack([M[m],np.ones(m.sum())],1); ab,*_=np.linalg.lstsq(B,data[m],rcond=None); return data[m]-B@ab
tot=0
for ft in [0.1,0.3,0.5,1.0,2.0,3.0,4.0]:
    k=int(np.argmin(abs(fr-ft))); cnt[0]=0; t=time.time()
    s=least_squares(r,[1.7,1.3],args=(FM[k].astype(float),),x_scale=[.1,.1]); dt=time.time()-t
    print("f=%.1f: %.1f s per variant, %d evals, w=%s" % (ft, dt, cnt[0], np.round(np.abs(s.x),2)))
