"""SSIM (совместимый со scikit-image: окно 7x7, data_range=255) и его аналитический градиент."""
import numpy as np
from scipy.ndimage import uniform_filter
W=7; CN=W*W/(W*W-1); PAD=(W-1)//2
def ssim_grad(x,y,L=255.):
    """x,y float images. returns mean SSIM (skimage-compatible) and dS/dx (borders approx)."""
    C1=(0.01*L)**2; C2=(0.03*L)**2
    U=lambda a: uniform_filter(a,size=W,mode='reflect')
    mx,my=U(x),U(y)
    sxx=CN*(U(x*x)-mx*mx); syy=CN*(U(y*y)-my*my); sxy=CN*(U(x*y)-mx*my)
    A1=2*mx*my+C1; A2=2*sxy+C2; B1=mx*mx+my*my+C1; B2=sxx+syy+C2
    S=A1*A2/(B1*B2)
    mask=np.zeros_like(x); mask[PAD:-PAD,PAD:-PAD]=1; n=mask.sum()
    val=(S*mask).sum()/n
    g=mask/n
    dA1=g*A2/(B1*B2); dA2=g*A1/(B1*B2); dB1=-g*S/B1; dB2=-g*S/B2
    g_mx=dA1*2*my+dB1*2*mx
    g_sxy=dA2*2; g_sxx=dB2
    # sxy=CN(U(xy)-mx my); sxx=CN(U(x^2)-mx^2)
    g_mx=g_mx - CN*g_sxy*my - 2*CN*g_sxx*mx
    gx=U(g_mx)+CN*y*U(g_sxy)+2*CN*x*U(g_sxx)
    return val,gx
