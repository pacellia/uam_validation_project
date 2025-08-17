
from cobaya.theory import Theory
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

class UAMModel(Theory):
    params = {'H0': None, 'omegam': None, 'rd': None}

    def initialize(self):
        self.c_km_s = 299792.458
        self.PI2 = np.pi/2.0
        self.EPS = 1e-9
        self.use_hybrid = getattr(self, 'use_hybrid', False)
        self.w_emit_cache, self.H_cache = {}, {}
        self.dl_cache, self.dm_cache = {}, {}

    
def get_w_emit(self, z):
    """
    PX1 mapping: solve (1+z) = sec^2(w) * exp(tan w)
    Log-domain: ln(1+z) = -2 ln cos w + tan w
    Returns w_emit in (-pi/2 + eps, +pi/2 - eps)
    """
    import numpy as np
    from scipy.optimize import brentq
    if isinstance(z, (list, tuple, np.ndarray)):
        arr = np.asarray(z, dtype=float)
        return np.array([self.get_w_emit(zi) for zi in arr])
    z = float(z)
    if z < 0.0:
        raise ValueError("z must be >= 0")
    if z == 0.0:
        return 0.0
    ln1pz = np.log1p(z)
    a = -self.PI2 + self.EPS
    b = +self.PI2 - self.EPS
    def g(w):
        c = np.cos(w)
        if c == 0.0 or not np.isfinite(c):
            return 1e20  # Very large value
        ln_sec2 = -2.0 * np.log(np.abs(c))
        return ln1pz - (ln_sec2 + np.tan(w))
    # Try main bracket; else scanned grid
    ga, gb = g(a), g(b)
    if not (np.isfinite(ga) and np.isfinite(gb) and ga * gb <= 0):
        grid = np.linspace(a, b, 800)
        prev_w, prev_g = grid[0], g(grid)
        for w in grid[1:]:
            gw = g(w)
            if np.isfinite(prev_g) and np.isfinite(gw) and prev_g*gw <= 0:
                a, b = prev_w, w
                break
            prev_w, prev_g = w, gw
        else:
            a, b = -1.45, +1.45
    w_emit = brentq(g, a, b, xtol=1e-12, rtol=1e-12, maxiter=300)
    return float(w_emit)

