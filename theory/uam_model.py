# uam_model.py
# Beta-generalized UAM distances (SN/BAO-ready)
# This model is now the new standard.

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad

class UAMModel:  # <-- Name changed from UAMModelBeta to UAMModel
    """
    Beta-generalized Unified Angular Mathematics (UAM) model.
    """

    def __init__(self):
        pass

    def initialize(self):
        self.c_km_s   = 299792.458
        self.PI2      = np.pi / 2.0
        self.EPS      = 1e-9
        self._exp_clip = 700.0
        self.w_cache  = {}
        self.H_cache  = {}
        self.dl_cache = {}

    @property
    def provider(self):
        return self._provider

    @provider.setter
    def provider(self, p):
        self._provider = p

    def _get_param_with_default(self, key, default):
        try:
            return float(self.provider.get_param(key))
        except Exception:
            return float(default)

    def _residual_beta(self, w, z, beta):
        cw  = np.cos(w)
        c2  = cw * cw
        if c2 < self.EPS:
            c2 = self.EPS
        tw  = np.tan(w)
        exp_arg = -beta * tw
        exp_arg = np.clip(exp_arg, -self._exp_clip, self._exp_clip)
        return (1.0 + z) - c2 * np.exp(exp_arg)

    def get_w_emit(self, z):
        zf   = float(z)
        beta = self._get_param_with_default("beta", 1.0)
        key  = (round(zf, 12), beta)
        if key in self.w_cache:
            return self.w_cache[key]
        if zf <= 0.0:
            self.w_cache[key] = 0.0
            return 0.0
        eps  = 1e-6
        a    = -(self.PI2 - eps)
        b    = 0.0
        Fa = self._residual_beta(a, zf, beta)
        Fb = self._residual_beta(b, zf, beta)
        if Fa * Fb > 0.0:
            step = 1e-3
            aa   = a
            max_iter = int((self.PI2 - 2*eps) / step) + 5
            it = 0
            while it < max_iter and Fa * Fb > 0.0 and aa < -1e-9:
                aa += step
                Fa = self._residual_beta(aa, zf, beta)
                it += 1
            a = aa
        if Fa * Fb > 0.0:
            bb = -1e-6
            Fb = self._residual_beta(bb, zf, beta)
            if Fa * Fb > 0.0:
                step_b = 1e-3
                max_itb = int((self.PI2 - 2*eps) / step_b) + 5
                itb = 0
                while itb < max_itb and Fa * Fb > 0.0 and bb > -(self.PI2 - 2e-3):
                    bb -= step_b
                    Fb  = self._residual_beta(bb, zf, beta)
                    itb += 1
            b = bb
        if Fa * Fb > 0.0:
            raise ValueError(f"Beta-UAM bracketing failed for z={zf}, beta={beta}: F(a)={Fa}, F(b)={Fb}")
        w = brentq(lambda w_: self._residual_beta(w_, zf, beta),
                   a, b, xtol=1e-12, rtol=1e-12, maxiter=500)
        self.w_cache[key] = w
        return w

    def get_Hubble(self, z):
        zf   = float(z)
        H0   = self._get_param_with_default("H0", 70.0)
        beta = self._get_param_with_default("beta", 1.0)
        key  = (round(zf, 10), H0, beta)
        if key in self.H_cache:
            return self.H_cache[key]
        w   = self.get_w_emit(zf)
        tw  = np.tan(w)
        denom = (1.0 + zf) * (2.0 * tw + beta)
        if np.abs(denom) < self.EPS:
            denom = np.sign(denom) * self.EPS if denom != 0.0 else self.EPS
        E = 1.0 / denom
        H = H0 * E
        self.H_cache[key] = H
        return H

    def get_luminosity_distance(self, z):
        zf   = float(z)
        if zf <= 0.0:
            return 0.0
        H0   = self._get_param_with_default("H0", 70.0)
        beta = self._get_param_with_default("beta", 1.0)
        key  = (round(zf, 10), H0, beta)
        if key in self.dl_cache:
            return self.dl_cache[key]
        def integrand(zz):
            Hz = self.get_Hubble(zz)
            if Hz <= 0.0:
                Hz = self.EPS
            return self.c_km_s / Hz
        DM, _ = quad(integrand, 0.0, zf, epsabs=1e-8, epsrel=1e-6, limit=300)
        DL = (1.0 + zf) * DM
        self.dl_cache[key] = DL
        return DL
