
# UAM Cobaya Theory Wrapper
# Wraps the existing UAMModel for use with Cobaya framework

from cobaya.theory import Theory
import numpy as np
from scipy.optimize import root
from scipy.interpolate import PchipInterpolator
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz

class UAMModel:
    # This is the original, standalone "Golden Master" model.
    # We keep it for reference and direct testing.
    params = {"H0": None}
    def __init__(self, H0=70.0):
        self.H0 = H0
        self.c_km_s = 299792.458
        self.omega_m_fixed = np.pi**3 / 100.0
        self._zmax = 2.6
        self._N = 6000
        self._zgrid = np.linspace(0.0, self._zmax, self._N)
        self._build_interpolants()
    def _build_interpolants(self):
        z = self._zgrid
        w = self._solve_w_grid(z)
        beta = self._beta_of_w(w)
        cosw = np.cos(w)
        cosw_safe = np.where(np.abs(cosw) < 1e-12, 1e-12, cosw)
        sec2 = 1.0 / (cosw_safe**2)
        denom = 1.0 + beta * z / (1.0 + z)
        denom_safe = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        E_sq = self.omega_m_fixed * (1.0 + z)**3 + (1.0 - self.omega_m_fixed) * sec2 * (1.0 / (denom_safe**2))
        E = np.sqrt(np.maximum(E_sq, 1e-24))
        invE = 1.0 / E
        I = cumtrapz(invE, z, initial=0.0)
        self._E_of_z = PchipInterpolator(z, E, extrapolate=True)
        self._I_of_z = PchipInterpolator(z, I, extrapolate=True)
    def _w_root(self, z):
        if z == 0.0: return 0.0
        def eqn(w, zval): return (1.0 + zval) * np.cos(w)**2 - np.exp(-np.tan(w))
        guess = float(-np.arctan(z / (1.0 + z)))
        try:
            sol = root(eqn, guess, args=(z,), tol=1e-10, method='hybr')
            return sol.x[0] if sol.success else np.nan
        except Exception:
            return np.nan
    def _solve_w_grid(self, zgrid):
        w_vals = np.array([self._w_root(zz) for zz in zgrid])
        if np.any(~np.isfinite(w_vals)):
            finite_z = zgrid[np.isfinite(w_vals)]
            finite_w = w_vals[np.isfinite(w_vals)]
            if len(finite_z) < 2: w_vals = np.nan_to_num(w_vals)
            else:
                interp_func = PchipInterpolator(finite_z, finite_w, extrapolate=True)
                w_vals = interp_func(zgrid)
        return w_vals
    def _beta_of_w(self, w):
        out = np.zeros_like(w)
        small_mask = np.abs(w) < 1e-6
        out[small_mask] = (w[small_mask]**2) / 24.0
        large_mask = ~small_mask
        ws = w[large_mask]
        out[large_mask] = np.abs(1.0 - (2.0 * np.sin(0.5 * ws) / ws))
        return out
    def get_luminosity_distance(self, z_array):
        z = np.atleast_1d(z_array).astype(float)
        I = self._I_of_z(z)
        Dl = (self.c_km_s / self.H0) * (1.0 + z) * I
        return Dl if isinstance(z_array, np.ndarray) else float(Dl[0])

class UAMCobaya(Theory):
    # This is the wrapper that makes the UAM model compatible with Cobaya.
    params = {"H0": None}
    def initialize(self):
        self.uam_instance = UAMModel(H0=70.0) # H0 will be updated by Cobaya
    def get_Hubble(self, z, **kwargs):
        H0_current = self.provider.get_param("H0")
        self.uam_instance.H0 = H0_current
        return H0_current * self.uam_instance._E_of_z(z)
    def get_luminosity_distance(self, z_array, **kwargs):
        H0_current = self.provider.get_param("H0")
        self.uam_instance.H0 = H0_current
        return self.uam_instance.get_luminosity_distance(z_array)
    def get_angular_diameter_distance(self, z_array, **kwargs):
        Dl = self.get_luminosity_distance(z_array)
        return Dl / (1.0 + np.atleast_1d(z_array))**2
    def get_can_provide(self):
        return ["Hubble", "luminosity_distance", "angular_diameter_distance"]
    def calculate(self, state, want_derived=True, **params_dict):
        # Update H0 for each new point in the chain
        H0_current = params_dict.get('H0')
        if H0_current is not None:
            self.uam_instance.H0 = H0_current
