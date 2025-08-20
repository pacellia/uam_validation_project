
from cobaya.theory import Theory
import numpy as np
from scipy.interpolate import PchipInterpolator
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz

class LCDMFlat(Theory):
    params = {"H0": None, "omegam": None}
    def initialize(self):
        self.c_km_s = 299792.458
        self._zmax = 2.6
        self._N = 6000
        self._zgrid = np.linspace(0.0, self._zmax, self._N)
        self._last_om = None
        self._I_of_z = None
    def _build_interpolants(self, om):
        if (self._last_om is not None) and (abs(om - self._last_om) < 1e-14): return
        z = self._zgrid
        E = np.sqrt(om * (1.0 + z)**3 + (1.0 - om))
        invE = 1.0 / np.where(E > 1e-12, E, 1e-12)
        I = cumtrapz(invE, z, initial=0.0)
        self._I_of_z = PchipInterpolator(z, I, extrapolate=True)
        self._last_om = om
    def calculate(self, state, want_derived=True, **params_dict):
        om = params_dict.get("omegam", self.omegam)
        self._build_interpolants(om)
    def get_luminosity_distance(self, z_array, **kwargs):
        H0 = self.provider.get_param("H0")
        z = np.atleast_1d(z_array).astype(float)
        I = self._I_of_z(z)
        Dl = (self.c_km_s / H0) * (1.0 + z) * I
        return Dl if isinstance(z_array, np.ndarray) else float(Dl[0])
    def get_Hubble(self, z, **kwargs):
        H0 = self.provider.get_param("H0")
        om = self.provider.get_param("omegam")
        E = np.sqrt(om * (1.0 + np.asarray(z))**3 + (1.0 - om))
        return H0 * E
    def get_angular_diameter_distance(self, z_array, **kwargs):
        Dl = self.get_luminosity_distance(z_array)
        return Dl / (1.0 + np.atleast_1d(z_array))**2
    def get_can_provide(self):
        return ["Hubble", "luminosity_distance", "angular_diameter_distance"]
