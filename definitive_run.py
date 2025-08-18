
# ======================================================================
# Golden Master — Single Cell
# UAM (Geometric Resistance, β≥0, Ωm fixed=π^3/100, H0 free)  vs  flat ΛCDM (H0, Ωm)
# Pantheon+ likelihood via Cobaya. Robust post-proc with χ²/AIC/BIC and ΔBIC.
# ======================================================================

# 0) Deps
!pip install -q cobaya camb getdist pandas scipy
!cobaya-install sn.pantheonplus -p external_pkgs --no-progress-bar

import os, sys, yaml, glob, re, numpy as np, pandas as pd
from cobaya.run import run
from math import isfinite

# ---------- Common paths ----------
BASE = "/content/GoldenMaster_Plus"
os.makedirs(BASE, exist_ok=True)
CHAINS = os.path.join(BASE, "chains")
os.makedirs(CHAINS, exist_ok=True)

# ---------- 1) Write theories ----------

# 1a) UAM Geometric Resistance (Ωm fixed = π^3/100, β(z) = |1 - 2 sin(w/2)/w|)
uam_code = r'''
from cobaya.theory import Theory
import numpy as np
from scipy.optimize import root
from scipy.interpolate import PchipInterpolator
try:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
except ImportError:
    from scipy.integrate import cumtrapz

class UAMGeomRes(Theory):
    # One free parameter; Ωm fixed internally to π^3/100
    params = {"H0": None}

    def initialize(self):
        self.log.info("✅ UAMGeomRes (β≥0, resistance) with precomputed I(z).")
        self.c_km_s = 299792.458
        self.omega_m_fixed = np.pi**3 / 100.0  # ≈0.31006

        # Precompute grid once
        self._zmax = 2.6
        self._N    = 6000
        z = np.linspace(0.0, self._zmax, self._N)

        # Solve w(z) on subgrid then PCHIP to full grid
        W = self._solve_w_grid(z)

        # β(z) = |1 - 2 sin(w/2)/w| with small-w series
        beta = self._beta_of_w(W)

        # Resistance factor: 1 / (1 + β z/(1+z))^2  (guard denominator)
        denom = 1.0 + beta * (z/(1.0+z))
        denom = np.where(denom > 1e-12, denom, 1e-12)

        # E(z) = sqrt( Ωm (1+z)^3 + (1-Ωm) * sec^2(w) * resistance )
        cosw = np.clip(np.cos(W), 1e-12, 1.0)
        sec2 = 1.0/(cosw*cosw)
        E = np.sqrt(self.omega_m_fixed*(1.0+z)**3 + (1.0 - self.omega_m_fixed) * sec2 * (1.0/(denom*denom)))
        E = np.where(E > 1e-12, E, 1e-12)
        invE = 1.0 / E

        I = cumtrapz(invE, z, initial=0.0)

        self._E_of_z = PchipInterpolator(z, E, extrapolate=True)
        self._I_of_z = PchipInterpolator(z, I, extrapolate=True)

    def _w_root(self, z):
        if z == 0.0:
            return 0.0
        def eqn(w, zval):
            return (1.0+zval)*np.cos(w)**2 - np.exp(-np.tan(w))
        guess = float(-np.arctan(z/(1.0+z)))
        sol = root(eqn, guess, args=(z,), tol=1e-10)
        return sol.x[0] if sol.success else 0.0

    def _solve_w_grid(self, zgrid):
        m = max(1000, len(zgrid)//6)
        idx = np.linspace(0, len(zgrid)-1, m).astype(int)
        zc  = zgrid[idx]
        wc  = np.array([self._w_root(zz) for zz in zc])
        mask = ~np.isfinite(wc)
        if mask.any():
            zcv, wcv = zc[~mask], wc[~mask]
            fill = PchipInterpolator(zcv, wcv, extrapolate=True)
            wc[mask] = fill(zc[mask])
        return PchipInterpolator(zc, wc, extrapolate=True)(zgrid)

    def _beta_of_w(self, w):
        out = np.zeros_like(w)
        small = np.abs(w) < 1e-6
        out[small] = (w[small]*w[small])/24.0
        ws = w[~small]
        out[~small] = np.abs(1.0 - (2.0*np.sin(0.5*ws)/ws))
        return out

    def get_Hubble(self, z, **kwargs):
        H0 = self.provider.get_param("H0")
        z = np.asarray(z)
        E = self._E_of_z(z)
        H = H0 * E
        return np.where((E<=0) | ~np.isfinite(E), np.inf, H)

    def get_luminosity_distance(self, z_array, **kwargs):
        H0 = self.provider.get_param("H0")
        z  = np.atleast_1d(z_array).astype(float)
        I  = self._I_of_z(z)
        Dl = (self.c_km_s / H0) * (1.0 + z) * I
        return Dl if isinstance(z_array, np.ndarray) else float(Dl[0])

    def get_angular_diameter_distance(self, z_array, **kwargs):
        z  = np.atleast_1d(z_array).astype(float)
        Dl = self.get_luminosity_distance(z)
        Da = Dl / (1.0 + z)**2
        return Da if isinstance(z_array, np.ndarray) else float(Da[0])

    def get_can_provide(self):
        return ["Hubble", "luminosity_distance", "angular_diameter_distance"]

    def calculate(self, state, want_derived=True, **params_dict):
        return
'''
with open(os.path.join(BASE, "uam_geomres.py"), "w") as f:
    f.write(uam_code)

# 1b) Flat ΛCDM (H0, Ωm) with precomputed I(z)
lcdm_code = r'''
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
        self.log.info("✅ LCDMFlat initialized (precomputed I[z]).")
        self.c_km_s = 299792.458
        self._zmax = 2.6; self._N=6000
        self._zgrid = np.linspace(0.0, self._zmax, self._N)
        self._last_om = None; self._I_of_z  = None; self._E_of_z  = None

    def _build_interpolants(self, om):
        if (self._last_om is not None) and (abs(om - self._last_om) < 1e-14):
            return
        z = self._zgrid
        E = np.sqrt(om*(1.0+z)**3 + (1.0 - om))
        E = np.where(E > 1e-12, E, 1e-12)
        I = cumtrapz(1.0/E, z, initial=0.0)
        self._E_of_z = PchipInterpolator(z, E, extrapolate=True)
        self._I_of_z = PchipInterpolator(z, I, extrapolate=True)
        self._last_om = om

    def get_Hubble(self, z, **kwargs):
        H0 = self.provider.get_param("H0"); om = self.provider.get_param("omegam")
        self._build_interpolants(om); z = np.asarray(z); E = self._E_of_z(z); H = H0 * E
        return np.where((E<=0)|(~np.isfinite(E)), np.inf, H)

    def get_luminosity_distance(self, z_array, **kwargs):
        H0 = self.provider.get_param("H0"); om = self.provider.get_param("omegam")
        self._build_interpolants(om); z = np.atleast_1d(z_array).astype(float); I = self._I_of_z(z)
        Dl = (self.c_km_s / H0) * (1.0 + z) * I
        return Dl if isinstance(z_array, np.ndarray) else float(Dl[0])

    def get_angular_diameter_distance(self, z_array, **kwargs):
        z = np.atleast_1d(z_array).astype(float); Dl = self.get_luminosity_distance(z)
        Da = Dl / (1.0 + z)**2
        return Da if isinstance(z_array, np.ndarray) else float(Da[0])

    def get_can_provide(self):
        return ["Hubble", "luminosity_distance", "angular_diameter_distance"]

    def calculate(self, state, want_derived=True, **params_dict):
        return
'''
with open(os.path.join(BASE, "lcdm_flat.py"), "w") as f:
    f.write(lcdm_code)

if BASE not in sys.path:
    sys.path.append(BASE)

# ---------- 2) YAMLs ----------
# Pantheon+ + UAM (H0 only)
uam_cfg = {
    "likelihood": {"sn.pantheonplus": None},
    "theory": {"uam_geomres.UAMGeomRes": None},
    "params": {
        "H0": {"prior": {"min": 60.0, "max": 80.0}, "ref": 67.4}
    },
    "sampler": {"mcmc": {"Rminus1_stop": 0.01, "max_tries": 2000}},
    "output": os.path.join(CHAINS, "uam_run"),
    "force": True
}
with open(os.path.join(BASE, "uam.yaml"), "w") as f:
    yaml.dump(uam_cfg, f)

# Pantheon+ + ΛCDM (H0, Ωm)
lcdm_cfg = {
    "likelihood": {"sn.pantheonplus": None},
    "theory": {"lcdm_flat.LCDMFlat": None},
    "params": {
        "H0":     {"prior": {"min": 60.0, "max": 80.0}, "ref": 67.4},
        "omegam": {"prior": {"min": 0.1,  "max": 0.5},  "ref": 0.315}
    },
    "sampler": {"mcmc": {"Rminus1_stop": 0.01, "max_tries": 2000}},
    "output": os.path.join(CHAINS, "lcdm_run"),
    "force": True
}
with open(os.path.join(BASE, "lcdm.yaml"), "w") as f:
    yaml.dump(lcdm_cfg, f)

# ---------- 3) Run both ----------
print("\n--- Running UAM (GeomRes) + Pantheon+ ---")
run(yaml.safe_load(open(os.path.join(BASE, "uam.yaml"), "r")))
print("\n✅ UAM run complete.")

print("\n--- Running ΛCDM (flat) + Pantheon+ ---")
run(yaml.safe_load(open(os.path.join(BASE, "lcdm.yaml"), "r")))
print("\n✅ ΛCDM run complete.")

# ---------- 4) Post-proc helpers ----------
def read_stats(prefix, k, N=1701):
    files = sorted(glob.glob(prefix + ".*.txt"))
    if not files:
        return None, None, None
    df = pd.read_csv(files[0], sep=r"\s+", header=None, engine="python", dtype=str)
    cols = ["weight","likecol"]
    pn = prefix + ".paramnames"
    if os.path.exists(pn):
        with open(pn) as ph:
            for line in ph:
                cols.append(re.sub(r"[^A-Za-z0-9_]", "_", line.strip().split()[0]))
    if len(cols) < df.shape[1]: cols += [f"col{i}" for i in range(len(cols), df.shape[1])]
    elif len(cols) > df.shape[1]: cols = cols[:df.shape[1]]
    df.columns = cols
    for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    like = df["likecol"].to_numpy()
    neg2 = -2.0*like
    chi2 = float(np.nanmin(neg2)) if np.mean(neg2>0) >= 0.9 else float(np.nanmin(2.0*like))
    AIC = chi2 + 2*k
    BIC = chi2 + k*np.log(N)
    return chi2, AIC, BIC

# ---------- 5) Compute stats for both ----------
chi2_u, AIC_u, BIC_u = read_stats(os.path.join(CHAINS, "uam_run"),  k=1, N=1701)
chi2_l, AIC_l, BIC_l = read_stats(os.path.join(CHAINS, "lcdm_run"), k=2, N=1701)

if chi2_u is not None and chi2_l is not None:
    d_chi2 = chi2_l - chi2_u
    d_AIC  = AIC_l  - AIC_u
    d_BIC  = BIC_l  - BIC_u

    print("\n==================== RESULTS ====================")
    print(f"UAM  : χ²={chi2_u:.3f}, AIC={AIC_u:.3f}, BIC={BIC_u:.3f}")
    print(f"ΛCDM : χ²={chi2_l:.3f}, AIC={AIC_l:.3f}, BIC={BIC_l:.3f}")
    print("-------------------------------------------------")
    print(f"Δχ² (ΛCDM−UAM) = {d_chi2:.3f}")
    print(f"ΔAIC(ΛCDM−UAM) = {d_AIC:.3f}")
    print(f"ΔBIC(ΛCDM−UAM) = {d_BIC:.3f}")
    if d_BIC > 10: ev = "Very Strong for UAM"
    elif d_BIC > 6: ev = "Strong for UAM"
    elif d_BIC > 2: ev = "Positive for UAM"
    else: ev = "Weak or No evidence"
    print(f"Evidence (BIC scale): {ev}")
    print("=================================================\n")
else:
    print("\nCould not find one or both chain files for analysis.")
