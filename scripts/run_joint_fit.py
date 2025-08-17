
# =======================================================
# PHASE 2: Joint Fit for H0, r_d, and beta
# This script finds the best-fit parameters for the beta-UAM model
# using Pantheon+ SN data and anisotropic BAO data.
# =======================================================
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import quad

# --- Ensure our new model is importable ---
THEORY_DIR = "/content/uam_validation/theory"
if THEORY_DIR not in sys.path:
    sys.path.insert(0, THEORY_DIR)
from uam_model import UAMModel

# --- 1. Load the Verified Data ---
print("--- Loading Verified SN and BAO Data ---")
try:
    # Supernova Data
    SN_TABLE_PATH = "/content/uam_validation/data/pantheon_plus/working_sn_table.csv"
    SN_COV_PATH = "/content/uam_validation/data/pantheon_plus/working_sn_cov_SPD.npy"
    sn_df = pd.read_csv(SN_TABLE_PATH)
    sn_cov = np.load(SN_COV_PATH)
    sn_icov = np.linalg.inv(sn_cov)

    lower_map = {c.lower(): c for c in sn_df.columns}
    z_col = lower_map.get('zcmb') or lower_map.get('z')
    mb_col = lower_map.get('m_b_corr') or lower_map.get('mb')
    if not z_col or not mb_col:
        raise KeyError("Could not find required SN columns.")

    sn_z = sn_df[z_col].values
    sn_mb = sn_df[mb_col].values
    print(f"✅ Loaded Pantheon+ data: {len(sn_z)} supernovae.")

    # BAO Data
    BAO_MEAN_PATH = "/content/uam_validation/data/bao/working_bao_anisotropic.csv"
    BAO_COV_PATH = "/content/uam_validation/data/bao/working_bao_cov_SPD.csv"
    bao_df = pd.read_csv(BAO_MEAN_PATH)
    bao_cov = pd.read_csv(BAO_COV_PATH, header=None).values
    bao_icov = np.linalg.inv(bao_cov)
    bao_z = bao_df['z'].values
    bao_obs = np.concatenate([
        bao_df['DM_over_rd'].values,
        bao_df['DH_over_rd'].values
    ])
    print(f"✅ Loaded BAO data: {len(bao_z)} redshifts.")
    
except Exception as e:
    print(f"❌ FAILED to load data. Error: {e}")
    raise

# --- 2. Setup the UAM Model Provider and Chi-Squared Functions ---
class SimpleProvider:
    def __init__(self, params): self._params = params
    def get_param(self, key): return self._params.get(key)

# --- 3. Define the Objective Function for the Minimizer ---
def objective_function(params):
    H0, rd, beta = params
    model = UAMModel()
    model.provider = SimpleProvider({"H0": H0, "beta": beta})
    model.initialize()
    
    dl_model = np.array([model.get_luminosity_distance(z) for z in sn_z])
    mu_model = 5 * np.log10(dl_model) + 25
    
    Y = sn_mb - mu_model
    A = Y.T @ sn_icov @ Y
    B = np.sum(sn_icov @ Y)
    C = np.sum(sn_icov)
    sn_chi2_min = A - (B*B / C)
    
    c_km_s = 299792.458
    dm_theory = np.array([model.get_luminosity_distance(z) / (1+z) for z in bao_z])
    h_theory = np.array([model.get_Hubble(z) for z in bao_z])
    dh_theory = c_km_s / h_theory
    theory_vec = np.concatenate([dm_theory / rd, dh_theory / rd])
    residual = bao_obs - theory_vec
    bao_chi2 = residual.T @ bao_icov @ residual
    
    return sn_chi2_min + bao_chi2

# --- 4. Run the Minimization ---
print("\n--- Starting Joint Minimization for (H0, rd, beta) ---")
initial_guess = [70.0, 147.0, 1.0] 
result = minimize(
    objective_function, initial_guess, method='Nelder-Mead',
    options={'disp': True, 'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4}
)

# --- 5. Print the Final Results ---
if result.success:
    h0_fit, rd_fit, beta_fit = result.x
    min_chi2 = result.fun
    dof = len(sn_z) + len(bao_obs) - 3
    chi2_per_dof = min_chi2 / dof
    
    print("\n--- ✅ Best-Fit Results ---")
    print(f" H0   = {h0_fit:.4f} km/s/Mpc")
    print(f" r_d  = {rd_fit:.4f} Mpc")
    print(f" beta = {beta_fit:.4f}")
    print("---------------------------------")
    print(f" Minimum χ²      = {min_chi2:.4f}")
    print(f" Degrees of Freedom = {dof}")
    print(f" χ² / dof        = {chi2_per_dof:.4f}")
else:
    print(f"\n--- ❌ Minimization Failed: {result.message} ---")
