
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve_triangular

def quick_sn_shape_fit(data_file, cov_file, use_full_cov=True, H0=70.0):
    df = pd.read_csv(data_file, sep=' ', skipinitialspace=True)
    z = df['zHD'].values
    mu_obs = df['MU'].values
    C_full = np.load(cov_file)
    C = C_full if use_full_cov else np.diag(np.diag(C_full))
    import sys, os
    sys.path.insert(0, '/content/uam_validation')
    from theory.uam_model import UAMModel
    class MockProvider:
        def __init__(self, H0):
            self.params = {"H0": H0}
        def get_param(self, name):
            return self.params[name]
    model = UAMModel()
    model.use_hybrid = False
    model.provider = MockProvider(H0)
    model.initialize()
    mu_model = []
    for zi in z:
        dl = model.get_luminosity_distance(zi)
        mu_i = 5.0 * np.log10(dl) + 25.0 if dl > 0 else np.nan
        mu_model.append(mu_i)
    mu_model = np.array(mu_model)
    valid = np.isfinite(mu_model)
    if np.sum(valid) < 10:
        return {"error": "Too few valid points"}
    z_v, mu_obs_v, mu_model_v = z[valid], mu_obs[valid], mu_model[valid]
    C_v = C[valid][:, valid]
    L = cholesky(C_v, lower=True)
    y = mu_obs_v - mu_model_v
    ones = np.ones(len(y))
    z1 = solve_triangular(L, y, lower=True)
    z2 = solve_triangular(L, ones, lower=True)
    delta_mag = np.dot(z1, z2) / np.dot(z2, z2)
    resid = y - delta_mag
    z_resid = solve_triangular(L, resid, lower=True)
    chi2 = np.dot(z_resid, z_resid)
    return {
        "N": len(z_v),
        "H0": H0,
        "delta_mag": delta_mag,
        "chi2": chi2,
        "z": z_v,
        "mu_obs": mu_obs_v,
        "mu_model": mu_model_v + delta_mag,
        "resid": mu_obs_v - (mu_model_v + delta_mag),
        "sigma_mean": np.sqrt(np.mean(np.diag(C_v))),
        "resid_mean": np.mean(mu_obs_v - (mu_model_v + delta_mag)),
        "resid_rms": np.std(mu_obs_v - (mu_model_v + delta_mag))
    }

def plot_hubble_diagram(z, mu_obs, mu_model):
    plt.figure(figsize=(10, 6))
    plt.scatter(z, mu_obs, alpha=0.6, s=20, label='Pantheon+ Data')
    plt.plot(z, mu_model, 'r-', alpha=0.8, label='UAM Model')
    plt.xlabel('Redshift z')
    plt.ylabel('Distance Modulus μ')
    plt.title('Hubble Diagram: UAM vs Pantheon+')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_residuals(z, resid):
    plt.figure(figsize=(10, 4))
    plt.scatter(z, resid, alpha=0.6, s=20)
    plt.axhline(0, color='r', linestyle='--', alpha=0.8)
    plt.xlabel('Redshift z')
    plt.ylabel('Residuals (mag)')
    plt.title('Residuals: Data - UAM Model')
    plt.grid(True, alpha=0.3)
    plt.show()
