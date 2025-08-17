
import numpy as np
import pandas as pd
from pathlib import Path
from cobaya.likelihood import Likelihood

class SNBAO_UAM_Likelihood(Likelihood):
    def initialize(self):
        base = Path('/content/uam_validation/data')
        # Load SN
        sn_csv = base / 'pantheon_plus/working_sn_table.csv'
        sn_df  = pd.read_csv(sn_csv)
        cov   = np.load(base / 'pantheon_plus/working_sn_cov_SPD.npy')
        self.sn_icov = np.linalg.inv(cov)
        zcol = 'zcmb' if 'zcmb' in sn_df.columns else 'z'
        mcol = 'mu'   if 'mu'   in sn_df.columns else 'm_b_corr'
        self.sn_z, self.sn_mu = sn_df[zcol].values, sn_df[mcol].values
        # Load BAO
        bao_csv = base / 'bao/working_bao_anisotropic.csv'
        bao_df  = pd.read_csv(bao_csv)
        cov2    = pd.read_csv(base / 'bao/working_bao_cov_SPD.csv', header=None).values
        self.bao_icov = np.linalg.inv(cov2)
        self.bao_z   = bao_df['z'].values
        dm, dh       = bao_df['DM_over_rd'].values, bao_df['DH_over_rd'].values
        self.bao_obs = np.concatenate([dm, dh])
        self._calls = 0
        super().initialize()

    def get_requirements(self):
        return {
            'luminosity_distance': {'z': np.concatenate([self.sn_z, self.bao_z])},
            'Hubble':              {'z': self.bao_z},
            'rd':                   None,
        }

    def logp(self, **params_values):
        rd     = float(self.provider.get_param('rd'))
        dL_all = np.array(self.provider.get_result('luminosity_distance'))
        H_bao  = np.array(self.provider.get_result('Hubble'))
        Nsn    = len(self.sn_z)
        dL_sn  = dL_all[:Nsn]
        dL_bao = dL_all[Nsn:]
        # SN χ² (analytic M)
        mu_th    = 5.0 * np.log10(dL_sn) + 25.0
        Y        = self.sn_mu - mu_th
        A        = Y.T @ self.sn_icov @ Y
        B        = float(np.sum(self.sn_icov @ Y))
        C        = float(np.sum(self.sn_icov))
        chi2_sn  = A - (B*B)/C
        # BAO χ²
        c_km    = 299792.458
        dm_th   = dL_bao / (1.0 + self.bao_z)
        dh_th   = c_km / H_bao
        vec_th  = np.concatenate([dm_th/rd, dh_th/rd])
        R       = self.bao_obs - vec_th
        chi2_bao = R.T @ self.bao_icov @ R
        self._calls += 1
        if self._calls % 100 == 1:
            H0   = self.provider.get_param('H0')
            beta = self.provider.get_param('beta')
            self.log.info(f'[call {self._calls}] H0={H0:.2f}, beta={beta:.3f}, rd={rd:.3f}')
        return -0.5 * (chi2_sn + chi2_bao)
