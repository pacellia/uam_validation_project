
import numpy as np
import pandas as pd
from cobaya.likelihood import Likelihood
import os
class HubbleMeasurements(Likelihood):
    data_file: str = "hz_data.csv"
    def initialize(self):
        data_path = os.path.join(self.get_code_path(), '..', 'data', self.data_file)
        hz_data = pd.read_csv(data_path)
        self.z = hz_data['z'].values
        self.Hz = hz_data['Hz'].values
        self.err = hz_data['err'].values
    def get_requirements(self):
        return {"Hubble": {"z": self.z}}
    def logp(self, **params_values):
        H_theory = self.provider.get_Hubble(self.z)
        chi2 = np.sum(((self.Hz - H_theory) / self.err)**2)
        return -0.5 * chi2
