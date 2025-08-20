
# UAM Cosmological Validation Project

This repository contains the code, data, MCMC chains, and results for the cosmological validation of the Unifying Aether Model (UAM) against the standard Lambda-CDM (ΛCDM) model.

## Key Finding
The primary conclusion is that the one-parameter UAM "Golden Master" model is statistically competitive with, and on some metrics superior to, the two-parameter flat ΛCDM model across multiple independent cosmological probes.

### Pantheon+SH0ES Supernovae (Distance Modulus)
- **Result:** UAM is statistically preferred by the Bayesian Information Criterion (BIC).
- **ΔBIC (BIC_ΛCDM - BIC_UAM):** +5.5 (Positive evidence for UAM)

### Cosmic Chronometers (Hubble Parameter)
- **Result:** UAM achieves a statistically identical fit to ΛCDM while using one fewer free parameter, validating its geometric predictions.
- **ΔBIC (BIC_ΛCDM - BIC_UAM):** +1.2 (Weak evidence for UAM)

## Repository Structure
- **/reports/**: Final markdown summary reports.
- **/code/**: Python scripts required to reproduce all analyses.
- **/data/**: Raw data files used in the analyses.
- **/chains/**: Full MCMC chain files produced by Cobaya.
- **README.md**: This file.
