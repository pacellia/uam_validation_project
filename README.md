
# Definitive UAM vs. ΛCDM Validation

This repository archives the definitive MCMC run of the Unified Acceleration Model (UAM) with the "Geometric Resistance" formulation, tested against the Pantheon+ supernova dataset (N=1701) using Cobaya.

## Overview
- **Model (UAM):** UAMGeomRes (1 free parameter: H0), with Ωm fixed to π³/100 ≈ 0.31006 and β(z) derived from geometry as |1 - 2*sin(w/2)/w|.
- **Model (ΛCDM):** Standard flat ΛCDM (2 free parameters: H0, Ωm).
- **Likelihood:** `sn.pantheonplus` (1701 supernovae) for both models.

## How to Reproduce
Run the `golden_master_run.py` script in a Google Colab environment. It is a self-contained script that installs dependencies, defines the theories, runs both MCMC analyses, and prints the final comparison.

## Definitive Results
- **UAM (k=1):** χ² = 1409.023, BIC = 1416.462
- **ΛCDM (k=2):** χ² = 1407.078, BIC = 1421.956
- **Verdict (ΔBIC):** +5.494 (Positive evidence for UAM)
