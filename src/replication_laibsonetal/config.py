"""All the general configuration of the project."""

from pathlib import Path
import pandas as pd

SRC = Path(__file__).parent.resolve()
ROOT = SRC.joinpath("..", "..").resolve()

BLD = ROOT.joinpath("bld").resolve()


DOCUMENTS = ROOT.joinpath("documents").resolve()

TEMPLATE_GROUPS = ["marital_status", "highest_qualification"]

# Moments names and age groups

MOMENT_NAMES = (
    [f"fborr_{g}" for g in ["21_30", "31_40", "41_50", "51_60"]]
    + [f"mborr_{g}" for g in ["21_30", "31_40", "41_50", "51_60"]]
    + [f"wdebt_{g}" for g in ["21_30", "31_40", "41_50", "51_60"]]
    + [f"wnodebt_{g}" for g in ["21_30", "31_40", "41_50", "51_60"]]
)

AGE_GROUPS = [
    list(range(21, 31)),
    list(range(31, 41)),
    list(range(41, 51)),
    list(range(51, 61)),
]


# Table 3: benchmark estimates. These numbers are copy-pasted from the paper Laibson et al. 2024.

RESULTS_PRESENT_BIASED = pd.Series([
    0.605, # % Visa 21-30
    0.585, # % Visa 31-40
    0.523, # % Visa 41-50
    0.475, # % Visa 51-60
    0.103, # mean Visa 21-30
    0.117, # mean Visa 31-40
    0.124, # mean Visa 41-50
    0.116, # mean Visa 51-60
    0.913, # wealth 21-30 | debt
    1.412, # wealth 31-40 | debt
    2.640, # wealth 41-50 | debt
    4.723, # wealth 51-60 | debt
    2.324, # wealth 21-30 | no debt
    3.248, # wealth 31-40 | no debt
    4.633, # wealth 41-50 | no debt
    7.475  # wealth 51-60 | no debt
], index=MOMENT_NAMES)

RESULTS_EXPONENTIAL = pd.Series([
    0.309, # % Visa 21-30
    0.287, # % Visa 31-40
    0.299, # % Visa 41-50
    0.257, # % Visa 51-60
    0.044, # mean Visa 21-30
    0.051, # mean Visa 31-40
    0.059, # mean Visa 41-50
    0.034, # mean Visa 51-60
    0.880, # wealth 21-30 | debt
    0.999, # wealth 31-40 | debt
    1.941, # wealth 41-50 | debt
    3.811, # wealth 51-60 | debt
    2.017, # wealth 21-30 | no debt
    2.850, # wealth 31-40 | no debt
    3.967, # wealth 41-50 | no debt
    5.358  # wealth 51-60 | no debt
], index=MOMENT_NAMES)

test_params_present_biased = pd.DataFrame(
    {"value": [0.5305, 0.9891, 1.9355]},
    index=["beta", "delta", "rho"],
)

test_params_exponential = pd.DataFrame(
    {"value": [1, 0.9601, 1.4663]},
    index=["beta", "delta", "rho"],
)