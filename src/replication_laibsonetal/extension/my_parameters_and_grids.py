from pprint import pprint

import jax.numpy as jnp
import pandas as pd
import plotly.express as px
from lcm import MarkovTransition
from lcm.typing import Period
from lcm import PiecewiseLogSpacedGrid, PiecewiseLinSpacedGrid, Piece

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    Model,
    Regime,
    categorical,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)

import lcm.shocks.ar1
import lcm.shocks.iid


import numpy as np
import pandas as pd
from pathlib import Path

from extension.my_model_functions import load_survival_probs


### Grids ###

wealth_x_grid    = LinSpacedGrid(start=-5_000, stop=400_000, n_points=25) 
wealth_illiquid_grid  = LinSpacedGrid(start=2_000, stop=3_500_000, n_points=25)
investment_x_grid     = LinSpacedGrid(start=-50_000, stop=50_000, n_points=100)
investment_z_grid     = LinSpacedGrid(start=-100_000, stop=100_000, n_points=100)

age_grid = AgeGrid(start=20, stop=91, step="1Y")


### Parameters ###
retirement_age = 64
dead_age = 91

SRC = Path('.').resolve().parent

project_path = SRC/ "replication_laibsonetal" / "data"
death_m_path = project_path / "DeathProbsE_M_Hist_TR2023.csv"
death_f_path = project_path / "DeathProbsE_F_Hist_TR2023.csv"

survival_probs = load_survival_probs(death_f_path, death_m_path)


params = {
    "discount_factor":    0.96,
    "risk_aversion":      1.5,
    "interest_rate":      0.0203,
    "interest_rate_debt": 0.1059, 
    "interest_rate_illiquid": 0,
    "working_life": {
        "next_regime": {
            "last_working_age": retirement_age -1,
            "survival_probs": survival_probs,
        },
        "number_of_kids": {
            "a0kids": 0.003,
            "a1kids": 0.358,
            "a2kids": 0.508,
        },
        "number_of_depadul": {
            "a0depadul": 0.00000459,
            "a1depadul": 0.452,
            "a2depadul": 0.438,
        },
        "deterministic": {
            "ywork_cons":      7.563,
            "ywork_agecoeff":  0.135,
            "ywork_age2coeff": -0.222,
            "ywork_age3coeff": 0.106,
            "ywork_kidscoeff": 0.013,
            "ywork_spousecoeff": 0.319,
            "ywork_depadulcoeff": 0.237,
        },
        "household_size": {
            "a0kids": 0.003,
            "a1kids": 0.358,
            "a2kids": 0.508,
            "a0depadul": 0.00000459,
            "a1depadul": 0.452,
            "a2depadul": 0.438,
        },
        "average_income": {
            "ywork_auto": 0.840,
            "ywork_vareps": 0.057,
            "ywork_varnu": 0.045,
        },
        "credit_limit": {
            "c0credit": 0.167,
            "c1credit": -0.002,
            "c2credit": 0.014,
        },
    },
     "retirement": {
         "next_regime": {
            "survival_probs": survival_probs,
        },
        "number_of_kids": {
            "a0kids": 0.003,
            "a1kids": 0.358,
            "a2kids": 0.508,
        },
        "number_of_depadul": {
            "a0depadul": 0.00000459,
            "a1depadul": 0.452,
            "a2depadul": 0.438,
        },
        "deterministic": {
            "ywork_cons":      7.563,
            "ywork_agecoeff":  0.135,
            "ywork_age2coeff": -0.222,
            "ywork_age3coeff": 0.106,
            "ywork_kidscoeff": 0.013,
            "ywork_spousecoeff": 0.319,
            "ywork_depadulcoeff": 0.237,
        },
        "household_size": {
            "a0kids": 0.003,
            "a1kids": 0.358,
            "a2kids": 0.508,
            "a0depadul": 0.00000459,
            "a1depadul": 0.452,
            "a2depadul": 0.438,
        },
        "average_income": {
            "ywork_auto": 0.840,
            "ywork_vareps": 0.057,
            "ywork_varnu": 0.045,
        },
        "credit_limit": {
            "c0credit": 0.167,
            "c1credit": -0.002,
            "c2credit": 0.014,
        },
    },
    "dead": {
        "utility": {
            "alpha": 0.5, 
            "mean_hhs": 2,
        },
        "average_earnings": {
            "ywork_cons":      7.563,
            "ywork_agecoeff":  0.135,
            "ywork_age2coeff": -0.222,
            "ywork_age3coeff": 0.106,
            "ywork_kidscoeff": 0.013,
            "ywork_spousecoeff": 0.319,
            "ywork_depadulcoeff": 0.237,
            "a0kids": 0.003,
            "a1kids": 0.358,
            "a2kids": 0.508,
            "a0depadul": 0.00000459,
            "a1depadul": 0.452,
            "a2depadul": 0.438,
            "ywork_auto": 0.840,
            "ywork_vareps": 0.057,
            "ywork_varnu": 0.045,
            }}
}