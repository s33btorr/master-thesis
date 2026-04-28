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

from model_functions import load_survival_probs


### Grids ###
wealth_illiquid_grid = PiecewiseLogSpacedGrid(
    pieces=(
        Piece(interval="[1, 100_000)", n_points=35), #n_points=40 por alguna razon, esto con menos puntos no funciona........!!!!!!!!
        Piece(interval="[100_000, 3_500_000]", n_points=15), #n_points=34
    )
)

wealth_liquid_grid = PiecewiseLinSpacedGrid(
    pieces=(
        Piece(interval="[-15_000, 0)", n_points=25), #n_points=45
        Piece(interval="[0, 15_000)", n_points=25), #n_points=50
        #Piece(interval="[50_000, 400_000]", n_points=31), #n_points=62
    )
)

age_grid = AgeGrid(start=20, stop=91, step="1Y")


### Parameters ###
retirement_age = 64
dead_age = 91 # pongo 91 porque vive hasta los 90, pero en este caso el periodo 91 es necesario ya que es el periodo de muerte

SRC = Path('.').resolve().parent
#print("CWD:", Path('.').resolve().parent)

project_path = SRC / "data"
death_m_path = project_path / "DeathProbsE_M_Hist_TR2023.csv"
death_f_path = project_path / "DeathProbsE_F_Hist_TR2023.csv"

survival_probs = load_survival_probs(death_f_path, death_m_path)


params = {
    "discount_factor":    0.96, # is it a problem that dead does not need anything and I still add them here?
    "risk_aversion":      1.5,
    "interest_rate":      0.0203,
    "interest_rate_debt": 0.1059, 
    "interest_rate_illiquid": 0, # habia puesto 1 y esto hacia que explotara porque
    "working_life": {
        "next_regime": {
            "last_working_age": retirement_age -1, #OJO NO ME MUESTRA ESTO EN EL DICC Y ES REPETITIVO (4 hmvg)
            "survival_probs": survival_probs,
        },
       # "next_regime_working": {
        #    "last_working_age": retirement_age -1,  # no entiendo la logica de por que hay que restar 1... lo hacen asi en el tiny sample
         #   "final_age":        dead_age - 1,         # 89.0
          #  "survival_probs": survival_probs,
        #},
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
        "borrowing_constraint": {
            "c0credit": 0.167,
            "c1credit": -0.002,
            "c2credit": 0.014,
            #"interest_rate": 0.0203,
            #"interest_rate_debt": 0.1059,
        },
        "household_size": {
            "a0kids": 0.003,
            "a1kids": 0.358,
            "a2kids": 0.508,
            "a0depadul": 0.00000459,
            "a1depadul": 0.452,
            "a2depadul": 0.438,
        },

    },
     "retirement": {
         "next_regime": {
            #"last_working_age": retirement_age -1,
            "survival_probs": survival_probs,
            #"final_age":        dead_age - 1,
        },
        #"next_regime_retirement": {
         #   "last_working_age": retirement_age -1,  
          #  "final_age":        dead_age - 1,
           # "survival_probs": survival_probs,
        #},
        #"deterministic": {
         #   "yret_cons":     6.0,
          #  "yret_agecoeff": -0.01,
        #},
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
        "borrowing_constraint": {
            "c0credit": 0.167,
            "c1credit": -0.002,
            "c2credit": 0.014,
            #"interest_rate": 0.0203,
            #"interest_rate_debt": 0.1059,
        },
        "household_size": {
            "a0kids": 0.003,
            "a1kids": 0.358,
            "a2kids": 0.508,
            "a0depadul": 0.00000459,
            "a1depadul": 0.452,
            "a2depadul": 0.438,
        },
    },
    "dead": {
        "utility": {
            "alpha": 0.5, # debe haber una forma de no tener que volver a escribir todos estos numeros
            #"delta": 0.96,
            #"interest_rate": 0.02,
            "mean_hhs": 2,
            "mean_hhy": 1000,
            #"risk_aversion": 2.0,
            }}
}


params_naive = {
    "discount_factor":    0.99, # is it a problem that dead does not need anything and I still add them here?
    "beta": 0.53,
    "delta": 0.99,
    "risk_aversion":      1.9,
    "interest_rate":      0.0203, 
    "interest_rate_debt": 0.1059, 
    "interest_rate_illiquid": 0, # habia puesto 1 y esto hacia que explotara porque
    "working_life": {
        "next_regime": {
            "last_working_age": retirement_age -1, #OJO NO ME MUESTRA ESTO EN EL DICC Y ES REPETITIVO (4 hmvg)
            "survival_probs": survival_probs,
        },
       # "next_regime_working": {
        #    "last_working_age": retirement_age -1,  # no entiendo la logica de por que hay que restar 1... lo hacen asi en el tiny sample
         #   "final_age":        dead_age - 1,         # 89.0
          #  "survival_probs": survival_probs,
        #},
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
        "borrowing_constraint": {
            "c0credit": 0.167,
            "c1credit": -0.002,
            "c2credit": 0.014,
            #"interest_rate": 0.0203,
            #"interest_rate_debt": 0.1059,
        },
        "household_size": {
            "a0kids": 0.003,
            "a1kids": 0.358,
            "a2kids": 0.508,
            "a0depadul": 0.00000459,
            "a1depadul": 0.452,
            "a2depadul": 0.438,
        },

    },
     "retirement": {
         "next_regime": {
            #"last_working_age": retirement_age -1,
            "survival_probs": survival_probs,
            #"final_age":        dead_age - 1,
        },
        #"next_regime_retirement": {
         #   "last_working_age": retirement_age -1,  
          #  "final_age":        dead_age - 1,
           # "survival_probs": survival_probs,
        #},
        #"deterministic": {
         #   "yret_cons":     6.0,
          #  "yret_agecoeff": -0.01,
        #},
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
        "borrowing_constraint": {
            "c0credit": 0.167,
            "c1credit": -0.002,
            "c2credit": 0.014,
            #"interest_rate": 0.0203,
            #"interest_rate_debt": 0.1059,
        },
        "household_size": {
            "a0kids": 0.003,
            "a1kids": 0.358,
            "a2kids": 0.508,
            "a0depadul": 0.00000459,
            "a1depadul": 0.452,
            "a2depadul": 0.438,
        },
    },
    "dead": {
        "utility": {
            "alpha": 0.5, # debe haber una forma de no tener que volver a escribir todos estos numeros
            #"delta": 0.96,
            #"interest_rate": 0.02,
            "mean_hhs": 2,
            "mean_hhy": 1000,
            #"risk_aversion": 2.0,
            }}
}