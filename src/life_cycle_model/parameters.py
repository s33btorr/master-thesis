from pprint import pprint

import jax.numpy as jnp
import pandas as pd
import plotly.express as px
from lcm import MarkovTransition
from lcm.typing import Period

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

# This does not go here #
SRC = Path().parent.resolve()
project_path = SRC / "brenda" / "master-thesis" / "src" / "replication_laibsonetal" / "data"
death_m_path = project_path / "DeathProbsE_M_Hist_TR2023.csv"
death_f_path = project_path / "DeathProbsE_F_Hist_TR2023.csv"
# this # 


def load_survival_probs(survival_document_paths_woman, survival_document_paths_man):
    
    """
    Uploads survival probabilities of man and woman for later use.

    """
    
    death_m = pd.read_csv(survival_document_paths_man, skiprows=2, header=None).iloc[:, 1:].values
    death_f = pd.read_csv(survival_document_paths_woman, skiprows=2, header=None).iloc[:, 1:].values

    deat_t = (death_m + death_f) / 2

    # Selección tipo MATLAB
    deat_t = deat_t[100:105, 20:91]

    # Media por columnas
    deat_t = np.mean(deat_t, axis=0)

    # Convertir a JAX
    deat_t = jnp.array(deat_t)

    # Probabilidades de supervivencia
    surv_t = 1.0 - deat_t

    # Último valor = 0
    survival_probs = surv_t.at[-1].set(0.0)

    return survival_probs

survival_probs = load_survival_probs(death_f_path, death_m_path)

