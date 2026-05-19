from pprint import pprint
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter


from errors.error_regimes_and_model import error1_model, error2_model
from lifecycle_model.parameters_and_grids import params, age_grid
from config import DEBUG, SEED

n_agents = 10_000

error1_result = error1_model.simulate(
    params=params, log_level="debug", log_path=DEBUG,
    initial_conditions={
        "regime": np.zeros(n_agents, dtype=int),
        "age": np.full(n_agents, float(age_grid.exact_values[0])), 
        "wealth_x": np.full(n_agents, (4709)), 
        "wealth_z": np.full(n_agents, 83188),  
        "perm_income": np.zeros(n_agents),            
        "trans_income": np.zeros(n_agents),          
    },
    period_to_regime_to_V_arr=None,
    seed=SEED,
)