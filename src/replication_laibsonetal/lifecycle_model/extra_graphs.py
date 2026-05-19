from pprint import pprint
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pandas as pd
import plotly.graph_objects as go


from lifecycle_model.regimes_and_model import model, model_exp, model_naive
from lifecycle_model.parameters_and_grids import params, age_grid, params_naive

from config import BLD, SRC


n_agents = 10_000
result = model.simulate(
    params=params, log_level="debug", log_path="./debug/",
    initial_conditions={
        "regime": np.zeros(n_agents, dtype=int),
        "age": np.full(n_agents, float(age_grid.exact_values[0])), 
        "wealth": np.full(n_agents, (4709)), 
        "wealth_z": np.full(n_agents, 83188),  
        "perm_income": np.zeros(n_agents),            
        "trans_income": np.zeros(n_agents),          
    },
    period_to_regime_to_V_arr=None,
)

df = result.to_dataframe(additional_targets="all")
df["age"] = df["age"].astype(int)
print(df)

bins = [21, 31, 41, 51, 61]
labels = ["21-30", "31-40", "41-50", "51-60"]

df_age = df.copy()
df_age["age_group"] = pd.cut(df_age["age"], bins=bins, right=False, labels=labels)

df_mean = (
    df_age
    .groupby(["age_group", "age"], as_index=False)
    .mean(numeric_only=True)
)

summary = (
    df_age
    .groupby("age_group")["wealth"]
    .agg(["min", "max"])
    .round(1)
)

df_mean = df.groupby("age", as_index=False).mean(numeric_only=True)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_mean["age"],
    y=df_mean["utility"],
    name="Utility",
    line=dict(color='royalblue', width=3)
))

fig.update_layout(
    title="Average Lifecycle Utility Profiles",
    xaxis_title="Age",
    yaxis_title="Utility",
    template="plotly_white",
    legend=dict(
        x=0.1, y=0.9,
        bgcolor='rgba(255, 255, 255, 0.5)',
        bordercolor="Black",
        borderwidth=1
    )
)

fig.show()

result_naive = model_naive.simulate(
    params=params_naive, log_level="debug", log_path="./debug/",
    initial_conditions={
        "regime": np.zeros(n_agents, dtype=int),
        "age": np.full(n_agents, float(age_grid.exact_values[0])),  # todos empiezan a los 20
        "wealth": np.full(n_agents, (4709)), #np.linspace(1, 20, n_agents), wealth = np.full(n_agents, (3.5894 - 0.1923) * earnings)
        "wealth_z": np.full(n_agents, 83188),   # riqueza inicial varía de 1 a 20 np.linspace(1, 20, n_agents),  wealth = np.full(n_agents, (0.1923) * earnings)
        "perm_income": np.zeros(n_agents),              # media del AR(1)
        "trans_income": np.zeros(n_agents),             # media del shock iid
        #"enable_jit": np.False,
    },
    period_to_regime_to_V_arr=None,
)

df_naive = result_naive.to_dataframe(additional_targets="all")
df_naive["age"] = df_naive["age"].astype(int)
print(df_naive)

df_mean_naive = df_naive.groupby("age", as_index=False).mean(numeric_only=True)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_mean_naive["age"],
    y=df_mean_naive["utility"],
    name="Utility",
    line=dict(color='royalblue', width=3)
))


fig.update_layout(
    title="Average Lifecycle Utility Profiles",
    xaxis_title="Age",
    yaxis_title="Utility",
    template="plotly_white",
    legend=dict(
        x=0.1, y=0.9,
        bgcolor='rgba(255, 255, 255, 0.5)',
        bordercolor="Black",
        borderwidth=1
    )
)

fig.show()