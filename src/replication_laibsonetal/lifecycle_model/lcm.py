from pprint import pprint
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter


from lifecycle_model.regimes_and_model import model, model_naive
from lifecycle_model.parameters_and_grids import params, age_grid, params_naive
from moments.moments_calculation import compute_simulated_moments

from config import BLD, SRC, SEED


n_agents = 10_000
result = model.simulate(
    params=params,
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

df = result.to_dataframe(additional_targets="all")
df["age"] = df["age"].astype(int)

df_mean = df.groupby("age", as_index=False).mean(numeric_only=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_mean["age"], y=df_mean["earnings"], name="Income", line=dict(color='firebrick', width=3)))
fig.add_trace(go.Scatter(x=df_mean["age"], y=df_mean["total_consumption"], name="Total Consumption", line=dict(color='royalblue', width=3)))
fig.add_trace(go.Scatter(x=df_mean["age"], y=df_mean["wealth_x"], name="Liquid Assets", line=dict(color='forestgreen', width=3)))
fig.add_trace(go.Scatter(x=df_mean["age"], y=df_mean["wealth_z"]/10, name="Illiquid Assets/10", line=dict(color='goldenrod', width=3)))

fig.update_layout(
    title="Average Lifecycle Profile for Exponential Estimate",
    xaxis_title="Age",
    yaxis_title="Units (thousands)",
    template="plotly_white", 
    legend=dict(
        x=0.1, y=0.9,
        bgcolor='rgba(255, 255, 255, 0.5)',
        bordercolor="Black",
        borderwidth=1
    )
)

output_path = SRC / BLD / "figures" / "exponential.png"
output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
fig.write_html(output_path)
fig.show()
print(df["perm_income"])
moments = compute_simulated_moments(df)
print(moments)

result_naive = model_naive.simulate(
    params=params_naive,
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

df_naive = result_naive.to_dataframe(additional_targets="all")
df_naive["age"] = df_naive["age"].astype(int)
print(df_naive)

df_mean_naive = df_naive.groupby("age", as_index=False).mean(numeric_only=True)
df_mean_naive["consumption"] = df_mean_naive["earnings"] - df_mean_naive["investment_x"] - df_mean_naive["investment_z"] #+ df_mean_naive["liquidation_cost"] tengo que ponerlo solo si investment z es negativo

fig_naive = go.Figure()
fig_naive.add_trace(go.Scatter(x=df_mean_naive["age"], y=df_mean_naive["earnings"], name="Income", line=dict(color='firebrick', width=3)))
fig_naive.add_trace(go.Scatter(x=df_mean_naive["age"], y=df_mean_naive["total_consumption"], name="Total Consumption", line=dict(color='royalblue', width=3)))
fig_naive.add_trace(go.Scatter(x=df_mean_naive["age"], y=df_mean_naive["wealth_x"], name="Liquid Assets", line=dict(color='forestgreen', width=3)))
fig_naive.add_trace(go.Scatter(x=df_mean_naive["age"], y=df_mean_naive["wealth_z"] / 10, name="Illiquid Assets/10", line=dict(color='goldenrod', width=3)))

fig_naive.update_layout(
    title="Average lifecycle profile for present-biased estimate",
    xaxis_title="Age",
    yaxis_title="Units (thousands)",
    template="plotly_white", # Fondo blanco como en la foto
    legend=dict(
        x=0.1, y=0.9,
        bgcolor='rgba(255, 255, 255, 0.5)',
        bordercolor="Black",
        borderwidth=1
    )
)

output_path = SRC / BLD / "figures" / "present_biased.png"
output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
fig_naive.write_html(output_path)
fig_naive.show()




