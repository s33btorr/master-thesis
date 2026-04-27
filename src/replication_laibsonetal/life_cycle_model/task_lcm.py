from pprint import pprint
import numpy as np
import jax.numpy as jnp
import pandas as pd
import plotly.express as px
from lcm import MarkovTransition
from lcm.typing import Period
from lcm import PiecewiseLogSpacedGrid, PiecewiseLinSpacedGrid, Piece
import plotly.graph_objects as go

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    Model,
    Regime,
    categorical,
    SolveSimulateFunctionPair,
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)

from regimes_and_model import model, model_exp, model_naive
from parameters_and_grids import params, age_grid, params_naive

n_agents = 100
"""result = model.simulate(
    params=params, #log_level="debug", log_path="./debug/",
    initial_conditions={
        "regime": np.zeros(n_agents, dtype=int),
        "age": np.full(n_agents, float(age_grid.exact_values[0])),  # todos empiezan a los 20
        "wealth": np.full(n_agents, (4709)), #np.linspace(1, 20, n_agents), wealth = np.full(n_agents, (3.5894 - 0.1923) * earnings)
        "wealth_illiquid": np.full(n_agents, 83188),   # riqueza inicial varía de 1 a 20 np.linspace(1, 20, n_agents),  wealth = np.full(n_agents, (0.1923) * earnings)
        "perm_income": np.zeros(n_agents),              # media del AR(1)
        "trans_income": np.zeros(n_agents),             # media del shock iid
        #"enable_jit": np.False,
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
df_mean["consumption"] = df_mean["wealth"] + df_mean["earnings"] - df_mean["investment_z"] + df_mean["liquidation_cost"]

fig = px.line(
    df_mean,
    x="age",
    y="consumption",
    title="Consumption by Age",
    template="plotly_dark",
)

fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=df_mean["age"], y=df_mean["earnings"], name="Income", line=dict(color='firebrick', width=3)))
fig.add_trace(go.Scatter(x=df_mean["age"], y=df_mean["consumption"], name="Total Consumption", line=dict(color='royalblue', width=3)))
fig.add_trace(go.Scatter(x=df_mean["age"], y=df_mean["wealth"], name="Liquid Assets", line=dict(color='forestgreen', width=3)))
fig.add_trace(go.Scatter(x=df_mean["age"], y=df_mean["wealth_illiquid"] / 10, name="Illiquid Assets/10", line=dict(color='goldenrod', width=3)))

fig.update_layout(
    title="Average Lifecycle Profile",
    xaxis_title="Age",
    #yaxis_title="Units (x 10^4)",
    template="plotly_white", # Fondo blanco como en la foto
    legend=dict(
        x=0.1, y=0.9,
        bgcolor='rgba(255, 255, 255, 0.5)',
        bordercolor="Black",
        borderwidth=1
    )
)

fig.show()"""

### Exp and naive agents ###

result_exp = model_exp.simulate(
    params=params, #log_level="debug", log_path="./debug/",
    initial_conditions={
        "regime": np.zeros(n_agents, dtype=int),
        "age": np.full(n_agents, float(age_grid.exact_values[0])),  # todos empiezan a los 20
        "wealth": np.full(n_agents, (4709)), #np.linspace(1, 20, n_agents), wealth = np.full(n_agents, (3.5894 - 0.1923) * earnings)
        "wealth_illiquid": np.full(n_agents, 83188),   # riqueza inicial varía de 1 a 20 np.linspace(1, 20, n_agents),  wealth = np.full(n_agents, (0.1923) * earnings)
        "perm_income": np.zeros(n_agents),              # media del AR(1)
        "trans_income": np.zeros(n_agents),             # media del shock iid
        #"enable_jit": np.False,
    },
    period_to_regime_to_V_arr=None,
)

df_exp = result_exp.to_dataframe(additional_targets="all")
df_exp["age"] = df_exp["age"].astype(int)
print(df_exp)

df_mean_exp = df_exp.groupby("age", as_index=False).mean(numeric_only=True)
df_mean_exp["consumption"] = df_mean_exp["wealth"] + df_mean_exp["earnings"] - df_mean_exp["investment_x"]- df_mean_exp["investment_z"] + df_mean_exp["liquidation_cost"]

fig_exp = go.Figure()
fig_exp.add_trace(go.Scatter(x=df_mean_exp["age"], y=df_mean_exp["earnings"], name="Income", line=dict(color='firebrick', width=3)))
fig_exp.add_trace(go.Scatter(x=df_mean_exp["age"], y=df_mean_exp["consumption"], name="Total Consumption", line=dict(color='royalblue', width=3)))
fig_exp.add_trace(go.Scatter(x=df_mean_exp["age"], y=df_mean_exp["wealth"], name="Liquid Assets", line=dict(color='forestgreen', width=3)))
fig_exp.add_trace(go.Scatter(x=df_mean_exp["age"], y=df_mean_exp["wealth_illiquid"] / 10, name="Illiquid Assets/10", line=dict(color='goldenrod', width=3)))

fig_exp.update_layout(
    title="Average Lifecycle Profile",
    xaxis_title="Age",
    #yaxis_title="Units (x 10^4)",
    template="plotly_white", # Fondo blanco como en la foto
    legend=dict(
        x=0.1, y=0.9,
        bgcolor='rgba(255, 255, 255, 0.5)',
        bordercolor="Black",
        borderwidth=1
    )
)

fig_exp.show()

"""fig_exp = px.line(
    df_mean_exp,
    x="age",
    y="consumption",
    title="Consumption by Age",
    template="plotly_dark",
)

fig_exp.show()"""

"""result_naive = model_naive.simulate(
    params=params_naive, #log_level="debug", log_path="./debug/",
    initial_conditions={
        "regime": np.zeros(n_agents, dtype=int),
        "age": np.full(n_agents, float(age_grid.exact_values[0])),  # todos empiezan a los 20
        "wealth": np.full(n_agents, (4709)), #np.linspace(1, 20, n_agents), wealth = np.full(n_agents, (3.5894 - 0.1923) * earnings)
        "wealth_illiquid": np.full(n_agents, 83188),   # riqueza inicial varía de 1 a 20 np.linspace(1, 20, n_agents),  wealth = np.full(n_agents, (0.1923) * earnings)
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
df_mean_naive["consumption"] = df_mean_naive["wealth"] + df_mean_naive["earnings"] - df_mean_naive["investment_x"] - df_mean_naive["investment_z"] + df_mean_naive["liquidation_cost"]

fig_naive = go.Figure()
fig_naive.add_trace(go.Scatter(x=df_mean_naive["age"], y=df_mean_naive["earnings"], name="Income", line=dict(color='firebrick', width=3)))
fig_naive.add_trace(go.Scatter(x=df_mean_naive["age"], y=df_mean_naive["consumption"], name="Total Consumption", line=dict(color='royalblue', width=3)))
fig_naive.add_trace(go.Scatter(x=df_mean_naive["age"], y=df_mean_naive["wealth"], name="Liquid Assets", line=dict(color='forestgreen', width=3)))
fig_naive.add_trace(go.Scatter(x=df_mean_naive["age"], y=df_mean_naive["wealth_illiquid"] / 10, name="Illiquid Assets/10", line=dict(color='goldenrod', width=3)))

fig_naive.update_layout(
    title="Average Lifecycle Profile",
    xaxis_title="Age",
    #yaxis_title="Units (x 10^4)",
    template="plotly_white", # Fondo blanco como en la foto
    legend=dict(
        x=0.1, y=0.9,
        bgcolor='rgba(255, 255, 255, 0.5)',
        bordercolor="Black",
        borderwidth=1
    )
)

fig_naive.show()"""

"""fig_naive = px.line(
    df_mean_naive,
    x="age",
    y="consumption",
    title="Consumption by Age",
    template="plotly_dark",
)

fig_naive.show()"""