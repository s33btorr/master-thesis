from pprint import pprint

from pathlib import Path
import numpy as np
import jax.numpy as jnp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from lifecycle_model.regimes_and_model import model, model_exp, model_naive
from lifecycle_model.parameters_and_grids import params, age_grid, params_naive

SRC = Path('.').resolve().parent

n_agents = 10_000
result = model.simulate(
    params=params,
    initial_conditions={
        "regime": np.zeros(n_agents, dtype=int),
        "age": np.full(n_agents, float(age_grid.exact_values[0])),  
        "wealth": np.full(n_agents, (4709)), 
        "wealth_illiquid": np.full(n_agents, 83188),   
        "perm_income": np.zeros(n_agents),          
        "trans_income": np.zeros(n_agents),
    },
    period_to_regime_to_V_arr=None,
)

df = result.to_dataframe(additional_targets="all")
df["age"] = df["age"].astype(int)
print(df)

# =========================================================
# 1. Ingreso promedio lifetime por individuo
# =========================================================

income_lifetime = (
    df.groupby("subject_id")["earnings"]
    .mean()
    .reset_index(name="avg_lifetime_income")
)

# =========================================================
# 2. Percentiles
# =========================================================

p30 = income_lifetime["avg_lifetime_income"].quantile(0.3)
p70 = income_lifetime["avg_lifetime_income"].quantile(0.7)

# =========================================================
# 3. Crear grupos
# =========================================================

def income_group(x):
    if x <= p30:
        return "Bottom 30%"
    elif x >= p70:
        return "Top 30%"
    else:
        return "Middle 30%"

income_lifetime["income_group"] = (
    income_lifetime["avg_lifetime_income"]
    .apply(income_group)
)

# =========================================================
# 4. Merge al dataframe original
# =========================================================

df_groups = df.merge(
    income_lifetime[["subject_id", "income_group"]],
    on="subject_id",
    how="left"
)

# =========================================================
# 5. Crear figuras separadas
# =========================================================

groups = ["Bottom 30%", "Middle 30%", "Top 30%"]

for group in groups:

    temp = df_groups[df_groups["income_group"] == group]

    df_mean = (
        temp.groupby("age", as_index=False)
        .mean(numeric_only=True)
    )

    df_mean["consumption"] = (
        df_mean["earnings"]
        - df_mean["investment_x"]
        - df_mean["investment_z"]
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_mean["age"],
        y=df_mean["earnings"],
        name="Income",
        line=dict(color='firebrick', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df_mean["age"],
        y=df_mean["consumption"],
        name="Consumption",
        line=dict(color='royalblue', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df_mean["age"],
        y=df_mean["wealth"],
        name="Liquid Assets",
        line=dict(color='forestgreen', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df_mean["age"],
        y=df_mean["wealth_illiquid"] / 10,
        name="Illiquid Assets / 10",
        line=dict(color='goldenrod', width=3)
    ))

    fig.update_layout(
        title=f"Lifecycle Profile - {group}",
        xaxis_title="Age",
        template="plotly_white",
        legend=dict(
            x=0.1,
            y=0.9,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor="Black",
            borderwidth=1
        )
    )

    safe_group = group.replace(" ", "_").replace("%", "pct")
    output_path = SRC / "bld" / "figures" / f"lifecycle_exponential_{safe_group}.png"
    output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    fig.show()


result_naive = model_naive.simulate(
    params=params_naive, log_level="debug", log_path="./debug/",
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

# =========================================================
# 1. Ingreso promedio lifetime por individuo
# =========================================================

income_lifetime = (
    df_naive.groupby("subject_id")["earnings"]
    .mean()
    .reset_index(name="avg_lifetime_income")
)

# =========================================================
# 2. Percentiles
# =========================================================

p30 = income_lifetime["avg_lifetime_income"].quantile(0.3)
p70 = income_lifetime["avg_lifetime_income"].quantile(0.7)

# =========================================================
# 3. Crear grupos
# =========================================================

def income_group(x):
    if x <= p30:
        return "Bottom 30%"
    elif x >= p70:
        return "Top 30%"
    else:
        return "Middle 30%"

income_lifetime["income_group"] = (
    income_lifetime["avg_lifetime_income"]
    .apply(income_group)
)

# =========================================================
# 4. Merge al dataframe original
# =========================================================

df_groups = df_naive.merge(
    income_lifetime[["subject_id", "income_group"]],
    on="subject_id",
    how="left"
)

# =========================================================
# 5. Crear figuras separadas
# =========================================================

groups = ["Bottom 30%", "Middle 30%", "Top 30%"]

for group in groups:

    temp = df_groups[df_groups["income_group"] == group]

    df_mean = (
        temp.groupby("age", as_index=False)
        .mean(numeric_only=True)
    )

    df_mean["consumption"] = (
        df_mean["earnings"]
        - df_mean["investment_x"]
        - df_mean["investment_z"]
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_mean["age"],
        y=df_mean["earnings"],
        name="Income",
        line=dict(color='firebrick', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df_mean["age"],
        y=df_mean["consumption"],
        name="Consumption",
        line=dict(color='royalblue', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df_mean["age"],
        y=df_mean["wealth"],
        name="Liquid Assets",
        line=dict(color='forestgreen', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df_mean["age"],
        y=df_mean["wealth_illiquid"] / 10,
        name="Illiquid Assets / 10",
        line=dict(color='goldenrod', width=3)
    ))

    fig.update_layout(
        title=f"Lifecycle Profile - {group}",
        xaxis_title="Age",
        template="plotly_white",
        legend=dict(
            x=0.1,
            y=0.9,
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor="Black",
            borderwidth=1
        )
    )

    safe_group = group.replace(" ", "_").replace("%", "pct")
    output_path = SRC / "bld" / "figures" / f"lifecycle_present_biased_{safe_group}.png"
    output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    fig.show()