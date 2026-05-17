from pprint import pprint
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pandas as pd
import plotly.graph_objects as go


from extension.my_regimes_and_model import model
from extension.my_parameters_and_grids import params, age_grid
from final.moments_calculation import compute_simulated_moments

from config import BLD, SRC


n_agents = 10_000

half_agents = n_agents // 2
wealth_initial = np.concatenate([
    np.full(half_agents, 1000),
    np.full(n_agents - half_agents, 80000)
])

result = model.simulate(
    params=params, #log_level="debug", log_path="./debug/",
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
fig.add_trace(go.Scatter(x=df_mean["age"], y=df_mean["earnings"], name="Income", line=dict(color='firebrick', width=3)))
fig.add_trace(go.Scatter(x=df_mean["age"], y=df_mean["total_consumption"], name="Total Consumption", line=dict(color='royalblue', width=3)))
fig.add_trace(go.Scatter(x=df_mean["age"], y=df_mean["wealth"], name="Liquid Assets", line=dict(color='forestgreen', width=3)))
fig.add_trace(go.Scatter(x=df_mean["age"], y=df_mean["wealth_illiquid"]/10, name="Illiquid Assets/10", line=dict(color='goldenrod', width=3)))

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

#output_path = SRC / BLD / "figures" / "exponential.png"
#output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
#fig.write_html(output_path)
fig.show()

moments = compute_simulated_moments(df)
print(moments)

output_path = SRC / BLD / "data_frame.xlsx"

#df.to_excel(output_path, index=False)

# --- filtrar agentes de bajos ingresos ---
# calcular ingreso promedio por agente
avg_earnings_per_agent = df.groupby("subject_id")["earnings"].mean()

# quedarte con el percentil 10 inferior
p10 = avg_earnings_per_agent.quantile(0.10)
low_income_agents = avg_earnings_per_agent[avg_earnings_per_agent < p10].index

# filtrar el df
df_low = df[df["subject_id"].isin(low_income_agents)]

# calcular medias por edad para ese subgrupo
df_mean_low = df_low.groupby("age", as_index=False).mean(numeric_only=True)

fig = go.Figure()

# --- perfil promedio TODOS los agentes ---
fig.add_trace(go.Scatter(x=df_mean["age"], y=df_mean["earnings"], 
    name="Income (all)", line=dict(color='firebrick', width=3)))
fig.add_trace(go.Scatter(x=df_mean["age"], y=df_mean["wealth"], 
    name="Liquid Assets (all)", line=dict(color='forestgreen', width=3)))
fig.add_trace(go.Scatter(x=df_mean["age"], y=df_mean["wealth_illiquid"]/10, 
    name="Illiquid Assets/10 (all)", line=dict(color='goldenrod', width=3)))
fig.add_trace(go.Scatter(x=df_mean["age"], y=df_mean["total_consumption"], 
    name="Consumption (all)", line=dict(color='royalblue', width=3)))

# --- perfil promedio BAJOS INGRESOS ---
fig.add_trace(go.Scatter(x=df_mean_low["age"], y=df_mean_low["earnings"], 
    name="Income (low)", line=dict(color='firebrick', width=3, dash='dash')))
fig.add_trace(go.Scatter(x=df_mean_low["age"], y=df_mean_low["wealth"], 
    name="Liquid Assets (low)", line=dict(color='forestgreen', width=3, dash='dash')))
fig.add_trace(go.Scatter(x=df_mean_low["age"], y=df_mean_low["wealth_illiquid"]/10, 
    name="Illiquid Assets/10 (low)", line=dict(color='goldenrod', width=3, dash='dash')))
fig.add_trace(go.Scatter(x=df_mean_low["age"], y=df_mean_low["total_consumption"], 
    name="Consumption (low)", line=dict(color='royalblue', width=3, dash='dash')))

fig.update_layout(
    title="Lifecycle Profile — All vs Low Income Agents",
    xaxis_title="Age",
    yaxis_title="Units",
    template="plotly_white",
    legend=dict(x=0.1, y=0.9)
)

fig.show()
