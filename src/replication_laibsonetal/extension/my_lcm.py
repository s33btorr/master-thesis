from pprint import pprint
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from extension.my_regimes_and_model import model
from extension.my_parameters_and_grids import params, age_grid
from final.moments_calculation import compute_simulated_moments

from config import BLD, SRC, SEED


n_agents = 10_000

half_agents = n_agents // 2
wealth_initial = np.concatenate([
    np.full(half_agents, 1000),
    np.full(n_agents - half_agents, 80000)
])

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
    seed=9700,
)


df = result.to_dataframe(additional_targets="all")
df["age"] = df["age"].astype(int)
print(df)

# =========================================================
# Estadísticas de apoyo para la tesis
# =========================================================

print("\n--- Consumo mínimo observado ---")
min_row = df.loc[df["total_consumption"].idxmin()]
print(min_row[["age", "total_consumption", "earnings",
               "wealth", "wealth_illiquid"]])

print(f"\n--- Hogares con consumo < $12,000 en algún período ---")
print(f"Observaciones: {(df['total_consumption'] < 12_000).sum():,}")
print(f"Agentes únicos: "
      f"{df.loc[df['total_consumption'] < 12_000, 'subject_id'].nunique():,}")

df_low_episodes = df[df["total_consumption"] < 12_000]
print(df_low_episodes.groupby("age")["subject_id"].count())

# Episodios en edades laborales activas del modelo (21-60)
df_working = df_low_episodes[
    df_low_episodes["age"].between(21, 60)
]
print(f"Episodios en edades 21-60: {len(df_working)}")
print(f"Agentes únicos en edades 21-60: "
      f"{df_working['subject_id'].nunique()}")


fig = go.Figure()

# Todos los períodos de esos agentes, coloreados por si están bajo el umbral
THRESHOLD = 15_996

df_above = df[df["total_consumption"] >= THRESHOLD]
df_below = df[df["total_consumption"] < THRESHOLD]

fig.add_trace(go.Scatter(
    x=df_above["age"],
    y=df_above["total_consumption"],
    mode="markers",
    marker=dict(color="steelblue", size=3, opacity=0.3),
    name="Consumption ≥ $15,996"
))

fig.add_trace(go.Scatter(
    x=df_below["age"],
    y=df_below["total_consumption"],
    mode="markers",
    marker=dict(color="firebrick", size=5, opacity=0.8),
    name="Consumption < $15,996"
))

fig.add_hline(
    y=THRESHOLD,
    line_dash="dash",
    line_color="black",
    line_width=1.5,
    annotation_position="top right"
)

fig.update_layout(
    title="Individual consumption episodes",
    xaxis_title="Age",
    yaxis_title="Total consumption",
    template="plotly_white",
    legend=dict(x=0.7, y=0.95)
)

fig.show()

output_path = SRC / BLD / "figures" / "consumption_episodes.png"
output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
fig.write_html(output_path)