from pprint import pprint
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib as mpl



from errors.error_regimes_and_model import error3_model_naive
from lifecycle_model.parameters_and_grids import params_naive, age_grid
from config import SRC, BLD, SEED


n_agents = 10_000


error3_result = error3_model_naive.simulate(
    params=params_naive,
    initial_conditions={
        "regime": np.zeros(n_agents, dtype=int),
        "age": np.full(n_agents, float(age_grid.exact_values[0])), 
        "wealth": np.full(n_agents, (4709)), 
        "wealth_illiquid": np.full(n_agents, 83188),  
        "perm_income": np.zeros(n_agents),            
        "trans_income": np.zeros(n_agents),          
    },
    period_to_regime_to_V_arr=None,
    seed=SEED,
)


df = error3_result.to_dataframe(additional_targets="all")
df["age"] = df["age"].astype(int)
print(df)

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

output_path = SRC / BLD / "figures" / "error3_naive.png"
output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
fig.write_html(output_path)
fig.show()

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df_mean["age"], df_mean["earnings"]/1000,
        label="Income", color="firebrick", linewidth=3)

ax.plot(df_mean["age"], df_mean["total_consumption"]/1000,
        label="Total Consumption", color="royalblue", linewidth=3)

ax.plot(df_mean["age"], df_mean["wealth"]/1000,
        label="Liquid Assets", color="forestgreen", linewidth=3)

ax.plot(df_mean["age"], df_mean["wealth_illiquid"] / 10000,
        label="Illiquid Assets/10", color="goldenrod", linewidth=3)

ax.set_xlabel("Age")
ax.set_ylabel("Units (thousands)")

ax.legend(
    loc="upper left",
    bbox_to_anchor=(0.1, 0.9),
    frameon=True,
    edgecolor="black"
)

ax.grid(True, alpha=0.3)

# Guardar PNG para LaTeX
output_path = SRC / BLD / "figures" / "error3_naive.png"
output_path.parent.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")