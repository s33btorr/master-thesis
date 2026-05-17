from pprint import pprint
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from lifecycle_model.regimes_and_model import (
    model,
    model_exp,
    model_naive,
)

from lifecycle_model.parameters_and_grids import (
    params,
    age_grid,
    params_naive,
)

from config import BLD, SRC

# =========================================================
# 0. Simulación
# =========================================================

n_agents = 10_000

# mitad pobres, mitad ricos en wealth ilíquido
initial_illiquid = np.concatenate([
    np.full(n_agents // 2, 1000),
    np.full(n_agents // 2, 83188),
])

result = model.simulate(
    params=params,
    initial_conditions={
        "regime": np.zeros(n_agents, dtype=int),

        "age": np.full(
            n_agents,
            float(age_grid.exact_values[0])
        ),

        "wealth": np.full(n_agents, 4709),

        "wealth_illiquid": initial_illiquid,

        "perm_income": np.zeros(n_agents),

        "trans_income": np.zeros(n_agents),
    },
    period_to_regime_to_V_arr=None,
)

# =========================================================
# 1. Dataframe
# =========================================================

df = result.to_dataframe(additional_targets="all")

df["age"] = df["age"].astype(int)

print(df)

# =========================================================
# 2. Wealth inicial por individuo
# =========================================================

initial_illiquid_df = (
    df[df["period"] == 0]
    .groupby("subject_id")["wealth_illiquid"]
    .first()
    .reset_index(name="initial_illiquid_wealth")
)

# =========================================================
# 3. Crear grupos manuales
# =========================================================

def wealth_group(x):

    if np.isclose(x, 1000):
        return "Low Illiquid Wealth"

    elif np.isclose(x, 83188):
        return "High Illiquid Wealth"

initial_illiquid_df["wealth_group"] = (
    initial_illiquid_df["initial_illiquid_wealth"]
    .apply(wealth_group)
)

# =========================================================
# 4. Merge al dataframe principal
# =========================================================

df_groups = df.merge(
    initial_illiquid_df[
        ["subject_id", "wealth_group"]
    ],
    on="subject_id",
    how="left"
)

# =========================================================
# 5. Crear figuras separadas
# =========================================================

groups = [
    "Low Illiquid Wealth",
    "High Illiquid Wealth",
]

for group in groups:

    temp = df_groups[
        df_groups["wealth_group"] == group
    ]

    # promedio por edad
    df_mean = (
        temp.groupby("age", as_index=False)
        .mean(numeric_only=True)
    )

    # =====================================================
    # Figura
    # =====================================================

    fig = go.Figure()

    # Income
    fig.add_trace(go.Scatter(
        x=df_mean["age"],
        y=df_mean["earnings"],
        name="Income",
        line=dict(
            color="firebrick",
            width=3
        )
    ))

    # Consumption
    fig.add_trace(go.Scatter(
        x=df_mean["age"],
        y=df_mean["total_consumption"],
        name="Consumption",
        line=dict(
            color="royalblue",
            width=3
        )
    ))

    # Liquid wealth
    fig.add_trace(go.Scatter(
        x=df_mean["age"],
        y=df_mean["wealth"],
        name="Liquid Assets",
        line=dict(
            color="forestgreen",
            width=3
        )
    ))

    # Illiquid wealth
    fig.add_trace(go.Scatter(
        x=df_mean["age"],
        y=df_mean["wealth_illiquid"] / 10,
        name="Illiquid Assets / 10",
        line=dict(
            color="goldenrod",
            width=3
        )
    ))

    # =====================================================
    # Layout
    # =====================================================

    fig.update_layout(
        title=f"Lifecycle Profile - {group}",

        xaxis_title="Age",

        template="plotly_white",

        legend=dict(
            x=0.1,
            y=0.9,
            bgcolor="rgba(255,255,255,0.5)",
            bordercolor="Black",
            borderwidth=1,
        )
    )

    # =====================================================
    # Save
    # =====================================================

    safe_group = (
        group
        .replace(" ", "_")
        .replace("%", "pct")
    )

    output_path = (
        SRC
        / BLD
        / "figures"
        / f"lifecycle_exponential_{safe_group}.html"
    )

    output_path.resolve().parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.write_html(output_path)

    fig.show()