import numpy as np
import pandas as pd

from final.moments_calculation import compute_simulated_moments, MOMENT_NAMES

INITIAL_WEALTH          = 4709.0    # med_liq_wealth * Ymean_(1)
INITIAL_WEALTH_ILLIQUID = 83188.0   # (med_total_wealth - med_liq_wealth) * Ymean_(1)



def simulate_moments(
    params: pd.DataFrame,
    *,
    model,
    params_base: dict,
    age_grid,
    n_agents: int,
    seed: int,
) -> pd.Series:
    """
    Runs model with PyLCM and generates moments.

    Returns:
        pd.Series with 16 moments from the simulation.
    """
    beta  = float(params.loc["beta",  "value"])
    delta = float(params.loc["delta", "value"])
    rho   = float(params.loc["rho",   "value"])

   
    full_params = {
        **params_base,
        "discount_factor": delta,   
        "risk_aversion":   rho,     
        "beta":            beta, 
    }

   
    result = model.simulate(
        params=full_params,
        initial_conditions={
            "regime":          np.zeros(n_agents, dtype=int),
            "age":             np.full(n_agents, float(age_grid.exact_values[0])),
            "wealth_x":          np.full(n_agents, INITIAL_WEALTH),
            "wealth_z": np.full(n_agents, INITIAL_WEALTH_ILLIQUID),
            "perm_income":     np.zeros(n_agents),
            "trans_income":    np.zeros(n_agents),
        },
        period_to_regime_to_V_arr=None,
        seed=seed,
    )

    df = result.to_dataframe(additional_targets="all")
    df["age"] = df["age"].astype(int)

    return compute_simulated_moments(df)
