"""
simulate.py
===========
Función que corre el modelo de ciclo de vida y devuelve momentos simulados.

Equivalente a LifecycleSim.m en Matlab — recibe preferencias [beta, delta, rho],
corre el modelo con lcm y devuelve los momentos simulados.
"""

import numpy as np
import pandas as pd

from final.moments_calculation import compute_simulated_moments, MOMENT_NAMES


# =============================================================================
# Condiciones iniciales
# =============================================================================
# Equivalente a la inicialización en LifecycleSim_ForwardIter.m:
#   initialX = interp1(..., med_liq_wealth * initialY, ...)
#   initialZ = interp1(..., (med_total_wealth - med_liq_wealth) * initialY, ...)
# Con los valores de fs_params.med_liq_wealth y fs_params.med_total_wealth

INITIAL_WEALTH          = 4709.0    # med_liq_wealth * Ymean_(1)
INITIAL_WEALTH_ILLIQUID = 83188.0   # (med_total_wealth - med_liq_wealth) * Ymean_(1)



def simulate_moments(
    params: pd.DataFrame,
    *,
    model,
    params_base: dict,
    age_grid,
    n_agents: int,
    #seed: int,
) -> pd.Series:
    """
    Corre el modelo y devuelve momentos simulados.

    Equivalente a LifecycleSim.m en Matlab.

    Args:
        params      : pd.DataFrame con columnas ['value'] — formato optimagic.
                      Index: ['beta', 'delta', 'rho']
        model       : modelo lcm (model_exp o model_naive)
        params_base : dict con todos los parámetros del modelo excepto
                      discount_factor, risk_aversion, beta
        age_grid    : AgeGrid del modelo
        n_agents    : número de agentes simulados (setup.pop = 10000 en Matlab)
        alive_      : array de 71 valores con probabilidades de supervivencia
                      (fs_params.alive_ en Matlab)

    Returns:
        pd.Series con 16 momentos simulados, indexados por MOMENT_NAMES.
    """
    # extraer preferencias — orden: [beta, delta, rho] igual que Matlab
    beta  = float(params.loc["beta",  "value"])
    delta = float(params.loc["delta", "value"])
    rho   = float(params.loc["rho",   "value"])

    # actualizar params del modelo con las preferencias del optimizador
    # equivalente a: setup.prefs = [beta delta rho] en LifecycleSim.m
    full_params = {
        **params_base,
        "discount_factor": delta,   # delta en Matlab
        "risk_aversion":   rho,     # rho en Matlab
        "beta":            beta,    # beta en Matlab
    }

    # correr el modelo
    # equivalente a BackwardInduct + ForwardIter en Matlab
    result = model.simulate(
        params=full_params,
        initial_conditions={
            "regime":          np.zeros(n_agents, dtype=int),
            "age":             np.full(n_agents, float(age_grid.exact_values[0])),
            "wealth":          np.full(n_agents, INITIAL_WEALTH),
            "wealth_illiquid": np.full(n_agents, INITIAL_WEALTH_ILLIQUID),
            "perm_income":     np.zeros(n_agents),
            "trans_income":    np.zeros(n_agents),
        },
        period_to_regime_to_V_arr=None,
        #seed=seed,
    )

    df = result.to_dataframe(additional_targets="all")
    df["age"] = df["age"].astype(int)

    return compute_simulated_moments(df)
