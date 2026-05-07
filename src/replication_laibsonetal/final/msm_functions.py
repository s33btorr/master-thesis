"""
msm.py
======
Función principal que corre el MSM con estimagic.

Equivalente a MSMfunction.m en Matlab.

Dos modelos:
    - Exponencial: beta=1 fijo, estima [delta, rho]
                   p0 = [1.00, 0.965, 2.60]  (EDFbatch_baseline.m, table3_exp)
    - Naive:       estima [beta, delta, rho]
                   p0 = [0.525, 0.992, 1.50] (EDFbatch_baseline.m, table3)
                   naif=1, betahat=1
"""

import numpy as np
import pandas as pd
import estimagic as em

from final.moments_calculation import load_empirical_moments, MOMENT_NAMES
from final.simulate_model import simulate_moments


# =============================================================================
# Valores iniciales — igual que EDFbatch_baseline.m
# =============================================================================

# Naive (table3): p0 = [0.525, 0.992, 1.50]
INIT_PREFS_NAIVE = {
    "beta":  0.525,
    "delta": 0.992,
    "rho":   1.50,
}

# Exponencial (table3_exp): p0 = [1.00, 0.965, 2.60]
INIT_PREFS_EXP = {
    "beta":  1.00,
    "delta": 0.965,
    "rho":   2.60,
}

MAX_DISCOUNT_FACTOR = 0.999


# =============================================================================
# Función principal
# =============================================================================

def run_msm(
    model,
    params_base: dict,
    age_grid,
    n_agents: int,
    data_moments: np.ndarray,
    vcv_secondstage: np.ndarray,
    model_type: str = "naive",
    weighting_method: int = 0,
    optimize_options: str = "scipy_neldermead",
) -> em.MomentsResult:
    """
    Corre el MSM y devuelve resultados con errores estándar.

    Equivalente a MSMfunction.m en Matlab.

    Args:
        model            : modelo lcm (model_naive o model_exp)
        params_base      : dict con parámetros del modelo excepto preferencias
        age_grid         : AgeGrid del modelo
        n_agents         : número de agentes (setup.pop = 10000 en Matlab)
        alive_           : array de 71 valores de supervivencia (fs_params.alive_)
        data_moments     : array de 16 momentos empíricos
        vcv_secondstage  : matriz 16x16 de covarianza de momentos empíricos
        model_type       : 'naive' o 'exp'
        weighting_method : 0=diagonal VCV (benchmark), 1=identidad, 2=VCV completa
        optimize_options : algoritmo de optimización

    Returns:
        MomentsResult de estimagic con optprefs, errores estándar, etc.
    """
    # --- momentos empíricos y covarianza ---
    empirical_moments, moments_cov = load_empirical_moments(
        data_moments, vcv_secondstage
    )

    # --- weighting matrix ---
    # estimagic builds the actual matrix internally from moments_cov.
    weighting_options = {
        0: "diagonal",
        1: "identity",
        2: "optimal",
    }
    if weighting_method not in weighting_options:
        raise ValueError(
            f"weighting method {weighting_method} no reconocido. Usa 0, 1 o 2."
        )
    weights = weighting_options[weighting_method]

    # --- parámetros iniciales y bounds ---
    # equivalente a setup.init_prefs y setup.matchbdr en Matlab
    if model_type == "naive":
        # estima [beta, delta, rho] — matchbdr = [1, 1, 1]
        # p0 = [0.525, 0.992, 1.50]
        start_params = pd.DataFrame(
            {
                "value":       [INIT_PREFS_NAIVE["beta"],
                                INIT_PREFS_NAIVE["delta"],
                                INIT_PREFS_NAIVE["rho"]],
                "lower_bound": [0.0,  0.0, 0.0],
                "upper_bound": [1.0,  MAX_DISCOUNT_FACTOR, np.inf],
            },
            index=["beta", "delta", "rho"],
        )
    elif model_type == "exp":
        # beta=1 fijo — matchbdr = [0, 1, 1]
        # p0 = [1.00, 0.965, 2.60]
        start_params = pd.DataFrame(
            {
                "value":       [INIT_PREFS_EXP["beta"],
                                INIT_PREFS_EXP["delta"],
                                INIT_PREFS_EXP["rho"]],
                "lower_bound": [1.0,  0.0, 0.0],   # beta fijo en 1
                "upper_bound": [1.0,  MAX_DISCOUNT_FACTOR, np.inf], # beta fijo en 1
            },
            index=["beta", "delta", "rho"],
        )
    else:
        raise ValueError(f"model_type '{model_type}' no reconocido. Usa 'naive' o 'exp'.")

    # --- correr MSM ---
    # equivalente a fminsearch(@MSMobj, ...) en Matlab
    res = em.estimate_msm(
        simulate_moments,
        empirical_moments,
        moments_cov,
        start_params,
        simulate_moments_kwargs={
            "model":       model,
            "params_base": params_base,
            "age_grid":    age_grid,
            "n_agents":    n_agents,
        },
        weights=weights,
        optimize_options=optimize_options,
    )

    return res
