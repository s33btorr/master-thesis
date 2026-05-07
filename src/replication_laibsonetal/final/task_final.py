"""Tasks running the results formatting (tables, figures)."""

import pandas as pd
import pytask

from replication_laibsonetal.config import BLD, SRC, TEMPLATE_GROUPS
from replication_laibsonetal.final.plot_template import plot_regression_by_age

"""
tasks.py
========
Punto de entrada principal. Equivalente a EDFbatch_baseline.m en Matlab.

Corre el MSM para el modelo naive y exponencial y guarda los resultados.

Uso:
    python tasks.py
"""

import numpy as np
import pandas as pd

from final.msm_functions import run_msm

from final.hardcoded_data import (
    DATA_MOMENTS,
    VCV_SECONDSTAGE,
    VCV_INCOME,
    VCV_CREDITLIM,
    VCV_DEMOGRAPHICS,
)


# =============================================================================
# IMPORTS DEL MODELO
# Reemplaza esto con tus imports reales del modelo
# =============================================================================
# from model import model_naive, model_exp, params_base, age_grid

# Ejemplo de cómo importar (ajusta al nombre de tu archivo):
# from model_pylcm import model as model_naive, params as params_base, age_grid


# =============================================================================
# PARÁMETROS DE SIMULACIÓN
# Equivalente a setup en EDFbatch_baseline.m
# =============================================================================

N_AGENTS = 10_000   # setup.pop = 10000

# Tasas de retorno — benchmark en Matlab:
# setup.returns = [1.0203, 1.0500, 1.1059]
INTEREST_RATE           = 0.0203   # R - 1
INTEREST_RATE_ILLIQUID  = 0.0500   # R_gamma - 1
INTEREST_RATE_DEBT      = 0.1059   # R_CC - 1


# =============================================================================
# DATOS DE PRIMERA Y SEGUNDA ETAPA
# Reemplaza con tus arrays reales
# =============================================================================

# alive_ — array de 71 valores (ages 20-90)
# Equivalente a fs_params.alive_ en Matlab
# Cargarlo de tus datos reales:
# alive_ = np.load("data/alive_.npy")
# alive_ = None  # reemplazar

# Momentos empíricos — array de 16 valores
# Equivalente a est_secondstage[:, 0] en Matlab
# data_moments = np.load("data/data_moments.npy")
data_moments = DATA_MOMENTS  # reemplazar

# Matriz de covarianza de momentos — 16x16
# Equivalente a VCV_secondstage en Matlab
# vcv_secondstage = np.load("data/vcv_secondstage.npy")
vcv_secondstage = VCV_SECONDSTAGE  # reemplazar


# =============================================================================
# TASK 1 — Simulación simple con parámetros óptimos ya estimados
# Equivalente a msm_estimation=0 en Matlab
# =============================================================================

def task_single_simulation(model, params_base, age_grid):
    """
    Corre una simulación simple con parámetros dados.
    Equivalente al bloque msm_estimation=0 en EDFbatch_baseline.m.
    """
    import numpy as np
    from simulate import simulate_moments

    # parámetros óptimos ya estimados (reemplazar con los reales)
    optprefs = pd.DataFrame(
        {"value": [0.525, 0.992, 1.50]},
        index=["beta", "delta", "rho"],
    )

    sim_moments = simulate_moments(
        optprefs,
        model=model,
        params_base=params_base,
        age_grid=age_grid,
        n_agents=N_AGENTS,
        alive_=alive_,
    )

    print("Simulated moments:")
    print(sim_moments)
    return sim_moments


# =============================================================================
# TASK 2 — Estimación MSM modelo naive
# Equivalente a MSMfunction con p0=[0.525, 0.992, 1.50], matchbdr=[1,1,1]
# =============================================================================

def task_msm_naive(model_naive, params_base, age_grid):
    """
    Corre el MSM para el modelo naive.
    Equivalente a run_name='table3' en EDFbatch_baseline.m.

    prefs estimadas: [beta, delta, rho]
    p0 = [0.525, 0.992, 1.50]
    """
    print("=" * 60)
    print("MSM — Modelo Naive")
    print("p0 = [beta=0.525, delta=0.992, rho=1.50]")
    print("=" * 60)

    res = run_msm(
        model=model_naive,
        params_base=params_base,
        age_grid=age_grid,
        n_agents=N_AGENTS,
        data_moments=data_moments,
        vcv_secondstage=vcv_secondstage,
        model_type="naive",
        weighting_method=0,         # benchmark: diagonal VCV
        optimize_options="scipy_neldermead",
    )

    print("\nResultados Naive:")
    print(res.summary())
    print(f"\noptprefs: {res.params['value'].to_dict()}")
    print(f"optq:     {res.res.value}")

    return res


# =============================================================================
# TASK 3 — Estimación MSM modelo exponencial
# Equivalente a run_name='table3_exp' en EDFbatch_baseline.m
# =============================================================================

def task_msm_exp(model_exp, params_base, age_grid):
    """
    Corre el MSM para el modelo exponencial.
    Equivalente a run_name='table3_exp' en EDFbatch_baseline.m.

    beta=1 fijo, estima [delta, rho]
    p0 = [1.00, 0.965, 2.60]
    """
    print("=" * 60)
    print("MSM — Modelo Exponencial (beta=1)")
    print("p0 = [beta=1.00, delta=0.965, rho=2.60]")
    print("=" * 60)

    res = run_msm(
        model=model_exp,
        params_base=params_base,
        age_grid=age_grid,
        n_agents=N_AGENTS,
        data_moments=data_moments,
        vcv_secondstage=vcv_secondstage,
        model_type="exp",
        weighting_method=0,
        optimize_options="scipy_neldermead",
    )

    print("\nResultados Exponencial:")
    print(res.summary())
    print(f"\noptprefs: {res.params['value'].to_dict()}")
    print(f"optq:     {res.criterion}")

    return res


# =============================================================================
# TASK 4 — Comparar momentos simulados vs empíricos
# =============================================================================

def task_compare_moments(res_naive, res_exp, data_moments):
    """
    Compara momentos simulados vs empíricos para ambos modelos.
    Equivalente a MSMout.optMoments en Matlab.
    """
    from moments import MOMENT_NAMES

    comparison = pd.DataFrame({
        "empirical":   data_moments,
        "naive":       res_naive.simulate_moments(res_naive.params),
        "exponential": res_exp.simulate_moments(res_exp.params),
    }, index=MOMENT_NAMES)

    print("\nMomentos simulados vs empíricos:")
    print(comparison.round(4))
    return comparison


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # --- imports del modelo (ajustar al nombre de tu archivo) ---
    from lifecycle_model.regimes_and_model import model_naive 
    from lifecycle_model.parameters_and_grids import params_naive, age_grid

    # --- task 2: MSM naive ---
    res_naive = task_msm_naive(model_naive, params_naive, age_grid)

    # --- task 3: MSM exponencial ---
    # res_exp = task_msm_exp(model_exp, params_base, age_grid)

    # --- task 4: comparar momentos ---
    # task_compare_moments(res_naive, res_exp, data_moments)

    print("Descomenta las tasks que quieras correr en __main__.")
    print("Asegúrate de cargar alive_, data_moments y vcv_secondstage primero.")
