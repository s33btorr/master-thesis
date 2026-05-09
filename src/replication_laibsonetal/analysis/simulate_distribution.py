import numpy as np
import pandas as pd
import scipy.stats as stats
from final.simulate_model import simulate_moments
from final.moments_calculation import MOMENT_NAMES
from final.hardcoded_data import DATA_MOMENTS
from lifecycle_model.regimes_and_model import model_naive, model_exp
from lifecycle_model.parameters_and_grids import params_naive, age_grid, params


# Momentos extraídos de la columna (1) "Present Biased" de la Tabla 3
VALORES_SIMULADOS_PAPER = pd.Series([
    0.605, # % Visa 21-30
    0.585, # % Visa 31-40
    0.523, # % Visa 41-50
    0.475, # % Visa 51-60
    0.103, # mean Visa 21-30
    0.117, # mean Visa 31-40
    0.124, # mean Visa 41-50
    0.116, # mean Visa 51-60
    0.913, # wealth 21-30 | debt
    1.412, # wealth 31-40 | debt
    2.640, # wealth 41-50 | debt
    4.723, # wealth 51-60 | debt
    2.324, # wealth 21-30 | no debt
    3.248, # wealth 31-40 | no debt
    4.633, # wealth 41-50 | no debt
    7.475  # wealth 51-60 | no debt
], index=MOMENT_NAMES)

# Momentos extraídos de la columna (2) "Exponential" de la Tabla 3
VALORES_SIMULADOS_EXP = pd.Series([
    0.309, # % Visa 21-30
    0.287, # % Visa 31-40
    0.299, # % Visa 41-50
    0.257, # % Visa 51-60
    0.044, # mean Visa 21-30
    0.051, # mean Visa 31-40
    0.059, # mean Visa 41-50
    0.034, # mean Visa 51-60
    0.880, # wealth 21-30 | debt
    0.999, # wealth 31-40 | debt
    1.941, # wealth 41-50 | debt
    3.811, # wealth 51-60 | debt
    2.017, # wealth 21-30 | no debt
    2.850, # wealth 31-40 | no debt
    3.967, # wealth 41-50 | no debt
    5.358  # wealth 51-60 | no debt
], index=MOMENT_NAMES)
# valores iniciales naive — igual que Matlab
test_params = pd.DataFrame(
    {"value": [0.5305, 0.9891, 1.9355]},
    index=["beta", "delta", "rho"],
)

def get_simulation_distribution(n_simulations=1000, **kwargs):
    """
    Corre la simulación muchas veces para capturar la variabilidad estocástica.
    """
    all_sims = []
    
    for i in range(n_simulations):
        # Es vital que NO fijes la semilla (seed) dentro de simulate_moments
        # para que cada iteración tenga shocks distintos.
        res = simulate_moments(**kwargs)
        all_sims.append(res)
        
    # Devolvemos un DataFrame donde cada fila es una simulación
    return pd.DataFrame(all_sims)

# --- EJECUCIÓN ---
sim_dist = get_simulation_distribution(
    n_simulations=10,
    params=test_params,
    model=model_naive,
    params_base=params_naive,
    age_grid=age_grid,
    n_agents=10_000 # Nota: En Matlab usaban 10,000. Si tu PC aguanta, súbelo a 10,000 para reducir el ruido.
)

# 1. Calculamos la media y el error estándar de nuestras propias simulaciones
sim_mean = sim_dist.mean()
sim_sd = sim_dist.std()

# 2. El Test de Bondad de Ajuste (t-stat por momento)
# Comparamos: ¿Está el dato empírico (DATA_MOMENTS) muy lejos de mi media simulada?
t_stats = (sim_mean - DATA_MOMENTS) / sim_sd

comparison_with_empirics = pd.DataFrame({
    "Media_Simulada": sim_mean,
    "Empirico_Data": DATA_MOMENTS,
    "Std_Error_Sim": sim_sd,
    "T_Stat": t_stats,
    "P_Value": 2 * (1 - stats.norm.cdf(np.abs(t_stats))) # Probabilidad de ser iguales
})

comparison_with_paper = pd.DataFrame({
    "Mi_Simulacion": sim_mean,
    "Simulacion_Paper": VALORES_SIMULADOS_PAPER, # Aquí pones los del paper (ej. el 1.5)
    "Std_Error_Mio": sim_sd,
    "T_Stat": (sim_mean - VALORES_SIMULADOS_PAPER) / sim_sd,
    "P_Value": 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
})

print(comparison_with_empirics)
print (comparison_with_paper)

