import numpy as np
import pandas as pd
import scipy.stats as stats
from final.simulate_model import simulate_moments


def get_simulation_distribution(n_simulations, master_seed, **kwargs):
    """
    Corre la simulación muchas veces para capturar la variabilidad estocástica.
    Usa un esquema de seed scheduling para que sea 100% replicable.
    """

    master_rng = np.random.default_rng(master_seed)
    seed_array = master_rng.integers(low=1, high=100_000, size=n_simulations)

    all_sims = []
    
    for i in range(n_simulations):
        # Es vital que NO fijes la semilla (seed) dentro de simulate_moments
        # para que cada iteración tenga shocks distintos.
        res = simulate_moments(seed=seed_array[i], **kwargs)
        all_sims.append(res)
        
    # Devolvemos un DataFrame donde cada fila es una simulación
    return pd.DataFrame(all_sims)

def dataframe_results(
        n_simulations,
        master_seed,
        params,
        model,
        params_base,
        age_grid,
        n_agents,
        results_paper,
):
        
    sim_dist = get_simulation_distribution(
        n_simulations=n_simulations,
        master_seed=master_seed,
        params=params,
        model=model,
        params_base=params_base,
        age_grid=age_grid,
        n_agents=n_agents
    )

    sim_mean = sim_dist.mean()
    sim_sd = sim_dist.std()

    t_stats = (sim_mean - results_paper) / sim_sd

    comparison_with_paper = pd.DataFrame({
        "Own Simulation": sim_mean,
        "Paper Simulation": results_paper,
        "Std. Error": sim_sd,
        "t-Statistic": t_stats,
        "p-Value": 2 * (1 - stats.norm.cdf(np.abs(t_stats)))
    })
    return comparison_with_paper