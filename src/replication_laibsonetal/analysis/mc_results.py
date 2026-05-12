import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path

from lifecycle_model.regimes_and_model import model_naive, model_exp
from lifecycle_model.parameters_and_grids import params_naive, age_grid, params
from IPython.display import display

from analysis.simulate_distribution import dataframe_results
from config import RESULTS_PRESENT_BIASED, RESULTS_EXPONENTIAL, test_params_present_biased, test_params_exponential

from config import BLD, SRC



number_of_simulations=100
seed=9700
agents_in_simulation=10_000

comparison_with_paper = dataframe_results(
    n_simulations=number_of_simulations,
    master_seed=seed,
    params=test_params_present_biased,
    model=model_naive,
    params_base=params_naive,
    age_grid=age_grid,
    n_agents=agents_in_simulation,
    results_paper=RESULTS_PRESENT_BIASED
)

print (comparison_with_paper)

latex_table = (
        comparison_with_paper.style
        .format({
            "Own Simulation": "{:.3f}",
            "Benchmark": "{:.3f}",
            "Std. Error": "{:.6f}",
            "t-Statistic": "{:.3f}",
            "p-Value": "{:.3f}"
        })
        .to_latex(
            hrules=True,
            caption="Goodness-of-Fit: Present-Biased Model",
            label="tab:gof_presentbiased"
        )
    )

output_path = SRC / BLD / "tables" / "present_biased.tex"
output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(latex_table)


comparison_with_paper = dataframe_results(
    n_simulations=number_of_simulations,
    master_seed=seed,
    params=test_params_exponential,
    model=model_naive,
    params_base=params_naive,
    age_grid=age_grid,
    n_agents=agents_in_simulation,
    results_paper=RESULTS_EXPONENTIAL
)

print (comparison_with_paper)

latex_table = (
        comparison_with_paper.style
        .format({
            "Own Simulation": "{:.3f}",
            "Benchmark": "{:.3f}",
            "Std. Error": "{:.6f}",
            "t-Statistic": "{:.3f}",
            "p-Value": "{:.3f}"
        })
        .to_latex(
            hrules=True,
            caption="Goodness-of-Fit: Exponential Model",
            label="tab:gof_exponential"
        )
    )

output_path = SRC / BLD / "tables" / "exponential.tex"
output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(latex_table)