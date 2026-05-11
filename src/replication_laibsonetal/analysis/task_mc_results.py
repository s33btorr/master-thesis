import numpy as np
import pandas as pd
import scipy.stats as stats
from pathlib import Path

from lifecycle_model.regimes_and_model import model_naive, model_exp
from lifecycle_model.parameters_and_grids import params_naive, age_grid, params
from IPython.display import display

from simulate_distribution import get_simulation_distribution, dataframe_results
from config import RESULTS_PRESENT_BIASED, RESULTS_EXPONENTIAL, test_params_present_biased, test_params_exponential


def task_mc_simulation_and_results_to_latex_biased(
    produces=Path("bld/tables/present_biased.tex")
):
    comparison_with_paper = dataframe_results(
        n_simulations=100,
        master_seed=9700,
        params=test_params_present_biased,
        model=model_naive,
        params_base=params_naive,
        age_grid=age_grid,
        n_agents=10_000,
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

    produces.parent.mkdir(parents=True, exist_ok=True)
    produces.write_text(latex_table)


def task_mc_simulation_and_results_to_latex_exponential(
    produces=Path("bld/tables/exponential.tex")
):
    comparison_with_paper = dataframe_results(
        n_simulations=100,
        master_seed=9700,
        params=test_params_exponential,
        model=model_naive, # I know this is confusing, but it is the same model but with beta=1
        params_base=params,
        age_grid=age_grid,
        n_agents=10_000,
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
                caption="Goodness-of-Fit: Present-Biased Model",
                label="tab:gof_presentbiased"
            )
        )
    
    produces.parent.mkdir(parents=True, exist_ok=True)
    produces.write_text(latex_table)