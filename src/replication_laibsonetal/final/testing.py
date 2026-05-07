import pandas as pd
from final.simulate_model import simulate_moments
from final.moments_calculation import MOMENT_NAMES
from final.hardcoded_data import DATA_MOMENTS
from lifecycle_model.regimes_and_model import model_naive 
from lifecycle_model.parameters_and_grids import params_naive, age_grid

# valores iniciales naive — igual que Matlab
test_params = pd.DataFrame(
    {"value": [0.525, 0.992, 1.50]},
    index=["beta", "delta", "rho"],
)

# simular
sim = simulate_moments(
    test_params,
    model=model_naive,
    params_base=params_naive,
    age_grid=age_grid,
    n_agents=1_000,
)

# comparar
import pandas as pd
comparison = pd.DataFrame({
    "simulado": sim.values,
    "empirico": DATA_MOMENTS,
    "diferencia": sim.values - DATA_MOMENTS,
}, index=MOMENT_NAMES)

print(comparison.round(4))
