from pprint import pprint

import jax.numpy as jnp
import pandas as pd
import plotly.express as px
from lcm import MarkovTransition
from lcm.typing import Period
from lcm import PiecewiseLogSpacedGrid, PiecewiseLinSpacedGrid, Piece

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    LogSpacedGrid,
    Model,
    Regime,
    categorical,
    SolveSimulateFunctionPair,

)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)

import lcm.shocks.ar1
import lcm.shocks.iid

from lifecycle_model.model_functions import (
    utility,
    total_consumption,
    liquidation_cost,
    household_size,
    deterministic_income,
    #deterministic_retirement_income,
    number_of_kids,
    number_of_depadul,
    earnings,
    beq_utility,
    average_earnings,
    exponential_H,
    beta_delta_H,
    average_income,
    credit_limit,
    )

from lifecycle_model.transition_functions import (
    end_of_period_wealth,
    next_wealth,
    end_of_period_wealth_illiquid,
    next_wealth_illiquid,
    next_regime_working,
    next_regime_retirement,
)
from lifecycle_model.constraints_functions import (
    borrowing_constraint, 
    illiquid_wealth_constraint, 
    ponzi_constraint, 
    budget_constraint, 
    special_constraint,
    special_illiquid_constraint
)

from lifecycle_model.parameters_and_grids import (
    wealth_illiquid_grid, 
    wealth_liquid_grid, age_grid, 
    retirement_age, dead_age, 
    investment_x_grid, 
    investment_z_grid,
)
 
# esto no se donde va en realidad #
@categorical(ordered=False)
class RegimeId:
    working_life: int
    retirement: int
    dead: int
###### ######


working_life = Regime(
    transition=MarkovTransition(next_regime_working),
    active=lambda age: age < retirement_age,
    states={
        "wealth": wealth_liquid_grid,
        "wealth_illiquid": wealth_illiquid_grid,
        # Permanent income shock: AR(1) via Rouwenhorst
        # Preferred over Tauchen for persistent processes (rho close to 1) EN MATLAB USAN TAUCHEN...
        "perm_income": lcm.shocks.ar1.Tauchen(
            n_points=3,
            gauss_hermite=False,
            rho=0.840,
            sigma=(0.057**0.5), # squared root of sigma e
            mu=0.0,
            n_std=1.5, #m esta en lifecycle sim pag 3
        ),
        #"perm_income": lcm.shocks.ar1.Rouwenhorst(
        #    n_points=3, # mirar si el numero de estados en matlab tb es 3
        #    rho=0.95,    # ywork_auto from fs_params
        #    sigma=0.1,  # sqrt(ywork_vareps) from fs_params
        #    mu=0,     # 0 — mean already captured in deterministic_income
        #),

        # Transitory income shock: iid Normal EN MATLAB CREO QUE ESTO ES MIL VECES MAS COMPLEJO... parece que sigue una grilla toda complicada...
        "trans_income": lcm.shocks.iid.Normal(
            n_points=3,
            gauss_hermite=False,
            mu=0,     # 0
            sigma=(0.045**0.5),  # sqrt(ywork_varnu) from fs_params
            n_std=3,
        ),
    },
    state_transitions={
        "wealth": next_wealth,
        "wealth_illiquid": next_wealth_illiquid,
    },
    actions={
        "investment_x": investment_x_grid,
        "investment_z": investment_z_grid,
    },
    functions={
        "utility": utility,
        "liquidation_cost": liquidation_cost,
        "household_size": household_size,
        "deterministic": deterministic_income,
        "total_consumption": total_consumption,
        "number_of_kids": number_of_kids,
        "number_of_depadul": number_of_depadul,
        "earnings": earnings,
        "end_of_period_wealth": end_of_period_wealth,
        "end_of_period_wealth_illiquid": end_of_period_wealth_illiquid,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "illiquid_wealth_constraint": illiquid_wealth_constraint,
        "budget_constraint": budget_constraint,
        "special_constraint": special_constraint,
        "special_illiquid_constraint": special_illiquid_constraint,
        "ponzi_constraint": ponzi_constraint,
    },
)

retirement = Regime(
    transition=MarkovTransition(next_regime_retirement),
    active=lambda age: (age >= retirement_age) & (age < dead_age),
    states={
        "wealth": wealth_liquid_grid,
        "wealth_illiquid": wealth_illiquid_grid,
        # Permanent income shock: AR(1) via Rouwenhorst
        # Preferred over Tauchen for persistent processes (rho close to 1) NO SE EN REALIDAD SI ES EL AR1 QUE HACEN EN MATLAB
        #"perm_income": lcm.shocks.ar1.Rouwenhorst(
        #    n_points=3, # mirar si el numero de estados en matlab tb es 3: SI
        #    rho=0.90,    # ywork_auto from fs_params
        #    sigma=0.05,  # sqrt(ywork_vareps) from fs_params
        #    mu=0,     # 0 — mean already captured in deterministic_income
        #),
        "perm_income": lcm.shocks.ar1.Tauchen(
            n_points=3,
            gauss_hermite=False,
            rho=0.840,
            sigma=(0.057**0.5), # squared root of sigma e
            mu=0.0,
            n_std=1.5, #m esta en lifecycle sim pag 3
        ),

        "trans_income": lcm.shocks.iid.Normal(
            n_points=3,
            gauss_hermite=False,
            mu=0,     # 0
            sigma=(0.045**0.5),  # sqrt(ywork_varnu) from fs_params
            n_std=3,
        ),
    },
    state_transitions={
        "wealth": next_wealth,
        "wealth_illiquid": next_wealth_illiquid,
    },
     actions={
        "investment_x": investment_x_grid,
        "investment_z": investment_z_grid,
    },
    functions={
        "utility": utility,
        "liquidation_cost": liquidation_cost,
        "household_size": household_size,
        "deterministic": deterministic_income,
        "total_consumption": total_consumption,
        #"deterministic": deterministic_retirement_income,
        "number_of_kids": number_of_kids,
        "number_of_depadul": number_of_depadul,
        "earnings": earnings,
        "end_of_period_wealth": end_of_period_wealth,
        "end_of_period_wealth_illiquid": end_of_period_wealth_illiquid,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "illiquid_wealth_constraint": illiquid_wealth_constraint,
        "budget_constraint": budget_constraint,
        "special_constraint": special_constraint,
        "ponzi_constraint": ponzi_constraint,
        "special_illiquid_constraint": special_illiquid_constraint,
    },
)

dead = Regime(
    transition=None,
    active=lambda age: True,
    functions={
        "utility": beq_utility,
        "liquidation_cost": liquidation_cost,
        "average_earnings": average_earnings,
        },
    states={
        "wealth": wealth_liquid_grid,
        "wealth_illiquid": wealth_illiquid_grid,
    },
)

model = Model(
    regimes={
        "working_life": working_life,
        "retirement": retirement,
        "dead": dead,
    },
    ages=age_grid,
    regime_id_class=RegimeId,
    description="Lifecycle consumption-savings model.",
    #enable_jit=False, # QUITAR CUANDO LO RESUELVA
)


### Standard exponential model ###

working_exp = Regime(
    transition=MarkovTransition(next_regime_working),
    active=lambda age: age < retirement_age,
    states={
        "wealth": wealth_liquid_grid,
        "wealth_illiquid": wealth_illiquid_grid,
        "perm_income": lcm.shocks.ar1.Tauchen(
            n_points=3,
            gauss_hermite=False,
            rho=0.840,
            sigma=(0.057**0.5), # squared root of sigma e
            mu=0.0,
            n_std=1.5, #m esta en lifecycle sim pag 3
        ),
        "trans_income": lcm.shocks.iid.Normal(
            n_points=5,
            gauss_hermite=False,
            mu=0,     # 0
            sigma=(0.045**0.5),  # sqrt(ywork_varnu) from fs_params
            n_std=3,
        ),
    },
    state_transitions={
        "wealth": next_wealth,
        "wealth_illiquid": next_wealth_illiquid,
    },
    actions={
        "investment_x": investment_x_grid,
        "investment_z": investment_z_grid,
    },
    functions={
        "utility": utility,
        "H": exponential_H,
        "liquidation_cost": liquidation_cost,
        "household_size": household_size,
        "deterministic": deterministic_income,
        "total_consumption": total_consumption,
        "number_of_kids": number_of_kids,
        "number_of_depadul": number_of_depadul,
        "earnings": earnings,
        "end_of_period_wealth": end_of_period_wealth,
        "end_of_period_wealth_illiquid": end_of_period_wealth_illiquid,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "illiquid_wealth_constraint": illiquid_wealth_constraint,
        "budget_constraint": budget_constraint,
        "special_constraint": special_constraint,
        "special_illiquid_constraint": special_illiquid_constraint,
    },
)

retirement_exp = Regime(
    transition=MarkovTransition(next_regime_retirement),
    active=lambda age: (age >= retirement_age) & (age < dead_age),
    states={
        "wealth": wealth_liquid_grid,
        "wealth_illiquid": wealth_illiquid_grid,
        "perm_income": lcm.shocks.ar1.Tauchen(
            n_points=3,
            gauss_hermite=False,
            rho=0.840,
            sigma=(0.057**0.5), # squared root of sigma e
            mu=0.0,
            n_std=1.5, #m esta en lifecycle sim pag 3
        ),
        "trans_income": lcm.shocks.iid.Normal(
            n_points=5,
            gauss_hermite=False,
            mu=0,     # 0
            sigma=(0.045**0.5),  # sqrt(ywork_varnu) from fs_params
            n_std=3,
        ),
    },
    state_transitions={
        "wealth": next_wealth,
        "wealth_illiquid": next_wealth_illiquid,
    },
     actions={
        "investment_x": investment_x_grid,
        "investment_z": investment_z_grid,
    },
    functions={
        "utility": utility,
        "H": exponential_H,
        "liquidation_cost": liquidation_cost,
        "household_size": household_size,
        "deterministic": deterministic_income,
        "total_consumption": total_consumption,
        "number_of_kids": number_of_kids,
        "number_of_depadul": number_of_depadul,
        "earnings": earnings,
        "end_of_period_wealth": end_of_period_wealth,
        "end_of_period_wealth_illiquid": end_of_period_wealth_illiquid,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "illiquid_wealth_constraint": illiquid_wealth_constraint,
        "budget_constraint": budget_constraint,
        "special_constraint": special_constraint,
        "special_illiquid_constraint": special_illiquid_constraint,
        "ponzi_constraint": ponzi_constraint,
    },
)

model_exp = Model(
    regimes={"working_life": working_exp, "retirement": retirement_exp, "dead": dead},
    ages=age_grid,
    regime_id_class=RegimeId,
    description="Lifecycle consumption-savings model with exponential agents.",
)


### Naive beta-delta model ###

working_naive = Regime(
    transition=MarkovTransition(next_regime_working),
    active=lambda age: age < retirement_age,
    states={
        "wealth": wealth_liquid_grid,
        "wealth_illiquid": wealth_illiquid_grid,
        "perm_income": lcm.shocks.ar1.Tauchen(
            n_points=3,
            gauss_hermite=False,
            rho=0.840,
            sigma=(0.057**0.5), # squared root of sigma e
            mu=0.0,
            n_std=1.5, #m esta en lifecycle sim pag 3
        ),
        "trans_income": lcm.shocks.iid.Normal(
            n_points=5,
            gauss_hermite=False,
            mu=0,     # 0
            sigma=(0.045**0.5),  # sqrt(ywork_varnu) from fs_params
            n_std=3,
        ),
    },
    state_transitions={
        "wealth": next_wealth,
        "wealth_illiquid": next_wealth_illiquid,
    },
    actions={
        "investment_x": investment_x_grid,
        "investment_z": investment_z_grid,
    },
    functions={
        "utility": utility,
        "H": SolveSimulateFunctionPair(
            solve=exponential_H,
            simulate=beta_delta_H,
        ),
        "liquidation_cost": liquidation_cost,
        "household_size": household_size,
        "deterministic": deterministic_income,
        "total_consumption": total_consumption,
        "number_of_kids": number_of_kids,
        "number_of_depadul": number_of_depadul,
        "earnings": earnings,
        "end_of_period_wealth": end_of_period_wealth,
        "end_of_period_wealth_illiquid": end_of_period_wealth_illiquid,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "illiquid_wealth_constraint": illiquid_wealth_constraint,
        "budget_constraint": budget_constraint,
        "special_constraint": special_constraint,
        "special_illiquid_constraint": special_illiquid_constraint,
    },
)

retirement_naive = Regime(
    transition=MarkovTransition(next_regime_retirement),
    active=lambda age: (age >= retirement_age) & (age < dead_age),
    states={
        "wealth": wealth_liquid_grid,
        "wealth_illiquid": wealth_illiquid_grid,
        "perm_income": lcm.shocks.ar1.Tauchen(
            n_points=3,
            gauss_hermite=False,
            rho=0.840,
            sigma=(0.057**0.5), # squared root of sigma e
            mu=0.0,
            n_std=1.5, #m esta en lifecycle sim pag 3
        ),
        "trans_income": lcm.shocks.iid.Normal(
            n_points=5,
            gauss_hermite=False,
            mu=0,     # 0
            sigma=(0.045**0.5),  # sqrt(ywork_varnu) from fs_params
            n_std=3,
        ),
    },
    state_transitions={
        "wealth": next_wealth,
        "wealth_illiquid": next_wealth_illiquid,
    },
     actions={
        "investment_x": investment_x_grid,
        "investment_z": investment_z_grid,
    },
    functions={
        "utility": utility,
        "H": SolveSimulateFunctionPair(
            solve=exponential_H,
            simulate=beta_delta_H,
        ),
        "liquidation_cost": liquidation_cost,
        "household_size": household_size,
        "deterministic": deterministic_income,
        "total_consumption": total_consumption,
        "number_of_kids": number_of_kids,
        "number_of_depadul": number_of_depadul,
        "earnings": earnings,
        "end_of_period_wealth": end_of_period_wealth,
        "end_of_period_wealth_illiquid": end_of_period_wealth_illiquid,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "illiquid_wealth_constraint": illiquid_wealth_constraint,
        "budget_constraint": budget_constraint,
        "special_constraint": special_constraint,
        "special_illiquid_constraint": special_illiquid_constraint,
        "ponzi_constraint": ponzi_constraint,
    },
)

model_naive = Model(
    regimes={"working_life": working_naive, "retirement": retirement_naive, "dead": dead},
    ages=age_grid,
    regime_id_class=RegimeId,
    description="Lifecycle consumption-savings model with naive agents.",
)

