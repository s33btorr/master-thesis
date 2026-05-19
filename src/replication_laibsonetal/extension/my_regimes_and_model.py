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

from extension.my_model_functions import (
    utility,
    total_consumption,
    liquidation_cost,
    household_size,
    deterministic_income,
    number_of_kids,
    number_of_depadul,
    earnings,
    beq_utility,
    average_earnings,
    exponential_H,
    beta_delta_H,
    average_income,
    credit_limit,
    consumption,
    )

from extension.my_transition_functions import (
    end_of_period_wealth,
    next_wealth,
    end_of_period_z_wealth,
    next_wealth_z,
    next_regime_working,
    next_regime_retirement,
)

from extension.my_constraints_functions import (
    borrowing_constraint, 
    z_wealth_constraint, 
    ponzi_constraint, 
    budget_constraint, 
    special_x_constraint,
    special_z_constraint,
    minimum_consumption,
)

from extension.my_parameters_and_grids import (
    wealth_z_grid, 
    wealth_x_grid, age_grid, 
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
        "wealth": wealth_x_grid,
        "wealth_illiquid": wealth_z_grid,
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
        "wealth_illiquid": next_wealth_z,
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
        "end_of_period_z_wealth": end_of_period_z_wealth,
        "average_income": average_income,
        "credit_limit": credit_limit,
        "consumption": consumption,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "z_wealth_constraint": z_wealth_constraint,
        "budget_constraint": budget_constraint,
        "special_x_constraint": special_x_constraint,
        "special_z_constraint": special_z_constraint,
        "ponzi_constraint": ponzi_constraint,
        #"minimum_consumption": minimum_consumption,
    },
)

retirement = Regime(
    transition=MarkovTransition(next_regime_retirement),
    active=lambda age: (age >= retirement_age) & (age < dead_age),
    states={
        "wealth": wealth_x_grid,
        "wealth_illiquid": wealth_z_grid,
        "perm_income": lcm.shocks.ar1.Tauchen(
            n_points=3,
            gauss_hermite=False,
            rho=0.840,
            sigma=(0.057**0.5), 
            mu=0.0,
            n_std=1.5,
        ),
        "trans_income": lcm.shocks.iid.Normal(
            n_points=3,
            gauss_hermite=False,
            mu=0, 
            sigma=(0.045**0.5),  
            n_std=3,
        ),
    },
    state_transitions={
        "wealth": next_wealth,
        "wealth_illiquid": next_wealth_z,
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
        "end_of_period_z_wealth": end_of_period_z_wealth,
        "average_income": average_income,
        "credit_limit": credit_limit,
        "consumption": consumption,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "z_wealth_constraint": z_wealth_constraint,
        "budget_constraint": budget_constraint,
        "special_x_constraint": special_x_constraint,
        "ponzi_constraint": ponzi_constraint,
        "special_z_constraint": special_z_constraint,
        #"minimum_consumption": minimum_consumption,
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
        "wealth": wealth_x_grid,
        "wealth_illiquid": wealth_z_grid,
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
)
