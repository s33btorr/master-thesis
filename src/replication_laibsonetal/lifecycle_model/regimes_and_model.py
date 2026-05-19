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
    end_of_period_x_wealth,
    next_wealth_x,
    end_of_period_z_wealth,
    next_wealth_z,
    next_regime_working,
    next_regime_retirement,
)

from lifecycle_model.constraints_functions import (
    borrowing_constraint, 
    z_wealth_constraint, 
    ponzi_constraint, 
    budget_constraint, 
    special_x_constraint,
    special_z_constraint
)

from lifecycle_model.parameters_and_grids import (
    wealth_z_grid, 
    wealth_x_grid, age_grid, 
    retirement_age, dead_age, 
    investment_x_grid, 
    investment_z_grid,
)
 
@categorical(ordered=False)
class RegimeId:
    working_life: int
    retirement: int
    dead: int


working_life = Regime(
    transition=MarkovTransition(next_regime_working),
    active=lambda age: age < retirement_age,
    states={
        "wealth_x": wealth_x_grid,
        "wealth_z": wealth_z_grid,
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
            mu=0,     # 0
            sigma=(0.045**0.5),  
            n_std=3,
        ),
    },
    state_transitions={
        "wealth_x": next_wealth_x,
        "wealth_z": next_wealth_z,
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
        "end_of_period_x_wealth": end_of_period_x_wealth,
        "end_of_period_z_wealth": end_of_period_z_wealth,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "z_wealth_constraint": z_wealth_constraint,
        "budget_constraint": budget_constraint,
        "special_x_constraint": special_x_constraint,
        "special_z_constraint": special_z_constraint,
        "ponzi_constraint": ponzi_constraint,
    },
)

retirement = Regime(
    transition=MarkovTransition(next_regime_retirement),
    active=lambda age: (age >= retirement_age) & (age < dead_age),
    states={
        "wealth_x": wealth_x_grid,
        "wealth_z": wealth_z_grid,
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
        "wealth_x": next_wealth_x,
        "wealth_z": next_wealth_z,
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
        "end_of_period_x_wealth": end_of_period_x_wealth,
        "end_of_period_z_wealth": end_of_period_z_wealth,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "z_wealth_constraint": z_wealth_constraint,
        "budget_constraint": budget_constraint,
        "special_x_constraint": special_x_constraint,
        "ponzi_constraint": ponzi_constraint,
        "special_z_constraint": special_z_constraint,
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
        "wealth_x": wealth_x_grid,
        "wealth_z": wealth_z_grid,
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


### Standard exponential model ###

working_exp = Regime(
    transition=MarkovTransition(next_regime_working),
    active=lambda age: age < retirement_age,
    states={
        "wealth_x": wealth_x_grid,
        "wealth_z": wealth_z_grid,
        "perm_income": lcm.shocks.ar1.Tauchen(
            n_points=3,
            gauss_hermite=False,
            rho=0.840,
            sigma=(0.057**0.5), 
            mu=0.0,
            n_std=1.5, 
        ),
        "trans_income": lcm.shocks.iid.Normal(
            n_points=5,
            gauss_hermite=False,
            mu=0,     # 0
            sigma=(0.045**0.5), 
        ),
    },
    state_transitions={
        "wealth_x": next_wealth_x,
        "wealth_z": next_wealth_z,
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
        "end_of_period_x_wealth": end_of_period_x_wealth,
        "end_of_period_z_wealth": end_of_period_z_wealth,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "z_wealth_constraint": z_wealth_constraint,
        "budget_constraint": budget_constraint,
        "special_x_constraint": special_x_constraint,
        "special_z_constraint": special_z_constraint,
        "ponzi_constraint": ponzi_constraint,
    },
)

retirement_exp = Regime(
    transition=MarkovTransition(next_regime_retirement),
    active=lambda age: (age >= retirement_age) & (age < dead_age),
    states={
        "wealth_x": wealth_x_grid,
        "wealth_z": wealth_z_grid,
        "perm_income": lcm.shocks.ar1.Tauchen(
            n_points=3,
            gauss_hermite=False,
            rho=0.840,
            sigma=(0.057**0.5), 
            mu=0.0,
            n_std=1.5,
        ),
        "trans_income": lcm.shocks.iid.Normal(
            n_points=5,
            gauss_hermite=False,
            mu=0,   
            sigma=(0.045**0.5),  
            n_std=3,
        ),
    },
    state_transitions={
        "wealth_x": next_wealth_x,
        "wealth_z": next_wealth_z,
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
        "end_of_period_x_wealth": end_of_period_x_wealth,
        "end_of_period_z_wealth": end_of_period_z_wealth,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "z_wealth_constraint": z_wealth_constraint,
        "budget_constraint": budget_constraint,
        "special_x_constraint": special_x_constraint,
        "special_z_constraint": special_z_constraint,
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
        "wealth_x": wealth_x_grid,
        "wealth_z": wealth_z_grid,
        "perm_income": lcm.shocks.ar1.Tauchen(
            n_points=3,
            gauss_hermite=False,
            rho=0.840,
            sigma=(0.057**0.5), 
            mu=0.0,
            n_std=1.5, 
        ),
        "trans_income": lcm.shocks.iid.Normal(
            n_points=5,
            gauss_hermite=False,
            mu=0,     # 0
            sigma=(0.045**0.5),  
            n_std=3,
        ),
    },
    state_transitions={
        "wealth_x": next_wealth_x,
        "wealth_z": next_wealth_z,
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
        "end_of_period_x_wealth": end_of_period_x_wealth,
        "end_of_period_z_wealth": end_of_period_z_wealth,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "z_wealth_constraint": z_wealth_constraint,
        "budget_constraint": budget_constraint,
        "special_x_constraint": special_x_constraint,
        "special_z_constraint": special_z_constraint,
        "ponzi_constraint": ponzi_constraint,

    },
)

retirement_naive = Regime(
    transition=MarkovTransition(next_regime_retirement),
    active=lambda age: (age >= retirement_age) & (age < dead_age),
    states={
        "wealth_x": wealth_x_grid,
        "wealth_z": wealth_z_grid,
        "perm_income": lcm.shocks.ar1.Tauchen(
            n_points=3,
            gauss_hermite=False,
            rho=0.840,
            sigma=(0.057**0.5), 
            mu=0.0,
            n_std=1.5, 
        ),
        "trans_income": lcm.shocks.iid.Normal(
            n_points=5,
            gauss_hermite=False,
            mu=0,     # 0
            sigma=(0.045**0.5),  
            n_std=3,
        ),
    },
    state_transitions={
        "wealth_x": next_wealth_x,
        "wealth_z": next_wealth_z,
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
        "end_of_period_x_wealth": end_of_period_x_wealth,
        "end_of_period_z_wealth": end_of_period_z_wealth,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "z_wealth_constraint": z_wealth_constraint,
        "budget_constraint": budget_constraint,
        "special_x_constraint": special_x_constraint,
        "special_z_constraint": special_z_constraint,
        "ponzi_constraint": ponzi_constraint,
    },
)

model_naive = Model(
    regimes={"working_life": working_naive, "retirement": retirement_naive, "dead": dead},
    ages=age_grid,
    regime_id_class=RegimeId,
    description="Lifecycle consumption-savings model with naive agents.",
)

