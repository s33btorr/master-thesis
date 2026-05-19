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
    end_of_period_z_wealth,
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

from errors.error_constraints_functions import(error1_special_constraint, error2_special_constraint)

from errors.error_grids import(error1_wealth_liquid_grid, error2_wealth_illiquid_grid, error2_wealth_liquid_grid)
 
@categorical(ordered=False)
class RegimeId:
    working_life: int
    retirement: int
    dead: int

#### ERROR 1 #####

error1_working_life = Regime(
    transition=MarkovTransition(next_regime_working),
    active=lambda age: age < retirement_age,
    states={
        "wealth": error1_wealth_liquid_grid,
        "wealth_illiquid": wealth_illiquid_grid,
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
        "end_of_period_z_wealth": end_of_period_z_wealth,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "illiquid_wealth_constraint": illiquid_wealth_constraint,
        "budget_constraint": budget_constraint,
        "error1_special_constraint": error1_special_constraint,
        "special_illiquid_constraint": special_illiquid_constraint,
        "ponzi_constraint": ponzi_constraint,
    },
)

error1_retirement = Regime(
    transition=MarkovTransition(next_regime_retirement),
    active=lambda age: (age >= retirement_age) & (age < dead_age),
    states={
        "wealth": error1_wealth_liquid_grid,
        "wealth_illiquid": wealth_illiquid_grid,
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
        "end_of_period_z_wealth": end_of_period_z_wealth,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "illiquid_wealth_constraint": illiquid_wealth_constraint,
        "budget_constraint": budget_constraint,
        "error1_special_constraint": error1_special_constraint,
        "ponzi_constraint": ponzi_constraint,
        "special_illiquid_constraint": special_illiquid_constraint,
    },
)

error1_dead = Regime(
    transition=None,
    active=lambda age: True,
    functions={
        "utility": beq_utility,
        "liquidation_cost": liquidation_cost,
        "average_earnings": average_earnings,
        },
    states={
        "wealth": error1_wealth_liquid_grid,
        "wealth_illiquid": wealth_illiquid_grid,
    },
)

error1_model = Model(
    regimes={
        "working_life": error1_working_life,
        "retirement": error1_retirement,
        "dead": error1_dead,
    },
    ages=age_grid,
    regime_id_class=RegimeId,
    description="Lifecycle consumption-savings model.",
)


##### ERROR 2 #####

error2_working_life = Regime(
    transition=MarkovTransition(next_regime_working),
    active=lambda age: age < retirement_age,
    states={
        "wealth": error2_wealth_liquid_grid,
        "wealth_illiquid": error2_wealth_illiquid_grid,
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
        "end_of_period_z_wealth": end_of_period_z_wealth,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "illiquid_wealth_constraint": illiquid_wealth_constraint,
        "budget_constraint": budget_constraint,
        "error2_special_constraint": error2_special_constraint,
        "special_illiquid_constraint": special_illiquid_constraint,
        "ponzi_constraint": ponzi_constraint,
    },
)

error2_retirement = Regime(
    transition=MarkovTransition(next_regime_retirement),
    active=lambda age: (age >= retirement_age) & (age < dead_age),
    states={
        "wealth": error2_wealth_liquid_grid,
        "wealth_illiquid": error2_wealth_illiquid_grid,
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
        "end_of_period_z_wealth": end_of_period_z_wealth,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "illiquid_wealth_constraint": illiquid_wealth_constraint,
        "budget_constraint": budget_constraint,
        "error2_special_constraint": error2_special_constraint,
        "ponzi_constraint": ponzi_constraint,
        "special_illiquid_constraint": special_illiquid_constraint,
    },
)

error2_dead = Regime(
    transition=None,
    active=lambda age: True,
    functions={
        "utility": beq_utility,
        "liquidation_cost": liquidation_cost,
        "average_earnings": average_earnings,
        },
    states={
        "wealth": error2_wealth_liquid_grid,
        "wealth_illiquid": error2_wealth_illiquid_grid,
    },
)

error2_model = Model(
    regimes={
        "working_life": error2_working_life,
        "retirement": error2_retirement,
        "dead": error2_dead,
    },
    ages=age_grid,
    regime_id_class=RegimeId,
    description="Lifecycle consumption-savings model.",
)

####### Error 3 #######

error3_working_naive = Regime(
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
        "end_of_period_z_wealth": end_of_period_z_wealth,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "illiquid_wealth_constraint": illiquid_wealth_constraint,
        "budget_constraint": budget_constraint,
        "ponzi_constraint": ponzi_constraint,

    },
)

error3_retirement_naive = Regime(
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
        "end_of_period_z_wealth": end_of_period_z_wealth,
        "average_income": average_income,
        "credit_limit": credit_limit,
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "illiquid_wealth_constraint": illiquid_wealth_constraint,
        "budget_constraint": budget_constraint,
        "ponzi_constraint": ponzi_constraint,
    },
)

error3_dead = Regime(
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

error3_model_naive = Model(
    regimes={"working_life": error3_working_naive, "retirement": error3_retirement_naive, "dead": error3_dead},
    ages=age_grid,
    regime_id_class=RegimeId,
    description="Lifecycle consumption-savings model with naive agents.",
)

