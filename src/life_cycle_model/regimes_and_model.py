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
)
from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)

### From here until the other hashtags, I would put in another python file (parameters and grids and then call it here...). But for now I will leave it here. ###
wealth_illiquid_grid = PiecewiseLogSpacedGrid(
    pieces=(
        Piece(interval="[1, 100_000)", n_points=35), #n_points=40 por alguna razon, esto con menos puntos no funciona........!!!!!!!!
        Piece(interval="[100_000, 3_500_000]", n_points=15), #n_points=34
    )
)

wealth_liquid_grid = PiecewiseLinSpacedGrid(
    pieces=(
        Piece(interval="[-45_000, 0)", n_points=22), #n_points=45
        Piece(interval="[0, 50_000)", n_points=25), #n_points=50
        Piece(interval="[50_000, 400_000]", n_points=31), #n_points=62
    )
)

age_grid = AgeGrid(start=20, stop=91, step="1Y")
retirement_age = 64
dead_age = 91 # pongo 91 porque vive hasta los 90, pero en este caso el periodo 91 es necesario ya que es el periodo de muerte

###. Until here    ####

working_life = Regime(
    transition=MarkovTransition(next_regime_working),
    active=lambda age: age < retirement_age,
    states={
        "wealth": LinSpacedGrid(start=0.1, stop=400000, n_points=50), 
        "wealth_illiquid": LinSpacedGrid(start=0, stop=3500000, n_points=50),
        #"wealth": LinSpacedGrid(start=-150, stop=400_000, n_points=50),
        #"wealth_illiquid": wealth_illiquid_grid,
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
            n_points=5,
            gauss_hermite=False,
            mu=0,     # 0
            sigma=(0.045**0.5),  # sqrt(ywork_varnu) from fs_params
            n_std=3,
        ),
    },
    state_transitions={
        "wealth": next_wealth,
        # perm_income and trans_income manage their own transitions
        "wealth_illiquid": next_wealth_illiquid,
    },
    actions={
        "investment_x": LinSpacedGrid(start=0.1, stop=150, n_points=100),
        "investment_z": LinSpacedGrid(start=-50, stop=50, n_points=50),
        #"investment_x":  LinSpacedGrid(start=-6_899, stop=6_899, n_points=100), # no funciona con mayor a 6,899
        #"investment_z": LinSpacedGrid(start=-1_000_000, stop=16_051, n_points=100), #no funciona con mayor a 16,051
    },
    functions={
        "utility": utility,
        "liquidation_cost": liquidation_cost,
        "household_size": household_size,
        "deterministic": deterministic_income,
        "consumption": consumption,
        "number_of_kids": number_of_kids,
        "number_of_depadul": number_of_depadul,
        "earnings": earnings,
        "end_of_period_wealth": end_of_period_wealth,
        "end_of_period_wealth_illiquid": end_of_period_wealth_illiquid,
        #"investment_illiquid_constrained": investment_illiquid_constrained
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "illiquid_wealth_constraint": illiquid_wealth_constraint,
    },
)

retirement = Regime(
    transition=MarkovTransition(next_regime_retirement),
    active=lambda age: (age >= retirement_age) & (age < dead_age),
    states={
        "wealth": LinSpacedGrid(start=0.1, stop=400000, n_points=50), 
        "wealth_illiquid": LinSpacedGrid(start=0, stop=3500000, n_points=50),
        #"wealth": LinSpacedGrid(start=-150, stop=400_000, n_points=50),
        #"wealth_illiquid": wealth_illiquid_grid,
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

        # Transitory income shock: iid Normal TAMBIEN VER SI ES IGUAL EN MATLAB IID SHOCKS
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
        # perm_income and trans_income manage their own transitions
        "wealth_illiquid": next_wealth_illiquid,
    },
     actions={
        "investment_x": LinSpacedGrid(start=0.1, stop=150, n_points=100),
        "investment_z": LinSpacedGrid(start=-50, stop=50, n_points=50),
        #"investment_x":  LinSpacedGrid(start=-6_899, stop=6_899, n_points=100),
        #"investment_z": LinSpacedGrid(start=-1_000_000, stop=16_051, n_points=100),
    },
    functions={
        "utility": utility,
        "liquidation_cost": liquidation_cost,
        "household_size": household_size,
        "deterministic": deterministic_income,
        "consumption": consumption,
        #"deterministic": deterministic_retirement_income,
        "number_of_kids": number_of_kids,
        "number_of_depadul": number_of_depadul,
        "earnings": earnings,
        "end_of_period_wealth": end_of_period_wealth,
        "end_of_period_wealth_illiquid": end_of_period_wealth_illiquid,
        #"investment_illiquid_constrained": investment_illiquid_constrained
    },
    constraints={
        "borrowing_constraint": borrowing_constraint,
        "illiquid_wealth_constraint": illiquid_wealth_constraint,
        #"ponzi_constraint": ponzi_constraint,
    },
)

dead = Regime(
    transition=None,
    active=lambda age: True,
    #active=lambda age: age >= dead_age,
    functions={
        "utility": beq_utility,
        "liquidation_cost": liquidation_cost,
        },
    states={
        "wealth": LinSpacedGrid(start=0.1, stop=400000, n_points=50), 
        "wealth_illiquid": LinSpacedGrid(start=0, stop=3500000, n_points=50),
        #"wealth": LinSpacedGrid(start=0.1, stop=400_000, n_points=50), # esta grilla hago que empiece en cero porque el wealth de dead no puede ser negatuvo
        #"wealth_illiquid": wealth_illiquid_grid,
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