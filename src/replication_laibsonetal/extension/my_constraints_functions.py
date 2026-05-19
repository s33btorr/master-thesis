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

def budget_constraint(
        total_consumption: FloatND,
    ) -> BoolND:
    """
    Consumption needs to be positive
    """
    return total_consumption > 0 

def borrowing_constraint(
    end_of_period_wealth: FloatND,
    credit_limit: float,
) -> BoolND:
    """
    Fixed borrowing constraint: end-of-period wealth >= -credit_limit.

    credit_limit >= 0 is the maximum borrowing allowed.
    Set credit_limit=0 for a no-borrowing constraint.

    TODO: replace with age-varying limit:
        credit_limit(age) = c0credit + c1credit*age + c2credit*age^2
    """
    return end_of_period_x_wealth>= - (credit_limit)

def illiquid_wealth_constraint(
    end_of_period_wealth_illiquid: FloatND,
) -> BoolND:
    """
    NO SE SI ESTA ES NECESARIA PORQUE YA DE POR SI NO PERMITO EN LA GRILLA QUE PILLE VALORES NEGATIVOS... de todas formas, no funciono....
    """
    return end_of_period_wealth_illiquid >= 0 # quiza agregar que ocurre cuando es falso esto


#El fragmento EV____(1:currix0-1,:,:,t) = -Inf; es fundamental. Evita que el agente muera con deudas. Al asignar una utilidad de −∞ a cualquier estado donde los activos sean negativos al final de la vida, el modelo obliga al agente a pagar todas sus deudas antes de que el ciclo termine.

def ponzi_constraint(
    end_of_period_wealth: FloatND,
    age: float,
) -> BoolND:
    return jnp.where(age == 90, end_of_period_wealth>=0, True)

def special_constraint(
    end_of_period_wealth: FloatND,
) -> BoolND:
    """
    I do not 100% understand why I need it...
    It does not work because it starts acumulating under the grid
    """
    return (end_of_period_x_wealth>= -5000) & (end_of_period_x_wealth<= 400_000)

def special_illiquid_constraint(
    end_of_period_wealth_illiquid: FloatND,
) -> BoolND:
    """
    I do not 100% understand it...
    It does not work because it starts acumulating under the grid
    """
    return end_of_period_wealth_illiquid <= 3_500_000

def minimum_consumption(
    consumption: FloatND,
) -> BoolND:
    return consumption>= 2_000
