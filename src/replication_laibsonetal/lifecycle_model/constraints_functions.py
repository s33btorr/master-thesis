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
    end_of_period_x_wealth: FloatND,
    credit_limit: float,
) -> BoolND:
    """
    There is a credit limit depending on age.
    Total liquid wealth before interest rates must be bigger than the credit limit.
    """
    return end_of_period_x_wealth>= - (credit_limit)

def z_wealth_constraint(
    end_of_period_z_wealth: FloatND,
) -> BoolND:
    """
    Illiquid wealth cannot be negative.
    """
    return end_of_period_z_wealth >= 0 # quiza agregar que ocurre cuando es falso esto


#El fragmento EV____(1:currix0-1,:,:,t) = -Inf; es fundamental. Evita que el agente muera con deudas. Al asignar una utilidad de −∞ a cualquier estado donde los activos sean negativos al final de la vida, el modelo obliga al agente a pagar todas sus deudas antes de que el ciclo termine.

def ponzi_constraint(
    end_of_period_x_wealth: FloatND,
    age: float,
) -> BoolND:
    """
    Households cannot die with negative liquid wealth (debt).
    """
    return jnp.where(age == 90, end_of_period_x_wealth>=0, True)

def special_x_constraint(
    end_of_period_x_wealth: FloatND,
) -> BoolND:
    """
    Limits grid of state: liquid wealth.
    """
    return (end_of_period_x_wealth>= -5_000) & (end_of_period_x_wealth<= 400_000)

def special_z_constraint(
    end_of_period_z_wealth: FloatND,
) -> BoolND:
    """
    Limits grid of state: illiquid wealth.
    """
    return end_of_period_z_wealth <= 3_500_000