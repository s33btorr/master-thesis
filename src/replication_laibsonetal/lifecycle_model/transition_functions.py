from pprint import pprint

import jax.numpy as jnp
import pandas as pd
import plotly.express as px
from lcm import MarkovTransition
from lcm.typing import Period

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

### Liquid Wealth X ###

def end_of_period_x_wealth(
    wealth_x: ContinuousState,
    investment_x: ContinuousAction,
) -> FloatND:
    """Total liquid wealth after adding investment in liquid wealth."""
    return wealth_x + investment_x

def next_wealth_x(
    end_of_period_x_wealth: FloatND,
    interest_rate: float,
    interest_rate_debt: float,
) -> ContinuousState:
   
    """
    Liquid wealth at start of next period, with two interest rates.
    Savings earn interest_rate, debt accrues at interest_rate_debt (higher).
    """
    return (
        jnp.maximum(end_of_period_x_wealth, 0) * (1 + interest_rate)
        + jnp.minimum(end_of_period_x_wealth, 0) * (1 + interest_rate_debt) 
    )


### Illiquid Wealth Z ###

def end_of_period_z_wealth(
    wealth_z: ContinuousState,
    investment_z: ContinuousAction,
) -> FloatND:
    """Total illiquid wealth after adding investment in illiquid wealth."""
    return wealth_z + investment_z
    

def next_wealth_z(
    end_of_period_z_wealth: FloatND,
    interest_rate_illiquid: float,
) -> ContinuousState:
    """
    Iliquid wealth at start of next period, with interest rates for illiquid assets.
    """
    return (end_of_period_z_wealth) * (1 + interest_rate_illiquid)


### Regime transitions ###

def next_regime_working(
    age: float,
    period: Period,
    survival_probs: FloatND,
    last_working_age: float,
) -> FloatND:
    """
    Transition from working regime to death with probabilities coming from data.
    """
    sp = survival_probs[period]
    return jnp.where(
        age >= last_working_age,
        jnp.array([0.0, sp, 1 - sp]),   # → retirement con prob sp
        jnp.array([sp, 0.0, 1 - sp]),   # → working_life con prob sp
    )

def next_regime_retirement(
    period: Period,
    survival_probs: FloatND,
) -> FloatND:
    """
    Transition from retirement regime to death with probabilities coming from data.
    """
    sp = survival_probs[period]
    return jnp.array([0.0, sp, 1 - sp])