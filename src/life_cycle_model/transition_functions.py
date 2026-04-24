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

def end_of_period_wealth(
    wealth: ContinuousState,
    investment_x: ContinuousAction,
) -> FloatND:
    """Liquid wealth after consumption, before interest — working life."""
    return wealth + investment_x

def next_wealth(
    end_of_period_wealth: FloatND,
    interest_rate: float,
    interest_rate_debt: float,
) -> ContinuousState:
   
    """Wealth at start of next period, with two interest rates.
 
    Savings earn interest_rate, debt accrues at interest_rate_debt (higher).
 
    Mirrors the Matlab (LifecycleSim_BackwardInduct.m):
        possibleX_ = max(nextX_/R, 0) + min(nextX_/R_CC, 0)
    where R is the savings rate and R_CC is the credit card rate.
   """
    return (
        jnp.maximum(end_of_period_wealth, 0) * (1 + interest_rate)
        + jnp.minimum(end_of_period_wealth, 0) * (1 + interest_rate_debt) 
    )


### Illiquid Wealth Z ###

def end_of_period_wealth_illiquid(
    wealth_illiquid: ContinuousState,
    investment_z: ContinuousAction,
) -> FloatND:
    """Illiquid wealth."""
    return wealth_illiquid + investment_z
    

def next_wealth_illiquid(
    end_of_period_wealth_illiquid: FloatND,
    interest_rate_illiquid: float,
) -> ContinuousState:
    return (end_of_period_wealth_illiquid) * (1 + interest_rate_illiquid)


### Regime transitions ###

def next_regime_working(
    age: float,
    period: Period,
    survival_probs: FloatND,
    last_working_age: float,
) -> FloatND:
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
    sp = survival_probs[period]
    return jnp.array([0.0, sp, 1 - sp])