from pprint import pprint

import jax.numpy as jnp
import pandas as pd
import plotly.express as px
from lcm import MarkovTransition
from lcm.typing import Period
from lcm import PiecewiseLogSpacedGrid, PiecewiseLinSpacedGrid, Piece

from lcm.typing import (
    BoolND,
    ContinuousAction,
    ContinuousState,
    DiscreteAction,
    FloatND,
    ScalarInt,
)

def error1_special_constraint(
    end_of_period_wealth: FloatND,
) -> BoolND:
    """
    I do not 100% understand why I need it...
    It does not work because it starts acumulating under the grid
    """
    return (end_of_period_wealth >= -45_000) & (end_of_period_wealth <= 400_000)

def error2_special_constraint(
    end_of_period_wealth: FloatND,
) -> BoolND:
    """
    I do not 100% understand why I need it...
    It does not work because it starts acumulating under the grid
    """
    return (end_of_period_wealth >= -6_000) & (end_of_period_wealth <= 400_000)