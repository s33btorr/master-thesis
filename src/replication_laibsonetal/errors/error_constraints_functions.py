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
    end_of_period_x_wealth: FloatND,
) -> BoolND:
    """
    Function that limits state grid: Liquid Wealth. 
    Used to obtain error when changing lowest value in wealth_x_grid.
    """
    return (end_of_period_x_wealth >= -45_000) & (end_of_period_x_wealth <= 400_000)

def error2_special_constraint(
    end_of_period_x_wealth: FloatND,
) -> BoolND:
    """
    Function that limits state grid: Liquid Wealth. 
    Used to obtain error when changing lowest value in wealth_z_grid.
    """
    return (end_of_period_x_wealth >= -6_000) & (end_of_period_x_wealth <= 400_000)