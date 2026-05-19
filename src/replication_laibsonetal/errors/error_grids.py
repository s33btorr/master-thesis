from pprint import pprint
from lcm import PiecewiseLogSpacedGrid, PiecewiseLinSpacedGrid, Piece

from lcm import (
    AgeGrid,
    DiscreteGrid,
    LinSpacedGrid,
    LogSpacedGrid,
)


import numpy as np
import pandas as pd
from pathlib import Path


### Grids ###

error1_wealth_liquid_grid    = LinSpacedGrid(start=-45_000, stop=400_000, n_points=25) 

error2_wealth_liquid_grid    = LinSpacedGrid(start=-6_000, stop=400_000, n_points=25) 
error2_wealth_illiquid_grid  = LinSpacedGrid(start=0.1, stop=3_500_000, n_points=25)

