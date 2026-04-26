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
        earnings: FloatND,
        wealth: ContinuousState,
        investment_z: ContinuousAction,
        investment_x: ContinuousAction,
        liquidation_cost: FloatND,   
    ) -> BoolND:

    liq_cost = liquidation_cost * jnp.minimum(investment_z, 0)
    consumption = earnings + wealth - investment_x - investment_z + liq_cost
    return consumption > 0 # I am not sure if it is necessary to put >0 or is enough with >=0, the problem is that in utility we do log(consumption) and if 0 is included, it is -inf

def borrowing_constraint(
    end_of_period_wealth: FloatND,
    #investment_x: ContinuousAction,
    age: float,
    c0credit: float,
    c1credit: float,
    c2credit: float,
    earnings: FloatND,
) -> BoolND:
    """
    Fixed borrowing constraint: end-of-period wealth >= -credit_limit.

    credit_limit >= 0 is the maximum borrowing allowed.
    Set credit_limit=0 for a no-borrowing constraint.

    TODO: replace with age-varying limit:
        credit_limit(age) = c0credit + c1credit*age + c2credit*age^2
    """
    credit_limit = c0credit + (c1credit*age) + (c2credit*(age**2)/100) # en el paper esta con el (age**2)/ 100, en el codigo esta sin el 100... PROBAR CON AMBOS
    return end_of_period_wealth >= - (earnings * credit_limit)

def illiquid_wealth_constraint(
    end_of_period_wealth_illiquid: FloatND,
) -> BoolND:
    """
    NO SE SI ESTA ES NECESARIA PORQUE YA DE POR SI NO PERMITO EN LA GRILLA QUE PILLE VALORES NEGATIVOS... de todas formas, no funciono....
    """
    return end_of_period_wealth_illiquid >= 0 # quiza agregar que ocurre cuando es falso esto


def consumtpion_positive(
    consumption: FloatND,
) -> BoolND:
    return consumption >= 0
# ME FALTA RESTRINGIR QUE EN EL ULTIMO PERIODO NO SE PUEDE MORIR CON DEUDAS

#El fragmento EV____(1:currix0-1,:,:,t) = -Inf; es fundamental. Evita que el agente muera con deudas. Al asignar una utilidad de −∞ a cualquier estado donde los activos sean negativos al final de la vida, el modelo obliga al agente a pagar todas sus deudas antes de que el ciclo termine.

def ponzi_constraint(
    end_of_period_wealth: FloatND,
    age: float,
) -> BoolND:
    return jnp.where(age == 90, end_of_period_wealth>0, True)

"""def liquid_wealth_constraint_last_period(
    end_of_period_wealth: FloatND,
) -> BoolND:
    
    return end_of_period_wealth >= 0 """