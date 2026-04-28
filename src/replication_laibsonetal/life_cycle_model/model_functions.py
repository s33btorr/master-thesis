from pprint import pprint
import numpy as np
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

### Auxiliary functions ###

def household_size(
    age: float,
    a0kids: float,
    a1kids: float,
    a2kids: float,
    a0depadul: float,
    a1depadul: float,
    a2depadul: float,
) -> FloatND:
    """
    Effective household size as a function of age.
 
    Mirrors the Matlab specification (LifecycleSim.m, baseline sqrtscale=False):
        spouse   = 2  (always)
        kids     = a0kids * exp(a1kids*age - a2kids*age^2)
        depadul  = a0depadul * exp(a1depadul*age - a2depadul*age^2)
        hhs      = 1*spouse + 0.4*kids + 1*depadul
 
    Parameters come from first-stage demographics regression (IPUMS-USA).
    Loaded in main.m from: IPUMS/output/{educ}/nlsur_nkids_ndepad_est_{educ}.xlsx
 
    TODO: add mortality correction to spouse term when mortalityn is implemented.
    """
    spouse  = 2.0
    kids    = a0kids * jnp.exp(a1kids * age - a2kids * age ** 2)
    depadul = a0depadul * jnp.exp(a1depadul * age - a2depadul * age ** 2)
    return 1.0 * spouse + 0.4 * kids + 1.0 * depadul  # if sqrtscale true, then is 1, 1, 1, instead of 0.4

def number_of_kids(
        age: float,
        a0kids: float,
        a1kids: float,
        a2kids: float,
) -> FloatND:
    return a0kids * jnp.exp((a1kids * age) - (a2kids * (age ** 2)/100))

def number_of_depadul(
        age: float,
        a0depadul: float,
        a1depadul: float,
        a2depadul: float,
) -> FloatND:
    return a0depadul * jnp.exp((a1depadul * age) - (a2depadul * (age ** 2)/100))

def liquidation_cost(
    age: float,
    #zilliq: float,
) -> FloatND:
    return 0.5 / (1 + jnp.exp((age - 50) / 10)) # tengo que incluir activos iliquidos completamente y que estos cuestan siempre 2 (200%) liquidarlos




### Income ###

def deterministic_income(
    age: float,
    ywork_cons: float,
    ywork_agecoeff: float,
    ywork_age2coeff: float,
    ywork_age3coeff: float,
    ywork_kidscoeff: float,
    ywork_spousecoeff: float,
    ywork_depadulcoeff: float,
    number_of_kids: FloatND,
    number_of_depadul: FloatND,
) -> FloatND:
    """
    Deterministic component of log income (cubic age polynomial).

    Mirrors the Matlab specification:
        ymean = cons + age*coeff + (age^2/100)*coeff2 + (age^3/10000)*coeff3

    Note: spouse, kids, depadul terms omitted for simplicity.
    They can be added as additional float arguments once the profiles are fixed.
    """
    spouse = 2 # ponen que es siempre 2...
    return (
        ywork_cons
        + ywork_kidscoeff * number_of_kids
        + ywork_spousecoeff * spouse
        + ywork_depadulcoeff * number_of_depadul
        + ywork_agecoeff * age
        + ywork_age2coeff * (age ** 2) / 100
        + ywork_age3coeff * (age ** 3) / 10000     
    ) 
# CREAR FUNCION QUE GENERE KIDS, DEPADUL Y SPOUSE POR SEPARADO.

def deterministic_retirement_income(
    age: float,
    yret_cons: float,
    yret_agecoeff: float,
) -> FloatND:
    """
    Deterministic component of log income (cubic age polynomial).

    Mirrors the Matlab specification:
        ymean = cons + age*coeff + (age^2/100)*coeff2 + (age^3/10000)*coeff3

    Note: spouse, kids, depadul terms omitted for simplicity.
    They can be added as additional float arguments once the profiles are fixed.
    """
    return (
        yret_cons
        + yret_agecoeff * age
    )

def earnings(
    perm_income: ContinuousState,
    trans_income: ContinuousState,
    deterministic: FloatND,
) -> FloatND:
    """
    Total labor earnings in levels.

    Log income = deterministic_mean + permanent_AR1_shock + transitory_iid_shock.
    Exponentiate to get level earnings.
    """
    return jnp.exp(deterministic + 0.5*(perm_income + trans_income)) # asi lo entiendo yo del codigo...



#### UTILITY ####

def utility(
    earnings: FloatND,
    investment_z: ContinuousAction,
    investment_x: ContinuousAction,
    liquidation_cost: FloatND,   
    wealth_illiquid: ContinuousState,
    household_size: FloatND,
    risk_aversion: float,
) -> FloatND:
  
    """CRRA utility scaled by household size.
 
    Mirrors the Matlab baseline (sqrtscale=False, LifecycleSim_BackwardInduct.m):
        U = hhs * (c/hhs)^(1-rho) / (1-rho)
 
    TODO: add sqrtscale variant (divides by sqrt(hhs) instead of hhs).
    TODO: add mortality-weighted discounting when mortalityn is implemented.
    TODO: add bequest utility at terminal age.
    """
    liq_cost = liquidation_cost * jnp.minimum(investment_z, 0)
    consumption = earnings - investment_x - investment_z + liq_cost
    total_consumption = consumption + (0.05 * wealth_illiquid) # 0.05 sale en el paper
    c_per_hh = total_consumption / household_size
    numerador = household_size * ((c_per_hh**(1 - risk_aversion)) - 1) # hice todo esto porque me estaba dando valores raros para ver si se acomoda. Asi lo hacen en matlab
    denominador = 1 - risk_aversion
    return jnp.where(risk_aversion == 1, household_size * jnp.log(c_per_hh), numerador / denominador)

def beq_utility(
        mean_hhs: float,
        mean_hhy: float,
        #earnings: FloatND,
        risk_aversion: float,
        wealth: ContinuousState,
        wealth_illiquid: ContinuousState,
        liquidation_cost: FloatND,
        interest_rate: float,
        alpha: float,
        discount_factor: float,
) -> FloatND:
        """
        Utility when agent dies.
        """
        # liquidation_cost = 1/3 if zilliq=1
        beq = wealth + (wealth_illiquid * (1-liquidation_cost))
        beq_annuity = interest_rate * (jnp.maximum(beq, 0)) # usan en matlab (jnp.maximum(interest_rate-1, 0)) porque puede ser tasa negativa... pero mi tasa 1 no es negativa, 2, seria mas bien maximo entre la tasa y 0 porque no es bruta la tasa que pongo
        u_baseline = mean_hhs * (((mean_hhy/mean_hhs)**(1-risk_aversion))-1) / (1-risk_aversion) # me falta agregar si rho=1
        u_bequest = mean_hhs * ((((mean_hhy + beq_annuity)/mean_hhs)**(1-risk_aversion))-1) / (1-risk_aversion)
        return (alpha/(1-discount_factor)) * (u_bequest - u_baseline)

def exponential_H(
    utility: float,
    E_next_V: float,
    discount_factor: float,
) -> float:
    return utility + discount_factor * E_next_V

def beta_delta_H(
    utility: float,
    E_next_V: float,
    beta: float,
    delta: float,
) -> float:
    return utility + beta * delta * E_next_V

### function for params ###

def load_survival_probs(survival_document_paths_woman, survival_document_paths_man):
    
    """
    Uploads survival probabilities of man and woman for later use.

    """
    
    death_m = pd.read_csv(survival_document_paths_man, skiprows=2, header=None).iloc[:, 1:].values
    death_f = pd.read_csv(survival_document_paths_woman, skiprows=2, header=None).iloc[:, 1:].values

    deat_t = (death_m + death_f) / 2

    # Selección tipo MATLAB
    deat_t = deat_t[100:105, 20:91]

    # Media por columnas
    deat_t = np.mean(deat_t, axis=0)

    # Convertir a JAX
    deat_t = jnp.array(deat_t)

    # Probabilidades de supervivencia
    surv_t = 1.0 - deat_t

    # Último valor = 0
    survival_probs = surv_t.at[-1].set(0.0)

    return survival_probs