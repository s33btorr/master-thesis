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
 
    Parameters come from first-stage demographics regression.
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
    
    """
    Number of kids as a function of age.
    """
    return a0kids * jnp.exp((a1kids * age) - (a2kids * (age ** 2)/100))

def number_of_depadul(
        age: float,
        a0depadul: float,
        a1depadul: float,
        a2depadul: float,
) -> FloatND:
    
    """
    Number of dependent adults as a function of age.
    """
    return a0depadul * jnp.exp((a1depadul * age) - (a2depadul * (age ** 2)/100))

def liquidation_cost(
    age: float,
    #zilliq: float,
) -> FloatND:
    
    """
    Liquidation cost as a function of age.
    """

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


def deterministic_retirement_income(
    age: float,
    yret_cons: float,
    yret_agecoeff: float,
) -> FloatND:
    """
    Deterministic component of log income for retirement (cubic age polynomial).
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
    Total earnings in levels.
    Log income = deterministic_mean + permanent_AR1_shock + transitory_iid_shock.
    """
    return jnp.exp(deterministic + (perm_income + trans_income))


def average_earnings( 
    ywork_cons: float,
    ywork_agecoeff: float,
    ywork_age2coeff: float,
    ywork_age3coeff: float,
    ywork_kidscoeff: float,
    ywork_spousecoeff: float,
    ywork_depadulcoeff: float,
    a0kids: float,
    a1kids: float,
    a2kids: float,
    a0depadul: float,
    a1depadul: float,
    a2depadul: float,
    ywork_auto: float,
    ywork_vareps: float,
    ywork_varnu: float,
) -> float:
    """
    Average earnings among years.
    The shocks are not included because they have expected value 0.
    """
    ages = jnp.arange(20, 91, dtype=jnp.float32)
    spouse = 2

    number_of_kids    = a0kids    * jnp.exp((a1kids    * ages) - (a2kids    * (ages ** 2) / 100))
    number_of_depadul = a0depadul * jnp.exp((a1depadul * ages) - (a2depadul * (ages ** 2) / 100))

    deterministic_profile = (
        ywork_cons
        + ywork_kidscoeff    * number_of_kids
        + ywork_spousecoeff  * spouse
        + ywork_depadulcoeff * number_of_depadul
        + ywork_agecoeff     * ages
        + ywork_age2coeff    * (ages ** 2) / 100
        + ywork_age3coeff    * (ages ** 3) / 10000
    )

    ywork_eps = ywork_vareps * 0.5
    ywork_nu = ywork_varnu * 0.5
    var_ar1 = ywork_eps / (1 - ywork_auto**2)
    var_iid = ywork_nu

    expected_earnings = jnp.exp(deterministic_profile + 0.5*(var_ar1 + var_iid))

    return jnp.mean(expected_earnings)

### CREDIT LIMIT ###

def average_income(
    deterministic: FloatND,
    ywork_auto: float,
    ywork_vareps: float,
    ywork_varnu: float,
) -> float:
    
    """
    Average income in specific period t.
    Same for all households, varies with age.
    """

    ywork_eps = ywork_vareps * 0.5
    ywork_nu = ywork_varnu * 0.5
    var_ar1 = ywork_eps / (1 - ywork_auto**2)
    var_iid = ywork_nu
    return jnp.exp(deterministic + 0.5*(var_ar1 + var_iid))

def credit_limit(
    age: float,
    c0credit: float,
    c1credit: float,
    c2credit: float,
    average_income: float,
) -> float:
    
    """
    Absolute value of credit limit that depends on average income in period t.
    """

    credit_limit_rate = c0credit + (c1credit*age) + (c2credit*(age**2)/100) # en el paper esta con el (age**2)/ 100, en el codigo esta sin el 100... PROBAR CON AMBOS
    return average_income * credit_limit_rate

#### UTILITY ####
def consumption(
    earnings: FloatND, 
    investment_x: ContinuousAction, 
    investment_z: ContinuousAction, 
    liquidation_cost: FloatND,
) -> FloatND:
    
    """
    Consumption without including illiquid wealth.
    """

    liq_cost = liquidation_cost * jnp.minimum(investment_z, 0)
    return earnings - investment_x - investment_z + liq_cost


def total_consumption(
    consumption: FloatND,
    wealth_z: ContinuousState,
) -> FloatND:
    
    """
    Consumption including illiquid wealth, as it enters in the utility.
    """

    return consumption + (0.05*wealth_z)


def utility(
    total_consumption: FloatND,
    household_size: FloatND,
    risk_aversion: float,
) -> FloatND:
  
    """
    CRRA utility scaled by household size.
    Note: Changes if rho=1.
    """
    c_per_hh = total_consumption / household_size
    numerador = household_size * ((c_per_hh**(1 - risk_aversion)) - 1) # hice todo esto porque me estaba dando valores raros para ver si se acomoda. Asi lo hacen en matlab
    denominador = 1 - risk_aversion
    return jnp.where(risk_aversion == 1, household_size * jnp.log(c_per_hh), numerador / denominador)

def beq_utility(
        mean_hhs: float,
        average_earnings: float,
        risk_aversion: float,
        wealth_x: ContinuousState,
        wealth_z: ContinuousState,
        liquidation_cost: FloatND,
        interest_rate: float,
        alpha: float,
        discount_factor: float,
) -> FloatND:
        """
        Utility when agent dies.
        """
        # liquidation_cost = 1/3 if zilliq=1
        beq = wealth_x + (wealth_z * (1-liquidation_cost))
        beq_annuity = interest_rate * (jnp.maximum(beq, 0)) # usan en matlab (jnp.maximum(interest_rate-1, 0)) porque puede ser tasa negativa... pero mi tasa 1 no es negativa, 2, seria mas bien maximo entre la tasa y 0 porque no es bruta la tasa que pongo
        u_baseline = mean_hhs * (((average_earnings/mean_hhs)**(1-risk_aversion))-1) / (1-risk_aversion) # me falta agregar si rho=1
        u_bequest = mean_hhs * ((((average_earnings + beq_annuity)/mean_hhs)**(1-risk_aversion))-1) / (1-risk_aversion)
        return (alpha/(1-discount_factor)) * (u_bequest - u_baseline)

def exponential_H(
    utility: float,
    E_next_V: float,
    discount_factor: float,
) -> float:
    """
    Value function for exponential agent.
    """
    return utility + discount_factor * E_next_V

def beta_delta_H(
    utility: float,
    E_next_V: float,
    beta: float,
    delta: float,
) -> float:
    """
    Value function for naive agent.
    """
    return utility + beta * delta * E_next_V

### function for params ###

def load_survival_probs(survival_document_paths_woman, survival_document_paths_man):
    
    """
    Uploads survival probabilities of man and woman from data.
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