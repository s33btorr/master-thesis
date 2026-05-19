import numpy as np
import pandas as pd
from config import MOMENT_NAMES, AGE_GROUPS


def compute_avg_income_by_age(df: pd.DataFrame) -> pd.Series:
    """
    Average income by age without people dead in the period calculated.
    """
    return df[df["regime"] != "dead"].groupby("age")["earnings"].mean()


def compute_simulated_moments(df: pd.DataFrame) -> pd.Series:
    """
    Calculates 16 moments.

    Args:
        df: DataFrame coming from PyLCM simulation.

    Returns:
        pd.Series with 16 moments.
    """
    df_alive = df[df["regime"] != "dead"].copy()
    df_alive["simL"] = df_alive["wealth_x"]

    df_alive["simW"] = df_alive["wealth_z"] + df_alive["wealth_x"]

    avg_income = compute_avg_income_by_age(df_alive)

    fborr_all     = []
    mborr_all     = []
    wealth_debt   = []
    wealth_nodebt = []

    for ages in AGE_GROUPS:
        df_group = df_alive[df_alive["age"].isin(ages)]
        avg_y    = np.array([avg_income.get(a, 1.0) for a in ages])

       
        fborr_by_age = (
            df_group.groupby("age")["simL"]
            .apply(lambda x: (x < 0).mean())
            .reindex(ages).fillna(0).values
        )
        fborr_all.append(fborr_by_age.mean())

       
        debt_by_age = (
            df_group.groupby("age")["simL"]
            .apply(lambda x: np.minimum(x, 0).mean())
            .reindex(ages).fillna(0).values
        )
        mborr_all.append((-debt_by_age / avg_y).mean())

      
        wd_ag  = []
        wnd_ag = []
        for i, a in enumerate(ages):
            df_age      = df_group[df_group["age"] == a]
            avg_y_a     = avg_y[i]
            debt_mask   = df_age["simL"] < 0
            nodebt_mask = ~debt_mask

            wd  = df_age.loc[debt_mask,   "simW"].mean() if debt_mask.any()   else 0.0
            wnd = df_age.loc[nodebt_mask, "simW"].mean() if nodebt_mask.any() else 0.0

            wd_ag.append(wd   / avg_y_a)
            wnd_ag.append(wnd / avg_y_a)

        wealth_debt.append(np.mean(wd_ag))
        wealth_nodebt.append(np.mean(wnd_ag))

    sim_moments = np.array(fborr_all + mborr_all + wealth_debt + wealth_nodebt)
    return pd.Series(sim_moments, index=MOMENT_NAMES)