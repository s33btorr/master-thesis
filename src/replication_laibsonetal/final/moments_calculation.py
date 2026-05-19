import numpy as np
import pandas as pd
from config import MOMENT_NAMES, AGE_GROUPS




# =============================================================================
# Funciones auxiliares
# =============================================================================

def compute_avg_income_by_age(df: pd.DataFrame) -> pd.Series:
    """
    Ingreso promedio por edad — equivalente a avgY_ en Matlab.
    Excluye agentes muertos.
    """
    return df[df["regime"] != "dead"].groupby("age")["earnings"].mean()


def compute_simulated_moments(df: pd.DataFrame) -> pd.Series:
    """
    Calcula los 16 momentos simulados.

    Equivalente a la sección 'Compute Moments' de LifecycleSim.m.

    Args:
        df: DataFrame con resultados del modelo (output de to_dataframe).

    Returns:
        pd.Series con 16 momentos, indexados por MOMENT_NAMES.
    """
    # excluir agentes muertos — equivalente a ponderar por alive_ en Matlab
    df_alive = df[df["regime"] != "dead"].copy()

    # simL__ en Matlab = riqueza líquida ANTES de recibir ingreso
    # en lcm, wealth ya es la riqueza al inicio del período (antes del ingreso)
    # por lo que simL__ = wealth directamente
    df_alive["simL"] = df_alive["wealth"]

    # simW__ en Matlab = riqueza total ANTES de recibir ingreso
    # = illiquid + liquid (antes de ingreso)
    df_alive["simW"] = df_alive["wealth_z"] + df_alive["wealth"]

    # ingreso promedio por edad — avgY_ en Matlab
    avg_income = compute_avg_income_by_age(df_alive)

    fborr_all     = []
    mborr_all     = []
    wealth_debt   = []
    wealth_nodebt = []

    for ages in AGE_GROUPS:
        df_group = df_alive[df_alive["age"].isin(ages)]
        avg_y    = np.array([avg_income.get(a, 1.0) for a in ages])

        # --- fborr_all ---
        # fracción con deuda líquida (simL__ < 0)
        # Matlab: mean(simL__ < 0) por edad, luego promedio del grupo
        fborr_by_age = (
            df_group.groupby("age")["simL"]
            .apply(lambda x: (x < 0).mean())
            .reindex(ages).fillna(0).values
        )
        fborr_all.append(fborr_by_age.mean())

        # --- mborr_all ---
        # deuda media / ingreso promedio
        # Matlab: -mean(min(0, simL__)) ./ avgY_ por edad, luego promedio del grupo
        debt_by_age = (
            df_group.groupby("age")["simL"]
            .apply(lambda x: np.minimum(x, 0).mean())
            .reindex(ages).fillna(0).values
        )
        mborr_all.append((-debt_by_age / avg_y).mean())

        # --- wealth_debt y wealth_nodebt ---
        # riqueza total media / ingreso, condicional en tener/no tener deuda
        # Matlab: mean(simW__(debt,a)) ./ avgY_(a) por edad, luego promedio del grupo
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


def load_empirical_moments(
    data_moments: np.ndarray,
    vcv_secondstage: np.ndarray,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Carga los momentos empíricos y su matriz de covarianza.

    Equivalente a SecondStageMoments.m en Matlab.

    Args:
        data_moments    : array de 16 momentos empíricos
        vcv_secondstage : matriz 16x16 de covarianza de momentos empíricos

    Returns:
        empirical_moments : pd.Series con 16 momentos
        moments_cov       : pd.DataFrame 16x16
    """
    empirical_moments = pd.Series(data_moments, index=MOMENT_NAMES)
    moments_cov = pd.DataFrame(
        vcv_secondstage,
        index=MOMENT_NAMES,
        columns=MOMENT_NAMES,
    )
    return empirical_moments, moments_cov


def compute_weighting_matrix(
    vcv_secondstage: np.ndarray,
    method: int = 0,
) -> np.ndarray:
    """
    Calcula la matriz de pesos W.

    Equivalente al switch 'weighting' en EDFbatch_baseline.m:
        0 → diagonal de la VCV (benchmark): inv(diag(diag(VCV)))
        1 → matriz identidad
        2 → VCV completa inversa

    Args:
        vcv_secondstage : matriz 16x16 de covarianza
        method          : 0 (benchmark), 1 (identidad), 2 (VCV completa)

    Returns:
        W : matriz de pesos 16x16
    """
    if method == 0:
        return np.diag(1.0 / np.diag(vcv_secondstage))
    elif method == 1:
        return np.eye(len(vcv_secondstage))
    elif method == 2:
        return np.linalg.inv(vcv_secondstage)
    else:
        raise ValueError(f"weighting method {method} no reconocido. Usa 0, 1 o 2.")