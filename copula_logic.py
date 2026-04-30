import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, optimize


EPS = 1e-8
COPULA_CFG_PATH = Path("Models/copula_cfg.json")

_alpha_sorted: np.ndarray | None = None
_alpha_n: int | None = None
_ah_values: np.ndarray | None = None
_iu_values: np.ndarray | None = None


def fit_marginals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: DataFrame with real_alpha, alpha_hat, implied_upside
    Output: cleaned DataFrame, stores empirical marginals globally
    """
    global _alpha_sorted, _alpha_n, _ah_values, _iu_values

    cols = ["real_alpha", "alpha_hat", "implied_upside"]
    data = df[cols].copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    for col in cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    _alpha_sorted = np.sort(data["real_alpha"].values)
    _alpha_n = len(_alpha_sorted)
    _ah_values = data["alpha_hat"].values.copy()
    _iu_values = data["implied_upside"].values.copy()

    return data


def ecdf(x: float, values: np.ndarray) -> float:
    """
    Input: scalar value and sample values
    Output: empirical CDF value in (0, 1)
    """
    ranks = stats.rankdata(np.append(values, x), method="average")[-1]
    return float(np.clip(ranks / (len(values) + 1.0), EPS, 1.0 - EPS))


def pseudo_obs(x: np.ndarray) -> np.ndarray:
    """
    Input: sample values
    Output: pseudo-observations in (0, 1)
    """
    arr = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce"), dtype=float)
    u = stats.rankdata(arr, method="average") / (len(arr) + 1.0)
    return np.clip(u, EPS, 1.0 - EPS)


def inv_ecdf(u: np.ndarray) -> np.ndarray:
    """
    Input: pseudo-observations in (0, 1)
    Output: quantiles of real_alpha via linear interpolation
    """
    u = np.clip(np.asarray(u, dtype=float), EPS, 1.0 - EPS)
    pos = u * (_alpha_n + 1.0) - 1.0
    frac = pos - np.floor(pos)
    lo = np.clip(np.floor(pos).astype(int), 0, _alpha_n - 1)
    hi = np.clip(lo + 1, 0, _alpha_n - 1)
    return _alpha_sorted[lo] * (1.0 - frac) + _alpha_sorted[hi] * frac


#log-like

def _clayton_loglik(u: np.ndarray, v: np.ndarray, theta: float) -> float:
    """
    Input: pseudo-observations and Clayton theta
    Output: Clayton copula log-likelihood
    """
    if theta <= 0:
        return -np.inf

    s = u ** (-theta) + v ** (-theta) - 1.0

    if np.any(s <= 0):
        return -np.inf

    lls = (
        np.log1p(theta)
        + (-theta - 1.0) * (np.log(u) + np.log(v))
        + (-2.0 - 1.0 / theta) * np.log(s)
    )

    val = float(np.sum(lls))
    return val if np.isfinite(val) else -np.inf


def _clayton_logpdf_vec(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """
    Input: pseudo-observations and Clayton theta
    Output: Clayton copula log-density
    """
    u = np.clip(u, EPS, 1.0 - EPS)
    v = np.clip(v, EPS, 1.0 - EPS)

    s = np.maximum(u ** (-theta) + v ** (-theta) - 1.0, EPS)

    return (
        np.log1p(theta)
        + (-theta - 1.0) * (np.log(u) + np.log(v))
        + (-2.0 - 1.0 / theta) * np.log(s)
    )


#cmle

def clayton_cmle(u: np.ndarray, v: np.ndarray) -> float:
    """
    Input: two pseudo-observation arrays
    Output: fitted Clayton theta
    """
    tau = float(stats.kendalltau(u, v).statistic)

    if not np.isfinite(tau) or tau <= 1e-4:
        return EPS

    theta0 = float(np.clip(2.0 * tau / (1.0 - tau), EPS, 40.0))

    res = optimize.minimize(
        lambda x: -_clayton_loglik(u, v, float(x[0])),
        x0=[theta0],
        method="L-BFGS-B",
        bounds=[(EPS, 80.0)],
    )

    return float(res.x[0]) if res.success else theta0


#cfg

def save_copula_config(params: dict, path: Path = COPULA_CFG_PATH) -> None:
    """
    Input: copula parameter dictionary and path
    Output: saves parameters to JSON
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=4)


def load_copula_config(path: Path = COPULA_CFG_PATH) -> dict:
    """
    Input: path to JSON
    Output: copula parameter dictionary
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def prepare_copula_runtime(
        train_path: str = "Datasets/train_dataset_full.csv"
) -> None:
    """
    Input: path to train dataset
    Output: loads empirical marginals for live prediction
    """
    global _alpha_sorted, _ah_values, _iu_values

    if _alpha_sorted is not None and _ah_values is not None and _iu_values is not None:
        return

    df = pd.read_csv(train_path)
    fit_marginals(df)

#Saving

def fit_copula_model(df: pd.DataFrame, path: Path = COPULA_CFG_PATH) -> dict:
    """
    Input: DataFrame with real_alpha, alpha_hat, implied_upside
    Output: fitted copula parameters saved to JSON and returned as dict
    """
    data = fit_marginals(df)

    u_ra = pseudo_obs(data["real_alpha"].values)
    u_ah = pseudo_obs(data["alpha_hat"].values)
    u_iu = pseudo_obs(data["implied_upside"].values)

    theta_ra_ah = clayton_cmle(u_ra, u_ah)
    theta_ra_iu = clayton_cmle(u_ra, u_iu)

    params = {
        "theta_ra_ah": theta_ra_ah,
        "theta_ra_iu": theta_ra_iu,
    }

    save_copula_config(params, path)
    return params


#Usage

def _weighted_quantile(values: np.ndarray, weights: np.ndarray, qs: list[float]) -> np.ndarray:
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cdf = np.cumsum(weights)
    cdf /= cdf[-1]

    return np.interp(qs, cdf, values)


def copula_predict(
        alpha_hat: float,
        implied_upside: float,
        path: Path = COPULA_CFG_PATH,
        grid_size: int = 500,
        thresholds: list[float] | None = None,
) -> pd.Series:
    """
    Input: raw alpha_hat, implied_upside
    Output: conditional alpha distribution metrics
    """
    params = load_copula_config(path)

    if _alpha_sorted is None or _ah_values is None or _iu_values is None:
        raise RuntimeError("Run fit_copula_model(df) or fit_marginals(df) before prediction.")

    if thresholds is None:
        thresholds = [0.0, -0.10, -0.20, 0.10, 0.20]

    u_ah_obs = ecdf(alpha_hat, _ah_values)
    u_iu_obs = ecdf(implied_upside, _iu_values)

    w = np.linspace(EPS, 1.0 - EPS, grid_size)

    theta_ra_ah = params["theta_ra_ah"]
    theta_ra_iu = params["theta_ra_iu"]

    v_ah = np.full_like(w, u_ah_obs)
    v_iu = np.full_like(w, u_iu_obs)

    lp = _clayton_logpdf_vec(w, v_ah, theta_ra_ah)
    lp += _clayton_logpdf_vec(w, v_iu, theta_ra_iu)

    lp -= np.max(lp)
    density = np.exp(lp)

    area = np.trapezoid(density, w)

    if not np.isfinite(area) or area <= 0:
        return pd.Series(dtype=float)

    density /= area

    alpha_vals = inv_ecdf(w)

    expected_alpha = float(np.trapezoid(density * alpha_vals, w))
    expected_alpha_sq = float(np.trapezoid(density * alpha_vals ** 2, w))
    alpha_std = float(np.sqrt(max(expected_alpha_sq - expected_alpha ** 2, 0.0)))

    q05, q25, q50, q75, q95 = _weighted_quantile(
        alpha_vals,
        density,
        [0.05, 0.25, 0.50, 0.75, 0.95],
    )

    prob_positive = float(np.trapezoid(density * (alpha_vals > 0).astype(float), w))
    prob_negative = float(np.trapezoid(density * (alpha_vals < 0).astype(float), w))

    prob_alpha_lt_minus_10 = float(
        np.trapezoid(density * (alpha_vals < -0.10).astype(float), w)
    )
    prob_alpha_lt_minus_20 = float(
        np.trapezoid(density * (alpha_vals < -0.20).astype(float), w)
    )

    positive_mask = alpha_vals > 0
    negative_mask = alpha_vals < 0

    positive_area = np.trapezoid(density * positive_mask.astype(float), w)
    negative_area = np.trapezoid(density * negative_mask.astype(float), w)

    expected_upside = (
        float(np.trapezoid(density * alpha_vals * positive_mask.astype(float), w) / positive_area)
        if positive_area > 0 else np.nan
    )

    expected_downside = (
        float(np.trapezoid(density * alpha_vals * negative_mask.astype(float), w) / negative_area)
        if negative_area > 0 else np.nan
    )

    alpha_efficiency = expected_alpha / alpha_std if alpha_std > 0 else np.nan

    result = {
        "prob_positive": prob_positive,
        "prob_negative": prob_negative,
        "prob_alpha_lt_minus_10": prob_alpha_lt_minus_10,
        "prob_alpha_lt_minus_20": prob_alpha_lt_minus_20,
        "expected_alpha": expected_alpha,
        "median_alpha": float(q50),
        "alpha_std": alpha_std,
        "alpha_q05": float(q05),
        "alpha_q25": float(q25),
        "alpha_q50": float(q50),
        "alpha_q75": float(q75),
        "alpha_q95": float(q95),
        "expected_upside": expected_upside,
        "expected_downside": expected_downside,
        "alpha_efficiency": float(alpha_efficiency),
        "alpha_sharpe_like": float(alpha_efficiency),
        "implied_upside_percentile": u_iu_obs,
        "alpha_hat_percentile": u_ah_obs,
    }

    for threshold in thresholds:
        key = str(threshold).replace("-", "minus_").replace(".", "_")

        result[f"prob_alpha_gt_{key}"] = float(
            np.trapezoid(density * (alpha_vals > threshold).astype(float), w)
        )

        result[f"prob_alpha_lt_{key}"] = float(
            np.trapezoid(density * (alpha_vals < threshold).astype(float), w)
        )

    return pd.Series(result)


def copula_predict_batch(
        df_test: pd.DataFrame,
        path: Path = COPULA_CFG_PATH,
        grid_size: int = 500,
        thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Input: test DataFrame with alpha_hat, implied_upside
    Output: DataFrame with conditional alpha distribution metrics
    """
    results = []

    for _, row in df_test.iterrows():
        pred = copula_predict(
            alpha_hat=row["alpha_hat"],
            implied_upside=row["implied_upside"],
            path=path,
            grid_size=grid_size,
            thresholds=thresholds,
        )

        results.append(pred)

    return pd.DataFrame(results)


if __name__ == "__main__":
    try:
        import os
        os.remove(str(COPULA_CFG_PATH))
    except FileNotFoundError:
        pass

    df = pd.read_csv("Datasets/train_dataset_full.csv")

    params = fit_copula_model(df)


#Notes
"""
Model intuition and empirical relationships

The copula model is built to capture the conditional relationship between the DCF-derived signal 
(implied_upside) and the realized market-relative performance (real_alpha). Empirically, this 
relationship is weak but stable and monotonic in rank space rather than in levels.

Across the dataset (train/test split around 2023), the dependence between signals and realized alpha 
is positive but moderate:

- Rank correlations (Spearman, Kendall) are consistently low-to-mid (~0.15–0.26),
  indicating that the signal contains information but is far from deterministic.
- Pearson correlations are slightly higher (~0.33–0.36), reflecting sensitivity to tail observations.
- Sign accuracy (predicting alpha > 0) is relatively strong (~72%), suggesting that the model 
  captures directional tendencies better than magnitude.

The copula approach focuses on modeling this dependence structure in a non-linear and distribution-aware way.
Compared to direct ML predictions:

- The copula produces slightly lower rank correlations, but remains competitive.
- It provides a full conditional distribution of alpha instead of a point estimate.
- Calibration analysis shows that predicted probabilities of positive alpha increase monotonically 
  with realized frequency, although the model tends to be mildly conservative (underestimates positive outcomes).

Importantly, the distribution of realized alpha is heavy-tailed and asymmetric:

- Extreme negative outcomes are more frequent than implied by a Gaussian assumption.
- The conditional expected alpha is often negative in aggregate, reflecting a generally 
  competitive or unfavorable market environment in the sample.
- Upside scenarios exist but are driven by right-tail events rather than central tendency.

Therefore, the model should be interpreted as:

    A probabilistic mapping from DCF signal strength to the conditional distribution of alpha,
    not as a precise point predictor.

The key output of the model is not alpha itself, but:

- Probability of outperformance (P(alpha > 0))
- Conditional expectation (E[alpha | signal])
- Distribution shape (quantiles, tails, asymmetry)

These quantities are more stable and informative than direct alpha predictions, 
especially under weak signal-to-noise conditions typical for equity markets.
"""