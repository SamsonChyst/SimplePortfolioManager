from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from dataset_processor import *
import numpy as np
from scipy import stats
from scipy import optimize
import pandas as pd

# cfg
INPUT_CSV = Path("Datasets/US_WaccComponents_Timeseries.csv")  # For US Market
# Source: Risk-free rates and equity premium Stern-Damodaran,
# Marginal tax rates - IRS, Annual AVG Credit Spreads - BBB OAS

_WACC_TABLE = pd.read_csv(INPUT_CSV, sep=";", header=0, index_col="Year")


@dataclass(frozen=True)
class ValuationKey:
    ticker: str
    valuation_year: int


# Valuation

def dcf_proxy(
        df: pd.DataFrame,
        date_col: str = "Date",
        net_debt_radius: float = 0.03,
        alpha_horizon: int = 1,
) -> pd.Series:
    """
    Input: sliced timeseries DataFrame
    Output: returns 5th historical year FCFF, FCFF CAGR, Y5 price, comparison price, and Y5 net debt
    """
    if df is None or df.empty:
        return pd.Series(dtype="float64")

    data = df.copy()

    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    elif isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index().rename(columns={data.index.name or "index": date_col})
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    else:
        raise ValueError(f"DataFrame must contain '{date_col}' column or DatetimeIndex")

    data = data.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    comp_idx = 4 + alpha_horizon
    if len(data) <= comp_idx:
        return pd.Series(dtype="float64")

    for col in [
        "price", "ev", "fcff", "net_debt", "total_debt",
        "market_cap", "market_price", "cash", "shares_yf"
    ]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        else:
            data[col] = np.nan

    if data["net_debt"].isna().any():
        fallback_from_balance = data["total_debt"] - data["cash"]
        data["net_debt"] = data["net_debt"].where(data["net_debt"].notna(), fallback_from_balance)

    if data["net_debt"].isna().any():
        implied_net_debt = data["ev"] - data["market_cap"]
        implied_scale = np.maximum.reduce([
            data["ev"].abs().fillna(0.0).to_numpy(dtype=float),
            data["market_cap"].abs().fillna(0.0).to_numpy(dtype=float),
            np.ones(len(data), dtype=float)
        ])
        implied_ratio = np.abs(implied_net_debt.to_numpy(dtype=float)) / implied_scale
        implied_net_debt = implied_net_debt.where(implied_ratio > net_debt_radius, 0.0)

        data["net_debt"] = data["net_debt"].where(data["net_debt"].notna(), implied_net_debt)

    hist = data.iloc[:5]
    comp = data.iloc[comp_idx]

    fcff_positive = hist.loc[hist["fcff"] > 0, "fcff"]
    fcff_idx = fcff_positive.last_valid_index()
    if fcff_idx is None:
        fcff_idx = hist["fcff"].last_valid_index()
    if fcff_idx is None:
        fcff_idx = hist.index[-1]

    y5 = hist.loc[fcff_idx]

    fcff_5 = y5["fcff"]
    shares_5 = y5["shares_yf"]
    price_5 = y5["price"]
    net_debt_5 = y5["net_debt"]
    total_debt_5 = y5["total_debt"]
    market_cap_5 = y5["market_cap"]
    market_price_5 = y5["market_price"]

    price_comp = comp["price"]
    market_price_comp = comp["market_price"]

    fcff_cagr = np.nan
    fcff_hist = hist[[date_col, "fcff"]].dropna()
    fcff_hist = fcff_hist[fcff_hist["fcff"] > 0]

    if len(fcff_hist) >= 2:
        fcff_hist = fcff_hist.sort_values(date_col).reset_index(drop=True)

        yoy_growth = []
        for i in range(1, len(fcff_hist)):
            prev_fcff = fcff_hist.loc[i - 1, "fcff"]
            curr_fcff = fcff_hist.loc[i, "fcff"]
            prev_date = fcff_hist.loc[i - 1, date_col]
            curr_date = fcff_hist.loc[i, date_col]

            year_gap = (curr_date - prev_date).days / 365.25
            if year_gap <= 0:
                continue

            if prev_fcff <= 0 or curr_fcff <= 0:
                continue

            growth = (curr_fcff / prev_fcff) ** (1 / year_gap) - 1
            if pd.notna(growth) and np.isfinite(growth):
                yoy_growth.append(growth)

        if len(yoy_growth) > 0:
            fcff_cagr = float(np.median(yoy_growth))
        else:
            x = ((fcff_hist[date_col] - fcff_hist[date_col].min()).dt.days / 365.25).to_numpy(dtype=float)
            y = np.log(fcff_hist["fcff"].to_numpy(dtype=float))
            mask = np.isfinite(x) & np.isfinite(y)

            x = x[mask]
            y = y[mask]

            if len(x) >= 2 and np.ptp(x) > 0:
                try:
                    slope = np.polyfit(x - x.mean(), y, 1)[0]
                    if np.isfinite(slope):
                        fcff_cagr = np.exp(slope) - 1
                except Exception:
                    fcff_cagr = np.nan

    if pd.notna(fcff_cagr) and np.isfinite(fcff_cagr):
        fcff_cagr = 0.15 * np.tanh(fcff_cagr / 0.15)

    real_price_change = np.nan
    if pd.notna(price_5) and pd.notna(price_comp) and np.isfinite(price_5) and price_5 != 0:
        real_price_change = (price_comp / price_5) - 1

    real_market_price_change = np.nan
    if pd.notna(market_price_5) and pd.notna(market_price_comp) and np.isfinite(market_price_5) and market_price_5 != 0:
        real_market_price_change = (market_price_comp / market_price_5) - 1

    return pd.Series({
        "fcff": fcff_5,
        "fcff_cagr": fcff_cagr,
        "shares_5": shares_5,
        "price_5": price_5,
        "price_comp": price_comp,
        "real_price_change": real_price_change,
        "net_debt_5": net_debt_5,
        "market_price_5": market_price_5,
        "market_price_comp": market_price_comp,
        "real_market_price_change": real_market_price_change,
        "total_debt_5": total_debt_5,
        "market_cap_5": market_cap_5
    })


def wacc_proxy(ticker: str, valuation_year: int) -> float:
    """
    Input: Unsliced daily DataFrame, final year of valuation
    Output: Weighted Average Cost of Capital for a Company at a given moment
    """
    key = ValuationKey(ticker=ticker, valuation_year=valuation_year)

    if key.valuation_year not in _WACC_TABLE.index:
        return np.nan

    WaccDf = _WACC_TABLE.loc[key.valuation_year, :]
    DailyDf = time_slicing(key.ticker, key.valuation_year - 4, key.valuation_year)
    YearlySeries = dcf_proxy(year_slicing(key.ticker, key.valuation_year - 4, key.valuation_year + 1))

    if DailyDf is None or DailyDf.empty:
        return np.nan

    if "beta_ewm" not in DailyDf.columns:
        return np.nan

    beta_series = pd.to_numeric(DailyDf["beta_ewm"], errors="coerce").dropna()
    if beta_series.empty:
        return np.nan

    beta = float(beta_series.median())

    risk_free = float(WaccDf["T_Bond_Rate"])
    equity_premium = float(WaccDf["Implied_ERP_(FCFE)"])
    cost_of_capital = (risk_free + beta * equity_premium)

    tax_shield = 1 - float(WaccDf["US_Marginal_Tax_Rate"])

    total_capital = YearlySeries["market_cap_5"] + YearlySeries["total_debt_5"]
    if pd.isna(total_capital) or total_capital <= 0:
        return np.nan

    equity_in_structure = YearlySeries["market_cap_5"] / total_capital
    debt_in_structure = YearlySeries["total_debt_5"] / total_capital

    credit_spread = float(WaccDf["Credit_Spread"])
    cost_of_debt = risk_free + credit_spread

    return (equity_in_structure * cost_of_capital) + (debt_in_structure * cost_of_debt * tax_shield)


def dcf_valuation(
        ticker: str,
        valuation_year: int,
        forecast_years: int = 5,
        growth_rate: float = 0.0275,
        terminal_growth_rate: float = 0.01375,
        shift: int = 5,
        alpha_horizon: int = 3,
        min_wacc_terminal_spread: float = 0.03,
) -> pd.Series:
    key = ValuationKey(ticker=ticker, valuation_year=valuation_year)

    yearly_df = year_slicing(
        ticker=key.ticker,
        start_year=key.valuation_year - 4,
        end_year=key.valuation_year + alpha_horizon,
        shift=shift,
    )
    proxy = dcf_proxy(yearly_df, alpha_horizon=alpha_horizon)

    if proxy.empty:
        return pd.Series(dtype="float64")

    fcff_0 = pd.to_numeric(proxy.get("fcff"), errors="coerce")
    net_debt = pd.to_numeric(proxy.get("net_debt_5"), errors="coerce")
    shares = pd.to_numeric(proxy.get("shares_5"), errors="coerce")
    price_5 = pd.to_numeric(proxy.get("price_5"), errors="coerce")
    real_price_change = pd.to_numeric(proxy.get("real_price_change"), errors="coerce")
    real_market_price_change = pd.to_numeric(proxy.get("real_market_price_change"), errors="coerce")

    if pd.isna(fcff_0) or not np.isfinite(fcff_0):
        return pd.Series(dtype="float64")

    wacc = float(wacc_proxy(key.ticker, key.valuation_year))

    if not np.isfinite(wacc):
        return pd.Series(dtype="float64")

    spread = wacc - terminal_growth_rate
    if not np.isfinite(spread) or spread <= min_wacc_terminal_spread:
        return pd.Series(dtype="float64")

    t = np.arange(1, forecast_years + 1, dtype=float)
    fcff_forecast = fcff_0 * ((1 + growth_rate) ** t)
    pv_fcff = fcff_forecast / ((1 + wacc) ** t)

    fcff_terminal_base = fcff_forecast[-1]
    terminal_value = (fcff_terminal_base * (1 + terminal_growth_rate)) / spread
    pv_terminal_value = terminal_value / ((1 + wacc) ** forecast_years)

    enterprise_value = float(np.sum(pv_fcff) + pv_terminal_value)
    equity_value = enterprise_value - net_debt if pd.notna(net_debt) else np.nan

    value_per_share = np.nan
    if pd.notna(shares) and np.isfinite(shares) and shares != 0:
        value_per_share = equity_value / shares

    implied_upside_vs_price_5 = np.nan
    if pd.notna(price_5) and np.isfinite(price_5) and price_5 != 0 and pd.notna(value_per_share):
        implied_upside_vs_price_5 = (value_per_share / price_5) - 1

    result = {
        "ticker": key.ticker,
        "valuation_year": key.valuation_year,
        "alpha_horizon": alpha_horizon,
        "wacc": wacc,
        "growth_rate": growth_rate,
        "terminal_growth_rate": terminal_growth_rate,
        "terminal_value": float(terminal_value),
        "pv_terminal_value": float(pv_terminal_value),
        "enterprise_value": enterprise_value,
        "net_debt": net_debt,
        "equity_value": equity_value,
        "value_per_share": value_per_share,
        "real_upside": real_price_change,
        "market_upside": real_market_price_change,
        "implied_upside": implied_upside_vs_price_5
    }

    return pd.Series(result)


def target_row(
        ticker: str,
        valuation_year: int,
        forecast_years: int = 5,
        growth_rate: float = 0.0275,
        terminal_growth_rate: float = 0.01375,
        shift: int = 5,
        alpha_horizon: int = 3,
) -> pd.Series:
    """
    Input: ticker and valuation year
    Output: one-row target observation for dataset construction
    """
    val = dcf_valuation(
        ticker=ticker,
        valuation_year=valuation_year,
        forecast_years=forecast_years,
        growth_rate=growth_rate,
        terminal_growth_rate=terminal_growth_rate,
        shift=shift,
        alpha_horizon=alpha_horizon,
    )

    if val is None or len(val) == 0:
        return pd.Series(dtype="float64")

    implied_upside = pd.to_numeric(val.get("implied_upside"), errors="coerce")
    real_upside = pd.to_numeric(val.get("real_upside"), errors="coerce")
    market_upside = pd.to_numeric(val.get("market_upside"), errors="coerce")

    if pd.isna(implied_upside) or not np.isfinite(implied_upside):
        return pd.Series(dtype="float64")

    if pd.isna(real_upside) or not np.isfinite(real_upside):
        return pd.Series(dtype="float64")

    if pd.isna(market_upside) or not np.isfinite(market_upside):
        return pd.Series(dtype="float64")

    real_alpha = real_upside - market_upside

    if pd.isna(real_alpha) or not np.isfinite(real_alpha):
        return pd.Series(dtype="float64")

    return pd.Series({
        "ticker": ticker,
        "valuation_year": int(valuation_year),
        "alpha_horizon": int(alpha_horizon),
        "real_alpha": float(real_alpha),
        "implied_upside": float(implied_upside),
    })


# ML Label is copula-based upon Bayesian P(real_alpha | DCF_signal, T_Bond_Rate)
# C-vine Copula based Labelling

def clean_copula_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: a DataFrame after load_train_jsons
    Output: a filtered DataFrame, without Nans & infinities
    """
    out = df.copy()

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["implied_upside", "real_alpha", "t_bond_rate"])

    out["implied_upside"] = pd.to_numeric(out["implied_upside"], errors="coerce")
    out["real_alpha"] = pd.to_numeric(out["real_alpha"], errors="coerce")
    out["t_bond_rate"] = pd.to_numeric(out["t_bond_rate"], errors="coerce")

    out = out.dropna(subset=["implied_upside", "real_alpha", "t_bond_rate"])
    out = out.drop_duplicates(subset=["ticker", "valuation_year"])

    return out.reset_index(drop=True)


def marginal_diagnostics(x: pd.Series) -> dict:
    arr = np.asarray(x, dtype=float)
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "min": float(np.min(arr)),
        "q01": float(np.quantile(arr, 0.01)),
        "q05": float(np.quantile(arr, 0.05)),
        "median": float(np.median(arr)),
        "q95": float(np.quantile(arr, 0.95)),
        "q99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
        "skew": float(stats.skew(arr, bias=False)),
        "kurtosis_excess": float(stats.kurtosis(arr, fisher=True, bias=False)),
    }


def prepare_copula_dataset(train_dir: str = "Datasets/train/") -> pd.DataFrame:
    """
    Input: directory of the train Dataset
    Output: clean DataFrame with empirical pseudo-observations for C-vine copula modeling
    C-vine root node is T_Bond_Rate (w_R) — strongest conditioning variable.
    Pair 1: Clayton(u_A, v_D)         — alpha vs DCF (unconditional)
    Pair 2: Clayton(u_A | w_R, v_D | w_R) — alpha vs DCF conditional on T_Bond
    """
    df = load_train_jsons(train_dir)
    df = clean_copula_dataset(df)

    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")

    out = df.copy()
    n = len(out)

    a = np.asarray(out["real_alpha"], dtype=float)
    d = np.asarray(out["implied_upside"], dtype=float)
    r = np.asarray(out["t_bond_rate"], dtype=float)

    out["u_A"] = stats.rankdata(a, method="average") / (n + 1.0)
    out["v_D"] = stats.rankdata(d, method="average") / (n + 1.0)
    out["w_R"] = stats.rankdata(r, method="average") / (n + 1.0)

    return out


@lru_cache(maxsize=4)
def _cached_copula_arrays(train_dir: str = "Datasets/train/"):
    df = prepare_copula_dataset(train_dir=train_dir)

    alpha_sample = np.sort(df["real_alpha"].dropna().to_numpy(dtype=float))
    dcf_sample = np.sort(df["implied_upside"].dropna().to_numpy(dtype=float))
    tbond_sample = np.sort(df["t_bond_rate"].dropna().to_numpy(dtype=float))

    return alpha_sample, dcf_sample, tbond_sample


def _copula_uv(df: pd.DataFrame, u_col: str = "u_A", v_col: str = "v_D", eps: float = 1e-12):
    """
    Input: prepared copula DataFrame with pseudo-observations
    Output: clipped numpy arrays u and v for copula likelihood evaluation
    """
    if df is None or df.empty:
        raise ValueError("Prepared copula dataset is empty.")
    if u_col not in df.columns or v_col not in df.columns:
        raise ValueError(f"Columns '{u_col}' and '{v_col}' must exist in DataFrame.")

    u = pd.to_numeric(df[u_col], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(df[v_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(u) & np.isfinite(v)
    u = u[mask]
    v = v[mask]

    if len(u) == 0:
        raise ValueError("No valid pseudo-observations found.")

    u = np.clip(u, eps, 1.0 - eps)
    v = np.clip(v, eps, 1.0 - eps)
    return u, v


def clayton_log_density(u: np.ndarray, v: np.ndarray, theta: float) -> np.ndarray:
    """
    Input: pseudo-observations u and v, and Clayton dependence parameter theta
    Output: pointwise log-density values of the Clayton copula used for canonical max-likelihood

    Clayton copula was chosen because it produced the best fit among tested families.
    In the model comparison it had the strongest information criteria:
    loglik = 18.8832, AIC = -35.7664, BIC = -30.9174, tau = 0.1063(the biggest out of others)
    which were better than Student-t, Gaussian, Gumbel, and Frank alternatives.
    """
    if not np.isfinite(theta) or theta <= 0.0:
        return np.full_like(u, -np.inf, dtype=float)

    log_u = np.log(np.clip(u, 1e-300, None))
    log_v = np.log(np.clip(v, 1e-300, None))

    neg_theta_log_u = -theta * log_u
    neg_theta_log_v = -theta * log_v

    overflow_mask = (neg_theta_log_u > 700) | (neg_theta_log_v > 700)
    if np.any(overflow_mask):
        return np.full_like(u, -np.inf, dtype=float)

    pu = np.exp(neg_theta_log_u)
    pv = np.exp(neg_theta_log_v)

    s = pu + pv - 1.0
    if np.any(s <= 0):
        return np.full_like(u, -np.inf, dtype=float)

    logc = (
            np.log1p(theta)
            + (-theta - 1.0) * (log_u + log_v)
            + (-2.0 - 1.0 / theta) * np.log(s)
    )

    logc = np.where(np.isfinite(logc), logc, -np.inf)
    return logc


def clayton_copula_loglik(u: np.ndarray, v: np.ndarray, theta: float) -> float:
    """
    Input: pseudo-observations u and v, and Clayton dependence parameter theta
    Output: total log-likelihood of the Clayton copula
    """
    logc = clayton_log_density(u, v, theta)
    if not np.all(np.isfinite(logc)):
        return -np.inf
    return float(np.sum(logc))


def _clayton_h_function(v: np.ndarray, u: np.ndarray, theta: float, eps: float = 1e-12) -> np.ndarray:
    """
    Input: pseudo-observations v (conditioning) and u (conditioned), Clayton theta
    Output: h(u|v) = partial C(u,v) / partial v — conditional CDF of u given v
    Used in C-vine to compute conditional pseudo-observations for the second pair copula.
    """
    v = np.clip(v, eps, 1.0 - eps)
    u = np.clip(u, eps, 1.0 - eps)

    log_v = np.log(v)
    log_u = np.log(u)

    neg_theta_log_v = -theta * log_v
    neg_theta_log_u = -theta * log_u

    if np.any(neg_theta_log_v > 700) or np.any(neg_theta_log_u > 700):
        return np.full_like(v, eps, dtype=float)

    pv = np.exp(neg_theta_log_v)
    pu = np.exp(neg_theta_log_u)

    s = pu + pv - 1.0
    s = np.clip(s, eps, None)

    h = (pv / v) * np.power(s, -1.0 / theta - 1.0)
    return np.clip(h, eps, 1.0 - eps)

RATE_THRESHOLD = 0.030 #Threshold of T-Bonds to stratificate market regimes of High price of debt/Low price of debt

def clayton_CMLE(
        train_dir: str = "Datasets/train/",
        rate_threshold: float = RATE_THRESHOLD,
) -> dict:
    """
    Input: prepared copula DataFrame with empirical pseudo-observations
    Output: fitted Clayton copula parameters for two rate regimes
    """
    df = prepare_copula_dataset(train_dir=train_dir)

    def _fit(subset: pd.DataFrame, label: str) -> dict:
        u, v = _copula_uv(subset, "u_A", "v_D")

        tau = stats.kendalltau(u, v, nan_policy="omit").statistic
        if tau is None or not np.isfinite(tau):
            theta0 = 1.0
        else:
            tau = float(np.clip(tau, 1e-4, 0.95))
            theta0 = max(2.0 * tau / (1.0 - tau), 1e-3)

        log_u = np.log(np.clip(u, 1e-300, None))
        log_v = np.log(np.clip(v, 1e-300, None))
        min_log = min(float(np.min(log_u)), float(np.min(log_v)))
        theta_max = min(700.0 / abs(min_log), 50.0) if min_log < 0 else 50.0

        def objective(x):
            ll = clayton_copula_loglik(u, v, float(x[0]))
            return np.inf if not np.isfinite(ll) else -ll

        res = optimize.minimize(
            objective,
            x0=np.array([theta0], dtype=float),
            method="L-BFGS-B",
            bounds=[(1e-6, theta_max)],
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000},
        )

        theta_hat = float(res.x[0])
        loglik = clayton_copula_loglik(u, v, theta_hat)
        n = len(u)
        k = 1
        tau_hat = theta_hat / (theta_hat + 2.0)

        return {
            "regime": label,
            "n": int(n),
            "k": int(k),
            "theta": float(theta_hat),
            "kendall_tau_implied": float(tau_hat),
            "loglik": float(loglik),
            "aic": float(2 * k - 2 * loglik),
            "bic": float(np.log(n) * k - 2 * loglik),
            "success": bool(res.success),
        }

    low  = df[df["t_bond_rate"] <  rate_threshold]
    high = df[df["t_bond_rate"] >= rate_threshold]

    return {
        "rate_threshold": rate_threshold,
        "low_rate":  _fit(low,  "low_rate"),
        "high_rate": _fit(high, "high_rate"),
    }


@lru_cache(maxsize=1)
def _cached_clayton_params(train_dir: str = "Datasets/train/") -> dict:
    return clayton_CMLE(train_dir=train_dir)


@lru_cache(maxsize=4)
def _cached_copula_arrays(train_dir: str = "Datasets/train/"):
    df = prepare_copula_dataset(train_dir=train_dir)

    def _arrays(subset: pd.DataFrame):
        alpha_sample = np.sort(subset["real_alpha"].dropna().to_numpy(dtype=float))
        dcf_sample   = np.sort(subset["implied_upside"].dropna().to_numpy(dtype=float))
        return alpha_sample, dcf_sample

    low  = df[df["t_bond_rate"] <  RATE_THRESHOLD]
    high = df[df["t_bond_rate"] >= RATE_THRESHOLD]

    return {
        "low_rate":  _arrays(low),
        "high_rate": _arrays(high),
    }


def label_maker(
        dcf_signal: float,
        t_bond_signal: float,
        grid_size: int = 1000,
        eps: float = 1e-6,
) -> pd.Series:
    """
    Input: DCF signal, T_Bond_Rate at valuation date
    Output: pandas.Series with P(alpha > 0 | DCF, regime), E(alpha | DCF, regime)
    Regime is determined by t_bond_signal vs RATE_THRESHOLD.
    """
    train_dir = "Datasets/train/"

    if pd.isna(dcf_signal) or not np.isfinite(dcf_signal):
        return pd.Series(dtype="float64")
    if pd.isna(t_bond_signal) or not np.isfinite(t_bond_signal):
        return pd.Series(dtype="float64")

    regime = "high_rate" if t_bond_signal >= RATE_THRESHOLD else "low_rate"

    params  = _cached_clayton_params(train_dir=train_dir)
    theta   = params[regime]["theta"]
    arrays  = _cached_copula_arrays(train_dir=train_dir)
    alpha_sample, dcf_sample = arrays[regime]

    if len(alpha_sample) == 0 or len(dcf_sample) == 0:
        return pd.Series(dtype="float64")

    v = np.searchsorted(dcf_sample, dcf_signal, side="right") / (len(dcf_sample) + 1.0)
    v = float(np.clip(v, eps, 1 - eps))

    u0 = np.searchsorted(alpha_sample, 0.0, side="right") / (len(alpha_sample) + 1.0)
    u0 = float(np.clip(u0, eps, 1 - eps))

    s0 = (u0 ** (-theta) + v ** (-theta) - 1.0)
    h0 = (v ** (-theta - 1.0)) * (s0 ** (-1.0 / theta - 1.0))
    p_alpha_gt_0 = float(np.clip(1.0 - h0, 0.0, 1.0))

    u_grid = np.linspace(eps, 1 - eps, grid_size)

    probs      = np.arange(1, len(alpha_sample) + 1) / (len(alpha_sample) + 1.0)
    alpha_grid = np.interp(u_grid, probs, alpha_sample)

    log_u = np.log(np.clip(u_grid, 1e-300, None))
    log_v_scalar = np.log(max(v, 1e-300))
    neg_theta_log_u = -theta * log_u
    neg_theta_log_v = -theta * log_v_scalar

    if neg_theta_log_v > 700 or np.any(neg_theta_log_u > 700):
        return pd.Series(dtype="float64")

    pu = np.exp(neg_theta_log_u)
    pv = np.exp(neg_theta_log_v)
    s  = pu + pv - 1.0

    density = np.where(
        s > 0,
        np.exp(
            np.log1p(theta)
            + (-theta - 1.0) * (log_u + log_v_scalar)
            + (-2.0 - 1.0 / theta) * np.log(np.maximum(s, 1e-300))
        ),
        0.0,
    )

    density = np.maximum(density, 0.0)
    mass    = np.trapezoid(density, u_grid)
    if not np.isfinite(mass) or mass <= 0:
        return pd.Series(dtype="float64")

    density /= mass
    expected_alpha = float(np.trapezoid(alpha_grid * density, u_grid))

    return pd.Series({
        "P(alpha > 0)": p_alpha_gt_0,
        "E(alpha)":     expected_alpha,
        "regime":       regime,
    })


# Notes
'''
When |Dcf| falls within the [0.38 – 1.22] range, the probability of the implied share price
and alpha sharing the same sign is >60%, peaking at 70.6%. This confirms that the correlation
between magnitude and sign convergence is strongest at conventional DCF values. Conversely,
accuracy drops to 49% for |Dcf| in [0.21–0.38] and settles into a purely random or inverse
distribution-45% at extremes (|Dcf|>4.49)
Though < 0.38 range can be also applicable due to real_upside-market_upside being negative,
while both signs of implied and real upsides match

Result: we will only use (0 – 1.22] range for copula construction
'''
