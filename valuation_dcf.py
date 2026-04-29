from pathlib import Path
from dataclasses import dataclass
from modules_processor import *
import numpy as np
import pandas as pd

#cfg
INPUT_CSV = Path("Datasets/US_WaccComponents_Timeseries.csv") #For US Market
#Source: Risk-free rates and equity premium Stern-Damodaran,
#Marginal tax rates - IRS, Annual AVG Credit Spreads - BBB OAS

_WACC_TABLE = pd.read_csv(INPUT_CSV, sep=";", header=0, index_col="Year")


@dataclass(frozen=True)
class ValuationKey:
    ticker: str
    valuation_year: int

#Valuation

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

    fcff_5         = y5["fcff"]
    shares_5       = y5["shares_yf"]
    price_5        = y5["price"]
    net_debt_5     = y5["net_debt"]
    total_debt_5   = y5["total_debt"]
    market_cap_5   = y5["market_cap"]
    market_price_5 = y5["market_price"]
    price_comp        = comp["price"]
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
        "fcff":                    fcff_5,
        "fcff_cagr":               fcff_cagr,
        "shares_5":                shares_5,
        "price_5":                 price_5,
        "price_comp":              price_comp,
        "real_price_change":       real_price_change,
        "net_debt_5":              net_debt_5,
        "market_price_5":          market_price_5,
        "market_price_comp":       market_price_comp,
        "real_market_price_change": real_market_price_change,
        "total_debt_5":            total_debt_5,
        "market_cap_5":            market_cap_5,
    })


def wacc_proxy(ticker: str, valuation_year: int) -> float:
    """
    Input: Unsliced daily DataFrame, final year of valuation
    Output: Weighted Average Cost of Capital for a Company at a given moment
    """
    key = ValuationKey(ticker=ticker, valuation_year=valuation_year)

    if key.valuation_year not in _WACC_TABLE.index:
        return np.nan

    WaccDf      = _WACC_TABLE.loc[key.valuation_year, :]
    DailyDf     = time_slicing(key.ticker, key.valuation_year - 4, key.valuation_year)
    YearlySeries = dcf_proxy(year_slicing(key.ticker, key.valuation_year - 4, key.valuation_year + 1))

    if DailyDf is None or DailyDf.empty:
        return np.nan
    if "beta_ewm" not in DailyDf.columns:
        return np.nan

    beta_series = pd.to_numeric(DailyDf["beta_ewm"], errors="coerce").dropna()
    if beta_series.empty:
        return np.nan

    beta = float(beta_series.median())

    risk_free      = float(WaccDf["T_Bond_Rate"])
    equity_premium = float(WaccDf["Implied_ERP_(FCFE)"])
    cost_of_capital = risk_free + beta * equity_premium
    tax_shield      = 1 - float(WaccDf["US_Marginal_Tax_Rate"])

    total_capital = YearlySeries["market_cap_5"] + YearlySeries["total_debt_5"]
    if pd.isna(total_capital) or total_capital <= 0:
        return np.nan

    equity_in_structure = YearlySeries["market_cap_5"] / total_capital
    debt_in_structure   = YearlySeries["total_debt_5"]  / total_capital
    credit_spread       = float(WaccDf["Credit_Spread"])
    cost_of_debt        = risk_free + credit_spread

    return (equity_in_structure * cost_of_capital) + (debt_in_structure * cost_of_debt * tax_shield)

def dcf_valuation(
        ticker: str,
        valuation_year: int,
        forecast_years: int = 5,
        growth_rate: float = 0.0275,
        terminal_growth_rate: float = 0.01375,
        shift: int = 5,
        alpha_horizon: int = 1,
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

    fcff_0                   = pd.to_numeric(proxy.get("fcff"),                errors="coerce")
    net_debt                 = pd.to_numeric(proxy.get("net_debt_5"),           errors="coerce")
    shares                   = pd.to_numeric(proxy.get("shares_5"),             errors="coerce")
    price_5                  = pd.to_numeric(proxy.get("price_5"),              errors="coerce")
    real_price_change        = pd.to_numeric(proxy.get("real_price_change"),    errors="coerce")
    real_market_price_change = pd.to_numeric(proxy.get("real_market_price_change"), errors="coerce")

    if pd.isna(fcff_0) or not np.isfinite(fcff_0):
        return pd.Series(dtype="float64")

    wacc = float(wacc_proxy(key.ticker, key.valuation_year))
    if not np.isfinite(wacc):
        return pd.Series(dtype="float64")

    spread = wacc - terminal_growth_rate
    if not np.isfinite(spread) or spread <= min_wacc_terminal_spread:
        return pd.Series(dtype="float64")

    t              = np.arange(1, forecast_years + 1, dtype=float)
    fcff_forecast  = fcff_0 * ((1 + growth_rate) ** t)
    pv_fcff        = fcff_forecast / ((1 + wacc) ** t)

    fcff_terminal_base = fcff_forecast[-1]
    terminal_value     = (fcff_terminal_base * (1 + terminal_growth_rate)) / spread
    pv_terminal_value  = terminal_value / ((1 + wacc) ** forecast_years)

    enterprise_value = float(np.sum(pv_fcff) + pv_terminal_value)
    equity_value     = enterprise_value - net_debt if pd.notna(net_debt) else np.nan

    value_per_share = np.nan
    if pd.notna(shares) and np.isfinite(shares) and shares != 0:
        value_per_share = equity_value / shares

    implied_upside_vs_price_5 = np.nan
    if pd.notna(price_5) and np.isfinite(price_5) and price_5 != 0 and pd.notna(value_per_share):
        implied_upside_vs_price_5 = (value_per_share / price_5) - 1

    return pd.Series({
        "ticker":               key.ticker,
        "valuation_year":       key.valuation_year,
        "alpha_horizon":        alpha_horizon,
        "wacc":                 wacc,
        "growth_rate":          growth_rate,
        "terminal_growth_rate": terminal_growth_rate,
        "terminal_value":       float(terminal_value),
        "pv_terminal_value":    float(pv_terminal_value),
        "enterprise_value":     enterprise_value,
        "net_debt":             net_debt,
        "equity_value":         equity_value,
        "value_per_share":      value_per_share,
        "real_upside":          real_price_change,
        "market_upside":        real_market_price_change,
        "implied_upside":       implied_upside_vs_price_5,
    })


def target_row(
        ticker: str,
        valuation_year: int,
        forecast_years: int = 5,
        growth_rate: float = 0.0275,
        terminal_growth_rate: float = 0.01375,
        shift: int = 5,
        alpha_horizon: int = 1,
        alpha_clip: float = 2.0,
) -> pd.Series:
    """
    Input: ticker and valuation year
    Output: one-row target observation for dataset construction
    alpha_clip=2.0 caps one-year excess returns at ±200% to suppress outliers
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
    real_upside    = pd.to_numeric(val.get("real_upside"),    errors="coerce")
    market_upside  = pd.to_numeric(val.get("market_upside"),  errors="coerce")

    if pd.isna(implied_upside) or not np.isfinite(implied_upside):
        return pd.Series(dtype="float64")
    if pd.isna(real_upside) or not np.isfinite(real_upside):
        return pd.Series(dtype="float64")
    if pd.isna(market_upside) or not np.isfinite(market_upside):
        return pd.Series(dtype="float64")

    real_alpha = real_upside - market_upside

    if pd.isna(real_alpha) or not np.isfinite(real_alpha):
        return pd.Series(dtype="float64")
    if abs(real_alpha) > alpha_clip:
        return pd.Series(dtype="float64")

    return pd.Series({
        "ticker":         ticker,
        "valuation_year": int(valuation_year),
        "alpha_horizon":  int(alpha_horizon),
        "real_alpha":     float(real_alpha),
        "implied_upside": float(implied_upside),
    })

#ML Label is copula-based upon Bayesian P(real_alpha | DCF_signal, alpha_hat, T-Bond_rate)
#Copula is fitted exclusively on high-rate regime (T_Bond >= 3.0%)
#where DCF signal has demonstrated meaningful predictive power (tau = 0.172)

#Notes
'''
When |Dcf| falls within the [0.38 - 1.22] range, the probability of the implied share price
and alpha sharing the same sign is >60%, peaking at 70.6%. This confirms that the correlation
between magnitude and sign convergence is strongest at conventional DCF values. Conversely,
accuracy drops to 49% for |Dcf| in [0.21-0.38] and settles into a purely random or inverse
distribution - 45% at extremes (|Dcf| > 4.49).
Though the < 0.38 range can also be applicable due to real_upside - market_upside being
negative while both signs of implied and real upsides match.

Result: only the (0 - 1.22] range is used for copula construction.
Note: the copula is trained exclusively on high-rate observations (T_Bond >= 3.0%) where
Kendall tau between implied_upside and real_alpha is 0.172 vs 0.057 in the low-rate regime.
Alpha horizon is one year. Alpha is capped at +-200% to suppress outliers.
'''