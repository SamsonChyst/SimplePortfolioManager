import pandas as pd
import json
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass

#cfg

ds = os.listdir("Datasets/tickers/") #Directory with parquets for every TICKER

INPUT_CSV = Path("Datasets/US_WaccComponents_Timeseries.csv") #For US Market
#Source: Risk-free rates and equity premium Stern-Damodaran,
#Marginal tax rates - IRS, Annual AVG Credit Spreads - BBB OAS


#DataFrame processing

@dataclass(frozen=True)
class ValuationKey:
    ticker: str
    valuation_year: int


def _load_ticker_df(ticker: str) -> pd.DataFrame:
    return pd.read_parquet(f"Datasets/tickers/{ticker}/timeseries.parquet")


def time_slicing(
        ticker: str,
        start_year: int,
        end_year: int,
        shift: int = 5,
) -> pd.DataFrame:
    """
    Input: Valid Ticker existing in Dataset, Start & End year, Shift-days after filing date
    Output: A sliced DataFrame by real date
    """
    if shift < 0:
        raise ValueError("shift must be >= 0")

    df = _load_ticker_df(ticker).copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    if "filing_date" not in df.columns:
        raise ValueError("DataFrame must contain 'filing_date' column")

    df = df.sort_index()
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")

    def get_boundary(year: int) -> pd.Timestamp | None:
        base_date = pd.Timestamp(year=year, month=1, day=1)

        candidates = df.loc[df.index >= base_date]
        if candidates.empty:
            return None

        first_row = candidates.iloc[0]
        filing_date = first_row["filing_date"]

        if pd.isna(filing_date):
            return None

        min_valid_date = filing_date + pd.Timedelta(days=shift)

        valid = candidates.loc[candidates.index >= min_valid_date]
        if valid.empty:
            return None

        return valid.index[0]

    start_date = get_boundary(start_year)
    end_exclusive = get_boundary(end_year + 1)

    if start_date is None:
        return pd.DataFrame()

    if end_exclusive is None:
        return df.loc[start_date:].copy()

    return df.loc[(df.index >= start_date) & (df.index < end_exclusive)].copy()


def year_slicing(ticker: str, start_year: int, end_year: int, shift: int = 5) -> pd.DataFrame:
    """
    Input: ticker, start and end decision years, shift-days after filing date
    Output: one row per year at the first valid real date
    """
    if shift < 0:
        raise ValueError("shift must be >= 0")

    df = _load_ticker_df(ticker).copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")

    if "filing_date" not in df.columns:
        raise ValueError("DataFrame must contain 'filing_date' column")

    df = df.sort_index()
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")

    rows = []

    for year in range(start_year, end_year + 1):
        base_date = pd.Timestamp(year=year, month=1, day=1)

        year_slice = df[df.index >= base_date]
        if year_slice.empty:
            continue

        first_row = year_slice.iloc[0]
        filing_date = first_row["filing_date"]

        if pd.isna(filing_date):
            continue

        min_valid_date = filing_date + pd.Timedelta(days=shift)

        if first_row.name >= min_valid_date:
            chosen = first_row.copy()
        else:
            valid_rows = year_slice[year_slice.index >= min_valid_date]
            if valid_rows.empty:
                continue
            chosen = valid_rows.iloc[0].copy()

        chosen["decision_year"] = year
        chosen["decision_date"] = chosen.name
        rows.append(chosen)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def has_fundamentals(ticker: str) -> bool: #1700+ as of last input dataset
    """
    Input: Ticker
    Output: Whether meta.json says has_fundamentals == True
    """
    df_path = f"Datasets/tickers/{ticker}/meta.json"

    if not os.path.exists(df_path):
        return False
    with open(df_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    return bool(meta_data.get("has_fundamentals", False))


#Valuation

def dcf_proxy(
        df: pd.DataFrame,
        date_col: str = "Date",
        net_debt_radius: float = 0.03,
) -> pd.Series:
    """
    Input: sliced timeseries DataFrame
    Output: returns 5th historical year FCFF, FCFF CAGR, Y5 price, Y6 price, and Y5 net debt
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
    if len(data) < 6:
        return pd.Series(dtype="float64")

    for col in [
        "price", "ev", "fcff", "net_debt", "total_debt",
        "long_term_debt", "short_term_debt", "market_cap",
        "market_price", "cash", "equity", "shares_yf"
    ]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        else:
            data[col] = np.nan

    implied_net_debt = data["ev"] - data["market_cap"]
    implied_scale = pd.concat(
        [
            data["ev"].abs(),
            data["market_cap"].abs(),
            pd.Series(1.0, index=data.index)
        ],
        axis=1
    ).max(axis=1)
    implied_ratio = implied_net_debt.abs() / implied_scale
    implied_net_debt = implied_net_debt.where(implied_ratio > net_debt_radius, 0.0)

    has_debt_parts = data["long_term_debt"].notna() | data["short_term_debt"].notna()
    debt_parts = data["long_term_debt"].fillna(0.0) + data["short_term_debt"].fillna(0.0)
    debt_parts = debt_parts.where(has_debt_parts, np.nan)

    fallback_balance = debt_parts - data["cash"]

    data["total_debt"] = data["total_debt"].where(data["total_debt"].notna(), debt_parts)
    data["net_debt"] = data["net_debt"].where(pd.notna(data["net_debt"]), data["total_debt"] - data["cash"])
    data["net_debt"] = data["net_debt"].where(pd.notna(data["net_debt"]), fallback_balance)
    data["net_debt"] = data["net_debt"].where(pd.notna(data["net_debt"]), implied_net_debt)

    hist = data.iloc[:5].copy()
    comp = data.iloc[5]

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
    price_6 = comp["price"]
    net_debt_5 = y5["net_debt"]
    total_debt_5 = y5["total_debt"]
    market_cap_5 = y5["market_cap"]
    market_price_5 = y5["market_price"]
    market_price_6 = comp["market_price"]

    fcff_cagr = np.nan
    fcff_hist = hist[[date_col, "fcff"]].dropna().copy()
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

            growth = (curr_fcff / prev_fcff) ** (1 / year_gap) - 1
            if pd.notna(growth) and np.isfinite(growth):
                yoy_growth.append(growth)

        if yoy_growth:
            fcff_cagr = float(np.median(yoy_growth))
        else:
            x = (fcff_hist[date_col] - fcff_hist[date_col].min()).dt.days / 365.25
            y = np.log(fcff_hist["fcff"].values)
            if len(x) >= 2:
                slope = np.polyfit(x - x.mean(), y, 1)[0]
                fcff_cagr = np.exp(slope) - 1

    if pd.notna(fcff_cagr):
        fcff_cagr = 0.15 * np.tanh(fcff_cagr / 0.15)

    real_price_change = np.nan
    if pd.notna(price_5) and pd.notna(price_6) and np.isfinite(price_5) and price_5 != 0:
        real_price_change = (price_6 / price_5) - 1

    real_market_price_change = np.nan
    if pd.notna(market_price_5) and pd.notna(market_price_6) and np.isfinite(market_price_5) and market_price_5 != 0:
        real_market_price_change = (market_price_6 / market_price_5) - 1

    return pd.Series({
        "fcff": fcff_5,
        "fcff_cagr": fcff_cagr,
        "shares_5": shares_5,
        "price_5": price_5,
        "price_6": price_6,
        "real_price_change": real_price_change,
        "net_debt_5": net_debt_5,
        "market_price_5": market_price_5,
        "market_price_6": market_price_6,
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

    WaccDf = pd.read_csv(INPUT_CSV, sep=";", header=0, index_col="Year").loc[key.valuation_year, :]
    DailyDf = time_slicing(key.ticker, key.valuation_year - 4, key.valuation_year)
    YearlySeries = dcf_proxy(year_slicing(key.ticker, key.valuation_year - 4, key.valuation_year + 1))

    beta = float(np.median(DailyDf["beta_ewm"].dropna()))
    risk_free = float(WaccDf["T_Bond_Rate"])
    equity_premium = float(WaccDf["Implied_ERP_(FCFE)"])
    cost_of_capital = (risk_free + beta * equity_premium)

    tax_shield = 1 - float(WaccDf["US_Marginal_Tax_Rate"])

    total_capital = YearlySeries["market_cap_5"] + YearlySeries["total_debt_5"]
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
) -> pd.Series:
    """
    Input: ticker and valuation year
    Output: FCFF DCF valuation with Enterprise Value, Equity Value and value per share
    """
    key = ValuationKey(ticker=ticker, valuation_year=valuation_year)

    yearly_df = year_slicing(
        ticker=key.ticker,
        start_year=key.valuation_year - 4,
        end_year=key.valuation_year + 1,
        shift=shift,
    )
    proxy = dcf_proxy(yearly_df)

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

    if terminal_growth_rate >= wacc:
        terminal_growth_rate = min(growth_rate / 2, wacc - 0.01)

    if terminal_growth_rate >= wacc:
        return pd.Series(dtype="float64")

    fcff_forecast = []
    pv_fcff = []

    for t in range(1, forecast_years + 1):
        fcff_t = fcff_0 * ((1 + growth_rate) ** t)
        pv_t = fcff_t / ((1 + wacc) ** t)
        fcff_forecast.append(fcff_t)
        pv_fcff.append(pv_t)

    fcff_terminal_base = fcff_forecast[-1]
    terminal_value = (fcff_terminal_base * (1 + terminal_growth_rate)) / (wacc - terminal_growth_rate)
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

#MAYBE I SHOULD CREATE 2 MARKERS 1-DCF SIGN(VECTOR GROWTH/DECLINE) 2-REAL ALPHA AND NORM BOTH OF THEM
#AFTER ML I SHOULD FILTER OUTCOMES TO MY TASTE CONSIDERING BOTH AXIS
