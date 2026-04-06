import pandas as pd
import json
import os
import numpy as np
from pathlib import Path

#cfg

ds = os.listdir("Datasets/tickers/") #Directory of parquets for every TICKER

INPUT_CSV = Path("Datasets/US_WaccComponents_Timeseries.csv") #Source: Stern-Damodaran & IRS
wacc_df = pd.read_csv(INPUT_CSV, sep = ";", header = 0, index_col = "Year")

#DataFrame processing

def time_slicing(
        ticker: str,
        start_year: int,
        end_year: int,
        shift: int = 5,
) -> pd.DataFrame:
    """
    Input: DataFrame to process, Start & End year, Shift-days after filing date
    Output: A sliced DataFrame by real date
    """
    if shift < 0:
        raise ValueError("shift must be >= 0")

    df = pd.read_parquet(f"Datasets/tickers/{ticker}/timeseries.parquet").copy()

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

    df = pd.read_parquet(f"Datasets/tickers/{ticker}/timeseries.parquet").copy()

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
        positive_multiples_only: bool = True,
) -> pd.Series:
    """
    Input: sliced timeseries DataFrame
    Output: returns 5th historical year FCFF, FCFF CAGR, EV/EBIT, Y5 price, Y6 price, and Y5 net debt
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

    for col in ["price", "ev", "ebit", "fcff", "net_debt"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        else:
            data[col] = np.nan

    fcff_1 = data.iloc[0]["fcff"]
    fcff_5 = data.iloc[4]["fcff"]
    ebit_5 = data.iloc[4]["ebit"]
    ev_5 = data.iloc[4]["ev"]
    price_5 = data.iloc[4]["price"]
    price_6 = data.iloc[5]["price"]
    net_debt_5 = data.iloc[4]["net_debt"]
    total_debt_5 = data.iloc[4]["total_debt"]
    market_cap_5 = data.iloc[4]["market_cap"]
    market_price_5 = data.iloc[4]["market_price"]
    market_price_6 = data.iloc[5]["market_price"]

    fcff_cagr = np.nan
    if pd.notna(fcff_1) and pd.notna(fcff_5) and fcff_1 > 0 and fcff_5 > 0:
        fcff_cagr = (fcff_5 / fcff_1) ** (1 / 4) - 1

    ev_ebit = np.nan
    if pd.notna(ev_5) and pd.notna(ebit_5):
        if positive_multiples_only:
            if ebit_5 > 0:
                ev_ebit = ev_5 / ebit_5
        elif ebit_5 != 0:
            ev_ebit = ev_5 / ebit_5

    return pd.Series({
        "fcff": fcff_5,
        "fcff_cagr": fcff_cagr,
        "ev_ebit": ev_ebit,
        "price_5": price_5,
        "price_6": price_6,
        "net_debt_5": net_debt_5,
        "market_price_5": market_price_5,
        "market_price_6": market_price_6,
        "total_debt_5": total_debt_5,
        "market_cap_5": market_cap_5
    })

def wacc_proxy(ticker: str, valuation_year: int) -> float:
    """
    Input: Unsliced daily DataFrame, final year of valuation
    Output: Weighted Average Cost of Capital for a Company at a given moment
    """
    DailyDf = time_slicing(ticker, valuation_year - 4, valuation_year)
    YearlySeries = year_slicing(ticker, valuation_year - 4, valuation_year)

    #Cost of Capital Components - CAPM model
    beta = np.median(DailyDf["beta_ewm"])
    risk_free = wacc_df[valuation_year, "T_Bond_Rate"]
    equity_premium = wacc_df[valuation_year, "Implied_ERP_(FCFE)"]

    CostOfCapital = (risk_free + beta * equity_premium)
    tax_rate = wacc_df[valuation_year, "US_Marginal_Tax_Rate"]


print(wacc_df)