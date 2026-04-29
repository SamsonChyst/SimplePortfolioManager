import pandas as pd
import numpy as np
import os
import json

def _load_ticker_df(ticker: str) -> pd.DataFrame:
    return pd.read_parquet(f"Datasets/tickers/{ticker}/timeseries.parquet")


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


def get_market_3y_return(ticker: str, valuation_year: int) -> float | None:
    """
    Input: Ticker and Valuation year
    Output: 3Y CAGR of S&P 500 prior to valuation date
    """
    try:
        df = time_slicing(ticker, valuation_year - 4, valuation_year)
        if df is None or df.empty or "market_price" not in df.columns:
            return None

        df = df[["market_price"]].dropna().sort_index()
        if len(df) < 2:
            return None

        price_end   = df["market_price"].iloc[-1]
        price_start = df["market_price"].iloc[0]

        if price_start <= 0 or not np.isfinite(price_start):
            return None

        years = (df.index[-1] - df.index[0]).days / 365.25
        if years < 1.0:
            return None

        cagr = (price_end / price_start) ** (3.0 / years) - 1.0
        return float(cagr) if np.isfinite(cagr) else None

    except Exception:
        return None

#Was kinda lazy to edit this function for main module so here is the copy
def get_market_3y_return_from_df(df: pd.DataFrame) -> float | None:
    """
    Input: parsed DataFrame with market_price
    Output: 3Y S&P 500 return from already parsed data
    """

    if df is None or df.empty or "market_price" not in df.columns:
        return None

    data = df[["market_price"]].dropna().sort_index()

    if len(data) < 2:
        return None

    price_start = data["market_price"].iloc[0]
    price_end = data["market_price"].iloc[-1]

    if price_start <= 0 or not np.isfinite(price_start):
        return None

    result = price_end / price_start - 1

    return float(result) if np.isfinite(result) else None


#train dataset

def load_train_jsons(train_dir: str = "Datasets/train/") -> pd.DataFrame:
    """
    Input: directory of the train Dataset
    Output: Combines all jsons into a one DataFrame (recursive through ticker folders)
    """
    rows = []
    for root, dirs, files in os.walk(train_dir):
        for fname in files:
            if not fname.lower().endswith(".json"):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)

                rows.append({
                    "ticker":         obj.get("ticker"),
                    "valuation_year": obj.get("valuation_year"),
                    "alpha_horizon":  obj.get("alpha_horizon"),
                    "implied_upside": obj.get("implied_upside"),
                    "real_alpha":     obj.get("real_alpha"),
                    "t_bond_rate":    obj.get("t_bond_rate"),
                })
            except Exception as e:
                print(f"Failed to read {path}: {e}")

    df = pd.DataFrame(rows)
    return df

#main

def live_market_features(
        ticker: str,
        t_bond_rate: float,
        benchmark: str = "^GSPC",
) -> pd.Series:
    """
    Input: ticker and current T-Bond rate
    Output: one live inference row with market ML features for a year's worth
    """

    from parser import (
        get_price,
        load_market_returns,
        compute_returns,
        compute_beta,
        compute_volatility,
        compute_log_volume,
        compute_market_regime_features,
    )

    ticker = str(ticker).upper().strip()

    today = pd.Timestamp.today().normalize()
    end = today + pd.Timedelta(days=1)

    one_year_ago = today - pd.DateOffset(years=1)
    three_years_ago = today - pd.DateOffset(years=3)

    start_str = three_years_ago.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    market_df = load_market_returns(start=start_str, end=end_str, benchmark=benchmark)
    df = get_price(ticker=ticker, start=start_str, end=end_str)

    if df is None or df.empty:
        return pd.Series(dtype="float64")

    df = df.join(market_df, how="left")

    df = compute_returns(df)
    df = compute_beta(df)
    df = compute_volatility(df)
    df = compute_log_volume(df)
    df = compute_market_regime_features(df)

    window_1y = df.loc[(df.index >= one_year_ago) & (df.index <= today)].copy()

    if window_1y.empty:
        return pd.Series(dtype="float64")

    beta = pd.to_numeric(window_1y["beta_ewm"], errors="coerce")
    volatility = pd.to_numeric(window_1y["volatility_21d"], errors="coerce")
    log_volume = pd.to_numeric(window_1y["log_volume"], errors="coerce")
    market_deviation = pd.to_numeric(window_1y["market_deviation"], errors="coerce")
    market_momentum = pd.to_numeric(window_1y["market_momentum"], errors="coerce")

    return pd.Series({
        "ticker": ticker,
        "t_bond_rate": float(t_bond_rate),
        "beta_ewm_median": float(beta.median()),
        "volatility_21d_mean": float(volatility.mean()),
        "volatility_21d_max": float(volatility.max()),
        "log_volume_mean": float(log_volume.mean()),
        "market_deviation_mean": float(market_deviation.mean()),
        "market_deviation_std": float(market_deviation.std()),
        "market_momentum_mean": float(market_momentum.mean()),
        "market_momentum_std": float(market_momentum.std()),
        "market_3y_return": get_market_3y_return_from_df(df),
    })