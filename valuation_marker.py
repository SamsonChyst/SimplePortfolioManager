import pandas as pd
import json
import os
import numpy as np
from calendar import monthrange


#cfg
ds = os.listdir("Datasets/tickers/")

#DataFrame processing

def time_slicing(ticker: str, start_year: int, end_year: int, shift: int = 31) -> pd.DataFrame:
    """
    Input: DataFrame to process, Start & End year, Shift-days after the beginning of the year
    shift in between [1-31], years between [2010-2025]
    Output: A sliced DataFrame by real date
    """
    if not (1 <= shift <= 31):
        raise ValueError("shift must be in [1, 31]")

    start_year = str(start_year)
    end_year = str(end_year)

    df = pd.read_parquet(f"Datasets/tickers/{ticker}/timeseries.parquet")
    start = pd.Timestamp(f"{start_year}-01-{shift:02d}")
    end = pd.Timestamp(f"{end_year}-01-{shift:02d}")

    return df.loc[(df.index >= start) & (df.index <= end)].copy()

def has_fundamentals(ticker: str) -> bool:
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
        n_years: int = 6,
        min_fund_years: int = 4,
        shift_month: int = 1,
        shift_day: int = 31,
        forward_days: int = 5,
        positive_multiples_only: bool = True,
        use_filing_date: bool = True,
        treat_missing_debt_cash_as_zero: bool = True,
) -> pd.DataFrame:
    """
    Input: sliced timeseries DataFrame
    Output: Annualizes fundamentals into single values, computes market multiples
    If < 4 history years of fundamentals - return None
    """
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    if date_col in data.columns:
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    elif isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index().rename(columns={data.index.name or "index": date_col})
        data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    else:
        raise ValueError(f"DataFrame must contain '{date_col}' column or DatetimeIndex")

    if "period_end" not in data.columns:
        raise ValueError("DataFrame must contain 'period_end' column")

    data["period_end"] = pd.to_datetime(data["period_end"], errors="coerce")
    if "filing_date" in data.columns:
        data["filing_date"] = pd.to_datetime(data["filing_date"], errors="coerce")
    else:
        data["filing_date"] = pd.NaT

    data = data.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    if data.empty:
        return pd.DataFrame()

    numeric_cols = [
        "price", "volume", "return_market", "ret", "beta_ewm", "volatility_21d",
        "log_volume", "revenue", "ebit", "net_income", "assets", "equity",
        "debt", "cash", "capex", "ocf", "shares_yf", "interest", "market_cap", "ev"
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        else:
            data[col] = np.nan

    for col in ["ticker", "industry"]:
        if col not in data.columns:
            data[col] = np.nan

    def make_anchor(year: int, month: int, day: int) -> pd.Timestamp:
        last_day = monthrange(year, month)[1]
        return pd.Timestamp(year=year, month=month, day=min(day, last_day))

    def valid_den(x):
        if pd.isna(x):
            return False
        return x > 0 if positive_multiples_only else x != 0

    fund_table = data.dropna(subset=["period_end"]).copy()
    if fund_table.empty:
        return pd.DataFrame()

    fund_table["fund_year"] = fund_table["period_end"].dt.year

    fund_table = (
        fund_table
        .sort_values(["fund_year", "filing_date", "period_end", date_col])
        .drop_duplicates(subset=["fund_year"], keep="last")
        .sort_values("fund_year")
        .reset_index(drop=True)
    )

    if use_filing_date and fund_table["filing_date"].notna().any():
        fund_table["available_date"] = fund_table["filing_date"]
    else:
        fund_table["available_date"] = fund_table["period_end"]

    start_year = data[date_col].min().year
    rows = []

    for i in range(n_years):
        decision_year = start_year + i
        decision_date = make_anchor(decision_year, shift_month, shift_day)

        after_decision = data[data[date_col] >= decision_date].copy()
        if after_decision.empty:
            continue

        market_idx = min(forward_days - 1, len(after_decision) - 1)
        market_row = after_decision.iloc[market_idx].copy()
        market_date = market_row[date_col]

        eligible_fund = fund_table[fund_table["available_date"] <= market_date].copy()

        if eligible_fund.empty:
            fund_row = None
        else:
            fund_row = eligible_fund.iloc[-1].copy()

        out = {
            "market_date": market_date,
            "ticker": market_row.get("ticker", np.nan),
            "industry": market_row.get("industry", np.nan),
            "price": market_row.get("price", np.nan),
            "volume": market_row.get("volume", np.nan),
            "return_market": market_row.get("return_market", np.nan),
            "ret": market_row.get("ret", np.nan),
            "beta_ewm": market_row.get("beta_ewm", np.nan),
            "volatility_21d": market_row.get("volatility_21d", np.nan),
            "log_volume": market_row.get("log_volume", np.nan),
            "shares_yf": market_row.get("shares_yf", np.nan),
            "market_cap": market_row.get("market_cap", np.nan),
            "ev": market_row.get("ev", np.nan),
        }

        fund_cols = [
            "filing_date", "period_end", "fund_year",
            "revenue", "ebit", "net_income", "assets", "equity",
            "debt", "cash", "capex", "ocf", "interest"
        ]

        if fund_row is None:
            for col in fund_cols:
                out[col] = np.nan
        else:
            for col in fund_cols:
                out[col] = fund_row.get(col, np.nan)

        price = out["price"]
        shares_yf = out["shares_yf"]
        debt = out["debt"]
        cash = out["cash"]
        ebit = out["ebit"]
        revenue = out["revenue"]
        ocf = out["ocf"]
        net_income = out["net_income"]
        equity = out["equity"]
        market_cap = out["market_cap"]
        ev = out["ev"]

        if pd.isna(market_cap) and pd.notna(price) and pd.notna(shares_yf):
            market_cap = price * shares_yf

        debt_for_ev = 0.0 if (treat_missing_debt_cash_as_zero and pd.isna(debt)) else debt
        cash_for_ev = 0.0 if (treat_missing_debt_cash_as_zero and pd.isna(cash)) else cash

        if pd.isna(ev) and pd.notna(market_cap) and pd.notna(debt_for_ev) and pd.notna(cash_for_ev):
            ev = market_cap + debt_for_ev - cash_for_ev

        pb = np.nan
        if pd.notna(market_cap) and valid_den(equity):
            pb = market_cap / equity

        pe = np.nan
        if pd.notna(market_cap) and valid_den(net_income):
            pe = market_cap / net_income

        ev_ebit = np.nan
        if pd.notna(ev) and valid_den(ebit):
            ev_ebit = ev / ebit

        ev_revenue = np.nan
        if pd.notna(ev) and valid_den(revenue):
            ev_revenue = ev / revenue

        ev_ocf = np.nan
        if pd.notna(ev) and valid_den(ocf):
            ev_ocf = ev / ocf

        out["market_cap"] = market_cap
        out["ev"] = ev
        out["ev_ebit"] = ev_ebit
        out["ev_revenue"] = ev_revenue
        out["ev_ocf"] = ev_ocf
        out["pe"] = pe
        out["pb"] = pb

        rows.append(out)

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame()

    hist_result = result.iloc[: max(0, n_years - 1)].copy()
    found_fund_years = hist_result["fund_year"].dropna().nunique()

    if found_fund_years < min_fund_years:
        return pd.DataFrame()

    return result.reset_index(drop=True)

def normalize_multiples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: DataFrame after dcf_proxy
    Output: Normalizes market multiples by the caps and log's
    """
    df = df.copy()
    caps = {
        "pe": 100,
        "pb": 20,
        "ev_ebit": 50,
        "ev_revenue": 20,
        "ev_ocf": 50,
    }
    for col, cap in caps.items():
        if col in df.columns:
            df[col] = df[col].clip(0, cap)
            df[col + "_log"] = np.log1p(df[col])
    return df

def fcff_CAGR(
        df: pd.DataFrame,
        year_col: str = "fund_year",
        ocf_col: str = "ocf",
        capex_col: str = "capex",
) -> float:
    if df is None or df.empty:
        return np.nan

    data = df.copy()

    required_cols = [year_col, ocf_col, capex_col]
    for col in required_cols:
        if col not in data.columns:
            return np.nan

    for col in required_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=[year_col]).sort_values(year_col).reset_index(drop=True)
    if len(data) < 5:
        return np.nan

    data["fcff"] = data[ocf_col] - data[capex_col]
    data["fcff"] = data["fcff"].replace([np.inf, -np.inf], np.nan)

    first_five = data.head(5)

    if first_five["fcff"].isna().any():
        return np.nan

    if (first_five["fcff"] <= 0).any():
        return np.nan

    ratios = first_five["fcff"].iloc[1:].to_numpy() / first_five["fcff"].iloc[:-1].to_numpy()

    if not np.isfinite(ratios).all():
        return np.nan

    return np.prod(ratios) ** (1 / len(ratios)) - 1