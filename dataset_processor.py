import pandas as pd
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


def load_train_parquets(
    train_dir: str = "Datasets/train/",
    drop_index: bool = True,
    add_ticker: bool = False,
) -> pd.DataFrame:
    """
    Input: Directory of the train Dataset with parquet files (recursive through ticker folders)
    Output: Combines all parquet datasets into one DataFrame
    """
    dfs = []

    for root, dirs, files in os.walk(train_dir):
        for fname in files:
            if not fname.lower().endswith(".parquet"):
                continue

            path = os.path.join(root, fname)

            try:
                df = pd.read_parquet(path)

                if add_ticker:
                    ticker = os.path.basename(root)
                    df["ticker"] = ticker

                if drop_index:
                    df = df.reset_index(drop=True)
                else:
                    df = df.reset_index()

                dfs.append(df)

            except Exception:
                continue
    if len(dfs) == 0:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)