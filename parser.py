import os
import json
import time
import threading
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed


#cfg
load_dotenv()

#change for new datasets after training
INPUT_CSV = Path("Datasets/companies.csv") #Source: Kaggle US Stocks & ETFs - Tickers, Company Info, Logos
OUTPUT_DIR = Path("Datasets/tickers") #2787 valid tickers

#DataFrame slicing
START_DATE = "2016-01-01"
END_DATE = "2026-03-31"

#Data Fetching
FETCH_START_DATE = "2015-01-01"
FETCH_END_DATE = "2027-01-01"

MAX_WORKERS = 4
SEC_DELAY = 0.2
FORCE_REBUILD = False

USER_AGENT = os.getenv("Mail")
if not USER_AGENT:
    raise ValueError("Set Mail in .env for SEC User-Agent, e.g. Mail=your_email@example.com")

ANNUAL_FORMS = {"10-K", "10-K/A"} #for parsing SEC fundamentals

#DataFrame reader

def safe_str(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s else None

def load_companies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["ticker", "industry"])

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["industry"] = df["industry"].map(safe_str)

    df = df[df["ticker"].notna() & (df["ticker"] != "")]
    df = df.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)

    return df

#Path processor

def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=True)
    tmp.replace(path)


def build_metadata(ticker: str, industry: str | None, df: pd.DataFrame, has_fundamentals: bool) -> dict:
    return {
        "ticker": ticker,
        "industry": industry,
        "rows": int(len(df)),
        "start_date": str(df.index.min().date()) if not df.empty else None,
        "end_date": str(df.index.max().date()) if not df.empty else None,
        "has_fundamentals": bool(has_fundamentals),
        "columns": list(df.columns),
    }

#Market parse agent *

def get_price(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Input: A TICKER and a time-horizon for parse
    Output: a Market DataFrame with a Date, Closing Price, Volume columns
    """
    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        needed = {"Close", "Volume"}
        if not needed.issubset(set(data.columns)):
            return pd.DataFrame()

        df = data[["Close", "Volume"]].copy()
        df.columns = ["price", "volume"]

        idx = pd.to_datetime(df.index, errors="coerce")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)

        df.index = pd.DatetimeIndex(idx).astype("datetime64[ns]")
        df.index.name = "Date"
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df.dropna(subset=["price"])

        return df

    except Exception:
        return pd.DataFrame()


def get_yf_shares_series(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        tk = yf.Ticker(ticker)
        shares = tk.get_shares_full(start=start, end=end)

        if shares is None or len(shares) == 0:
            return pd.DataFrame()

        if isinstance(shares, pd.Series):
            df = shares.to_frame(name="shares_yf")
        else:
            df = pd.DataFrame(shares)
            if df.shape[1] == 1:
                df.columns = ["shares_yf"]
            elif "shares_out" in df.columns:
                df = df.rename(columns={"shares_out": "shares_yf"})
            elif "Shares" in df.columns:
                df = df.rename(columns={"Shares": "shares_yf"})
            else:
                df = df.rename(columns={df.columns[0]: "shares_yf"})

        idx = pd.to_datetime(df.index, errors="coerce")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)

        df.index = pd.DatetimeIndex(idx).astype("datetime64[ns]")
        df.index.name = "Date"
        df["shares_yf"] = pd.to_numeric(df["shares_yf"], errors="coerce")
        df = df.dropna(subset=["shares_yf"]).sort_index()
        df = df[~df.index.duplicated(keep="last")]

        return df[["shares_yf"]]

    except Exception:
        return pd.DataFrame()


def merge_shares_asof(price_df: pd.DataFrame, shares_df: pd.DataFrame) -> pd.DataFrame:
    if shares_df is None or shares_df.empty:
        return price_df

    left = price_df.reset_index().rename(columns={"Date": "date"})
    right = shares_df.reset_index().rename(columns={"Date": "shares_date"})

    left["date"] = pd.to_datetime(left["date"], errors="coerce").astype("datetime64[ns]")
    right["shares_date"] = pd.to_datetime(right["shares_date"], errors="coerce").astype("datetime64[ns]")

    left = left.dropna(subset=["date"]).sort_values("date")
    right = right.dropna(subset=["shares_date"]).sort_values("shares_date")

    merged = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="shares_date",
        direction="backward",
        tolerance=pd.Timedelta(days=550),
    )

    merged = merged.drop(columns=["shares_date"], errors="ignore")
    merged = merged.set_index("date")
    merged.index.name = "Date"
    return merged


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: Market DataFrame with 'returns' column
    Output: Computes Log-returns for DataFrame
    """
    df = df.copy()
    df["ret"] = np.log(df["price"] / df["price"].shift(1))
    df["ret"] = df["ret"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def compute_volatility(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Input: Log-returns dataframe, rolling window
    Output: Computes 21d standard deviation of log-returns
    """
    df = df.copy()
    roll_std = df["ret"].rolling(window=window, min_periods=2).std()

    df["volatility_21d"] = (roll_std).abs()
    df["volatility_21d"] = df["volatility_21d"].replace([np.inf, -np.inf], np.nan)
    return df

def compute_market_regime_features(
    df: pd.DataFrame,
    momentum_window: int = 21,
    ma_window: int = 21,
) -> pd.DataFrame:
    """
    Input: DataFrame with 'market_price'
    Output: Adds market momentum and deviation from moving average
    """
    df = df.copy()
    # Momentum
    df["market_momentum"] = (
        df["market_price"] / df["market_price"].shift(momentum_window) - 1
    )
    # Deviation from moving average
    ma = df["market_price"].rolling(window=ma_window, min_periods=5).mean()
    df["market_deviation"] = df["market_price"] / ma - 1

    df["market_momentum"] = df["market_momentum"].replace([np.inf, -np.inf], np.nan)
    df["market_deviation"] = df["market_deviation"].replace([np.inf, -np.inf], np.nan)
    return df

def compute_log_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: Market DataFrame with 'Volumes' column
    Output: computes Log-Volumes (natural logarithm)
    """
    df = df.copy()
    df["log_volume"] = np.where(df["volume"] > 0, np.log(df["volume"]), np.nan)
    return df


def compute_beta(df: pd.DataFrame, ewm_span: int = 63) -> pd.DataFrame:
    """
    Input: DataFrame with Benchmark's and TICKER's log-returns, moving average window
    Output: Exponential Moving Average beta DataFrame for a time-window
    """
    df = df.copy()

    x = df["ret"]
    y = df["return_market"]

    mean_xy = (x * y).ewm(span=ewm_span, adjust=False).mean()
    mean_x = x.ewm(span=ewm_span, adjust=False).mean()
    mean_y = y.ewm(span=ewm_span, adjust=False).mean()

    cov_ewm = mean_xy - mean_x * mean_y
    var_ewm = y.ewm(span=ewm_span, adjust=False).var()

    df["beta_ewm"] = cov_ewm / var_ewm
    df["beta_ewm"] = df["beta_ewm"].replace([np.inf, -np.inf], np.nan)
    df["beta_ewm"] = df["beta_ewm"].ffill()
    return df


def load_market_returns(start: str, end: str, benchmark: str = "^GSPC") -> pd.DataFrame:
    """
    Input: Benchmark( S&P500 as standard ), time-horizon
    Output: Computes log-returns and keeps benchmark price
    """
    market = get_price(benchmark, start=start, end=end)
    if market.empty:
        raise RuntimeError(f"Failed to load market benchmark {benchmark}")

    market = compute_returns(market)
    market = market.rename(columns={
        "price": "market_price",
        "ret": "return_market"
    })
    return market[["market_price", "return_market"]]

#SEC parse agent *

class SecClient:
    def __init__(self, user_agent: str, delay: float = 0.2):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.delay = delay
        self._lock = threading.Lock()
        self._last_call = 0.0
        self.cik_map = self._load_cik_map()

    def _get_json(self, url: str, timeout: int = 30) -> dict | None:
        """
        Input: a json url, timeout(seconds) limit
        Output: Sets a parse-rate limit and prevents multithreading
        """
        with self._lock:
            now = time.monotonic()
            wait = self.delay - (now - self._last_call)
            if wait > 0:
                time.sleep(wait)

            try:
                resp = self.session.get(url, timeout=timeout)
                self._last_call = time.monotonic()
                resp.raise_for_status()
                return resp.json()
            except Exception:
                self._last_call = time.monotonic()
                return None

    def _load_cik_map(self) -> dict[str, str]:
        """
        Output: A {TICKER:SEC_NUMBER} Dict for all available to SEC companies
        """
        raw = self._get_json("https://www.sec.gov/files/company_tickers.json")
        if not raw:
            raise RuntimeError("Failed to load SEC company_tickers.json")

        return {
            v["ticker"].upper(): str(v["cik_str"]).zfill(10)
            for v in raw.values()
        }

    def get_cik(self, ticker: str) -> str | None:
        """
        Input: TICKER
        Output: str SEC number for the TICKER
        """
        return self.cik_map.get(str(ticker).upper().strip())

    def get_company_facts(self, ticker: str) -> dict | None:
        """
        Input: TICKER
        Output: Dict of all SEC reports of a TICKER
        """
        cik = self.get_cik(ticker)
        if not cik:
            return None

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        data = self._get_json(url)
        if not data:
            return None

        return data.get("facts", {}).get("us-gaap")

#Fundamentals grouping agent *

def get_metric_dataframe(
    facts: dict,
    metric: str,
    preferred_units: tuple[str, ...],
    annual_only: bool = True,
) -> pd.DataFrame:
    """
    Input: get_company_facts, metric name, preferred_units array, True for of 10-K or 10-K/A annual reports
    Output: A metric DataFrame of full-available time-horizon
    """
    if metric not in facts:
        return pd.DataFrame()

    units = facts[metric].get("units", {})
    values = None

    for unit in preferred_units:
        if unit in units:
            values = units[unit]
            break

    if values is None and units:
        values = next(iter(units.values()))

    if not values:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(values)
    if df.empty:
        return pd.DataFrame()

    required = {"val", "end", "filed", "form"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    if annual_only:
        df = df[df["form"].isin(ANNUAL_FORMS)]

    if df.empty:
        return pd.DataFrame()

    df["end"] = pd.to_datetime(df["end"], errors="coerce").astype("datetime64[ns]")
    df["filed"] = pd.to_datetime(df["filed"], errors="coerce").astype("datetime64[ns]")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")

    keep_cols = [c for c in ["filed", "end", "val", "form", "accn", "fy", "fp"] if c in df.columns]
    df = df[keep_cols].dropna(subset=["filed", "end", "val"])
    df = df.sort_values(["end", "filed"]).drop_duplicates(subset=["end"], keep="last")

    return df.sort_values("end").reset_index(drop=True)


def get_metric_dataframe_multi(
    facts: dict,
    metrics: tuple[str, ...],
    preferred_units: tuple[str, ...],
    annual_only: bool = True,
) -> pd.DataFrame:
    """
    Input: get_company_facts, various metrics to yield the one, preferred from them
    Output: get_metric_dataframe used for metrics with other possible names in SEC reports
    """
    for metric in metrics:
        metric_df = get_metric_dataframe(
            facts=facts,
            metric=metric,
            preferred_units=preferred_units,
            annual_only=annual_only,
        )
        if metric_df is not None and not metric_df.empty:
            return metric_df

    return pd.DataFrame()


def extract_fundamentals(facts: dict) -> pd.DataFrame | None:
    """
    Input: get_company_facts function's yield
    Output: DataFrame of following fundamentals for a full-available time-horizon
    """
    metric_map = {
        "equity": (
            (
                "StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            ),
            ("USD",),
        ),
        "total_debt": (
            (
                "DebtAndFinanceLeaseObligations",
                "DebtAndCapitalLeaseObligations",
                "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities",
                "LongTermDebtAndFinanceLeaseObligationsIncludingCurrentMaturities",
                "LongTermDebtCurrentAndNoncurrent",
                "LongTermDebtAndShortTermBorrowings",
                "ShortTermAndLongTermDebt",
            ),
            ("USD",),
        ),
        "long_term_debt": (
            (
                "LongTermDebt",
                "LongTermDebtNoncurrent",
                "LongTermDebtAndCapitalLeaseObligations",
                "LongTermDebtAndFinanceLeaseObligations",
                "LongTermDebtAndCapitalLeaseObligationsNoncurrent",
                "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
                "NotesPayableNoncurrent",
                "FinanceLeaseLiabilityNoncurrent",
                "OperatingLeaseLiabilityNoncurrent",
            ),
            ("USD",),
        ),
        "short_term_debt": (
            (
                "ShortTermBorrowings",
                "LongTermDebtCurrent",
                "ShortTermBankLoansAndNotesPayable",
                "CurrentPortionOfLongTermDebt",
                "CurrentPortionOfLongTermDebtAndCapitalLeaseObligations",
                "CurrentPortionOfLongTermDebtAndFinanceLeaseObligations",
                "NotesPayableCurrent",
                "FinanceLeaseLiabilityCurrent",
                "OperatingLeaseLiabilityCurrent",
                "CommercialPaper",
            ),
            ("USD",),
        ),
        "cash": (
            (
                "CashAndCashEquivalentsAtCarryingValue",
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            ),
            ("USD",),
        ),
        "capex": (
            (
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "CapitalExpendituresIncurredButNotYetPaid",
            ),
            ("USD",),
        ),
        "ocf": (
            (
                "NetCashProvidedByUsedInOperatingActivities",
                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
            ),
            ("USD",),
        ),
    }

    merged = None

    for out_name, (tags, units) in metric_map.items():
        metric_df = get_metric_dataframe_multi(
            facts=facts,
            metrics=tags,
            preferred_units=units,
            annual_only=True,
        )
        if metric_df.empty:
            continue

        metric_df = metric_df[["filed", "end", "val"]].rename(columns={"filed": f"filed_{out_name}", "val": out_name})

        if merged is None:
            merged = metric_df
        else:
            merged = merged.merge(metric_df, on=["end"], how="outer")

    if merged is None or merged.empty:
        return None

    merged["end"] = pd.to_datetime(merged["end"], errors="coerce").astype("datetime64[ns]")

    filed_cols = [c for c in merged.columns if c.startswith("filed_")]
    if filed_cols:
        merged[filed_cols] = merged[filed_cols].apply(pd.to_datetime, errors="coerce")
        merged["filed"] = merged[filed_cols].max(axis=1)
        merged = merged.drop(columns=filed_cols)

    merged = merged.sort_values(["end", "filed"]).reset_index(drop=True)

    value_cols = [c for c in merged.columns if c not in ("filed", "end")]
    merged = merged.dropna(subset=["filed", "end"], how="any")

    if value_cols:
        merged = merged[merged[value_cols].notna().sum(axis=1) >= 1]

    if merged.empty:
        return None

    merged = merged.sort_values(["end", "filed"]).drop_duplicates(subset=["end"], keep="last")
    return merged.reset_index(drop=True)


def merge_fundamentals_asof(price_df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: get_price and extract_fundamentals functions' results
    Output: Merges Market and Sec parse results by the nearest past filing date
    """
    if fund_df is None or fund_df.empty:
        return price_df

    left = price_df.reset_index().rename(columns={"Date": "date"})
    right = fund_df.copy().rename(columns={"filed": "filing_date", "end": "period_end"})

    left["date"] = pd.to_datetime(left["date"], errors="coerce").astype("datetime64[ns]")
    right["filing_date"] = pd.to_datetime(right["filing_date"], errors="coerce").astype("datetime64[ns]")
    right["period_end"] = pd.to_datetime(right["period_end"], errors="coerce").astype("datetime64[ns]")

    left = left.dropna(subset=["date"]).sort_values("date")
    right = right.dropna(subset=["filing_date"]).sort_values("filing_date")

    merged = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="filing_date",
        direction="backward",
        tolerance=pd.Timedelta(days=730),
    )

    merged = merged.set_index("date")
    merged.index.name = "Date"
    return merged


def add_valuation_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: joined market and fundamental DataFrame
    Output: DataFrame with columns dedicated to valuation(Net_Debt, EV, FCFF, Market Cap)
    """
    df = df.copy()

    def as_series(value, index):
        if isinstance(value, pd.Series):
            return pd.to_numeric(value, errors="coerce").reindex(index)
        if value is None:
            return pd.Series(np.nan, index=index, dtype="float64")
        return pd.Series(value, index=index, dtype="float64")

    price = as_series(df.get("price"), df.index)
    shares = as_series(df.get("shares_yf"), df.index)
    cash = as_series(df.get("cash"), df.index)
    total_debt = as_series(df.get("total_debt"), df.index)
    long_term_debt = as_series(df.get("long_term_debt"), df.index)
    short_term_debt = as_series(df.get("short_term_debt"), df.index)

    df["market_cap"] = price * shares

    has_debt_parts = long_term_debt.notna() | short_term_debt.notna()
    debt_parts = long_term_debt.fillna(0.0) + short_term_debt.fillna(0.0)
    debt_parts = debt_parts.where(has_debt_parts, np.nan)

    df["total_debt"] = total_debt.where(total_debt.notna(), debt_parts)
    df["net_debt"] = df["total_debt"] - cash

    ocf = as_series(df.get("ocf"), df.index)
    capex = as_series(df.get("capex"), df.index)
    df["fcff"] = ocf - capex
    df["ev"] = df["market_cap"] + df["total_debt"] - cash
    return df

#pipeline

def process_ticker(
    row: pd.Series,
    market_df: pd.DataFrame,
    sec_client: SecClient,
    output_dir: Path,
    start_date: str,
    end_date: str,
    force_rebuild: bool = False,
) -> str:
    ticker = row["ticker"]
    industry = row.get("industry")

    ticker_dir = output_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = ticker_dir / "timeseries.parquet"
    meta_path = ticker_dir / "meta.json"

    if not force_rebuild and parquet_path.exists() and meta_path.exists():
        return f"[SKIP] {ticker}: already exists"

    try:
        df = get_price(ticker, start=start_date, end=end_date)
        if df.empty:
            ticker_dir.rmdir()
            return f"[WARN] {ticker}: no market data"

        df = df.join(market_df, how="left")
        df = compute_returns(df)
        df = compute_beta(df)
        df = compute_volatility(df)
        df = compute_log_volume(df)
        df = compute_market_regime_features(df)

        shares_df = get_yf_shares_series(ticker, start=start_date, end=end_date)
        if not shares_df.empty:
            df = merge_shares_asof(df, shares_df)

        has_fundamentals = False
        facts = sec_client.get_company_facts(ticker)

        if facts:
            fund_df = extract_fundamentals(facts)
            if fund_df is not None and not fund_df.empty:
                df = merge_fundamentals_asof(df, fund_df)
                has_fundamentals = True

        df = add_valuation_columns(df)

        df = df.loc[(df.index >= pd.Timestamp(START_DATE)) & (df.index < pd.Timestamp(END_DATE))].copy()

        df["ticker"] = ticker
        df["industry"] = industry

        df["ticker"] = df["ticker"].astype("string")
        df["industry"] = df["industry"].astype("string")

        meta = build_metadata(
            ticker=ticker,
            industry=industry,
            df=df,
            has_fundamentals=has_fundamentals,
        )

        atomic_write_parquet(df, parquet_path)
        atomic_write_json(meta_path, meta)

        return f"[OK] {ticker}: saved"

    except Exception as e:
        return f"[ERROR] {ticker}: {e}"


def main():
    companies = load_companies(INPUT_CSV)
    print(f"[INFO] Unique tickers: {len(companies)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    market_df = load_market_returns(start=FETCH_START_DATE, end=FETCH_END_DATE)
    sec_client = SecClient(user_agent=USER_AGENT, delay=SEC_DELAY)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(
                process_ticker,
                row=row,
                market_df=market_df,
                sec_client=sec_client,
                output_dir=OUTPUT_DIR,
                start_date=FETCH_START_DATE,
                end_date=FETCH_END_DATE,
                force_rebuild=FORCE_REBUILD,
            )
            for _, row in companies.iterrows()
        ]

        for future in as_completed(futures):
            print(future.result())

if __name__ == "__main__":
    main()