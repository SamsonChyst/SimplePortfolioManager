import json
from pathlib import Path
import pandas as pd
from modules_processor import has_fundamentals, time_slicing, get_market_3y_return
from valuation_dcf import target_row

#cfg
RATE_THRESHOLD = 0.030 #Threshold of T-Bonds to stratificate market regimes

TICKERS_DIR = Path("Datasets/tickers")
TRAIN_DIR   = Path("Datasets/train")
OUTPUT_PATH = Path("Datasets/train_dataset.csv")
_WACC_TABLE = pd.read_csv(
    "Datasets/US_WaccComponents_Timeseries.csv",
    sep=";", header=0, index_col="Year"
)

#Builder

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def is_hidden(name: str) -> bool:
    return name.startswith(".")


def ticker_is_valid(ticker: str) -> bool:
    folder = TICKERS_DIR / ticker

    if not folder.exists() or not folder.is_dir():
        return False
    if is_hidden(ticker):
        return False
    if not (folder / "timeseries.parquet").exists():
        return False
    return True


def list_tickers() -> list[str]:
    return [
        t for t in os.listdir(TICKERS_DIR)
        if ticker_is_valid(t)
    ]


def row_is_valid(row: pd.Series) -> bool:
    if row is None or len(row) == 0:
        return False

    required = ["ticker", "valuation_year", "real_alpha", "implied_upside"]
    for col in required:
        if col not in row:
            return False

    for col in ["valuation_year", "real_alpha", "implied_upside"]:
        val = row[col]
        if pd.isna(val) or not pd.api.types.is_number(val):
            return False

    return True


def get_t_bond_rate(year: int) -> float | None:
    if year not in _WACC_TABLE.index:
        return None
    val = _WACC_TABLE.loc[year, "T_Bond_Rate"]
    if pd.isna(val):
        return None
    return float(val)


def aggregate_market_features(df: pd.DataFrame) -> dict:
    """
    Input: daily time_slicing DataFrame for valuation_year-1 to valuation_year
    Output: aggregated market features as dict for one ticker-year row
    """
    if df is None or df.empty:
        return {}

    result = {}

    if "beta_ewm" in df.columns:
        result["beta_ewm_median"] = float(pd.to_numeric(df["beta_ewm"], errors="coerce").median())

    if "volatility_21d" in df.columns:
        v = pd.to_numeric(df["volatility_21d"], errors="coerce")
        result["volatility_21d_mean"] = float(v.mean())
        result["volatility_21d_max"]  = float(v.max())

    if "log_volume" in df.columns:
        result["log_volume_mean"] = float(pd.to_numeric(df["log_volume"], errors="coerce").mean())

    if "market_deviation" in df.columns:
        m = pd.to_numeric(df["market_deviation"], errors="coerce")
        result["market_deviation_mean"] = float(m.mean())
        result["market_deviation_std"]  = float(m.std())

    if "market_momentum" in df.columns:
        m = pd.to_numeric(df["market_momentum"], errors="coerce")
        result["market_momentum_mean"] = float(m.mean())
        result["market_momentum_std"]  = float(m.std())

    return result


def build_row(
    ticker: str,
    target: dict,
) -> pd.Series | None:
    """
    Input: ticker and one target json payload
    Output: single-row pd.Series with valuation signal, market features, and real_alpha
    """
    valuation_year = target.get("valuation_year")
    implied_upside = target.get("implied_upside")
    t_bond_rate    = target.get("t_bond_rate")
    real_alpha     = target.get("real_alpha")

    if any(v is None for v in [valuation_year, implied_upside, t_bond_rate, real_alpha]):
        return None
    if float(t_bond_rate) < RATE_THRESHOLD:
        return None

    try:
        daily_df = time_slicing(ticker, valuation_year - 1, valuation_year)
    except Exception:
        return None

    if daily_df is None or daily_df.empty:
        return None

    market_features = aggregate_market_features(daily_df)
    if not market_features:
        return None

    row = {
        "ticker":         ticker,
        "valuation_year": int(valuation_year),
        "implied_upside": float(implied_upside),
        "t_bond_rate":    float(t_bond_rate),
        "real_alpha":     float(real_alpha),
    }
    row.update(market_features)

    market_3y_return = get_market_3y_return(ticker, valuation_year)
    if market_3y_return is not None:
        row["market_3y_return"] = market_3y_return

    return pd.Series(row)


def load_target_json(ticker: str) -> list[dict]:
    ticker_dir = TRAIN_DIR / ticker
    if not ticker_dir.exists():
        return []

    results = []
    for path in sorted(ticker_dir.glob("target_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                results.append(json.load(f))
        except Exception:
            continue
    return results


def build_json():
    ensure_dir(TRAIN_DIR)
    tickers = list_tickers()
    total   = len(tickers)

    print(f"[build_json] {total} tickers found")

    skipped_no_fund = 0
    skipped_no_rows = 0
    total_saved     = 0

    for i, ticker in enumerate(tickers, 1):
        if not has_fundamentals(ticker):
            skipped_no_fund += 1
            continue

        saved = 0
        for year in range(2014, 2026):
            try:
                row = target_row(ticker, year, alpha_horizon=1)
            except Exception as e:
                print(f"  [{ticker}] {year}: target_row failed - {e}")
                continue

            if not row_is_valid(row):
                continue
            if abs(float(row["implied_upside"])) > 1.22:
                continue

            t_bond = get_t_bond_rate(year)
            if t_bond is None:
                continue

            payload = {
                "ticker":         ticker,
                "valuation_year": int(row["valuation_year"]),
                "alpha_horizon":  1,
                "real_alpha":     float(row["real_alpha"]),
                "implied_upside": float(row["implied_upside"]),
                "t_bond_rate":    t_bond,
            }

            ticker_dir = TRAIN_DIR / ticker
            ensure_dir(ticker_dir)
            path = ticker_dir / f"target_{year}.json"
            if path.exists():
                path.unlink()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            saved += 1

        if saved == 0:
            skipped_no_rows += 1
        else:
            total_saved += saved

        if i % 50 == 0 or i == total:
            print(f"[{i}/{total}] processed | saved so far: {total_saved} | "
                  f"no fundamentals: {skipped_no_fund} | no valid rows: {skipped_no_rows}")

    print(f"\n[build_json] done - {total_saved} json files saved across "
          f"{total - skipped_no_fund - skipped_no_rows} tickers")


def build_dataset() -> None:
    tickers = list_tickers()
    total   = len(tickers)

    print(f"[build_dataset] {total} tickers found")
    print(f"[build_dataset] filtering to high-rate regime: T_Bond >= {RATE_THRESHOLD:.3f}")

    rows       = []
    no_targets = 0
    total_rows = 0

    for i, ticker in enumerate(tickers, 1):
        targets = load_target_json(ticker)
        if not targets:
            no_targets += 1
            continue

        for target in targets:
            try:
                row = build_row(ticker, target)
            except Exception as e:
                print(f"  [{ticker}] {target.get('valuation_year')}: FAILED - {e}")
                continue

            if row is None:
                continue

            rows.append(row)
            total_rows += 1

        if i % 50 == 0 or i == total:
            print(f"[{i}/{total}] processed | rows so far: {total_rows} | "
                  f"no targets: {no_targets}")

    if not rows:
        print("[build_dataset] no rows collected, aborting")
        return

    df = pd.DataFrame(rows).reset_index(drop=True)

    col_order = [
        "ticker", "valuation_year",
        "implied_upside", "t_bond_rate",
        "beta_ewm_median",
        "volatility_21d_mean", "volatility_21d_max",
        "log_volume_mean",
        "market_deviation_mean", "market_deviation_std",
        "market_momentum_mean", "market_momentum_std",
        "market_3y_return",
        "real_alpha",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    print(f"\nreal_alpha distribution:")
    print(df["real_alpha"].describe().round(4))

    print(f"\nimplied_upside distribution:")
    print(df["implied_upside"].describe().round(4))

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[build_dataset] done - {len(df)} rows saved to {OUTPUT_PATH}")
    print(df.describe().round(4))


if __name__ == "__main__":
    try:
        import shutil, os
        shutil.rmtree("Datasets/train/")
        os.remove("Datasets/train_dataset.csv")
    except FileNotFoundError:
        pass

    build_json()
    build_dataset()


#Notes
'''
Dataset Construction Methodology - Base Alpha Dataset

This module builds the base ticker-year dataset used before the machine learning and copula stages.
Each observation represents one ticker at one valuation year and contains three groups of information:

  1. Realized outcome:
     real_alpha = one-year stock return minus one-year S&P 500 return.

  2. Fundamental valuation signal:
     implied_upside is produced by the DCF valuation module and is kept as an independent
     fundamental signal. It is not transformed into old copula-based labels at this stage.

  3. Market regime features:
     aggregated market-based variables are computed from the one-year market window prior
     to the valuation year using filing-date-aware time slicing.

The dataset is restricted to the high-rate regime defined by:
  T_Bond_Rate >= RATE_THRESHOLD

This restriction is based on the empirical observation that the DCF signal has stronger
relationship with realized alpha during high-rate periods. However, this module does not
fit a copula and does not create probability labels. Its purpose is only to construct the
clean base dataset used later by:

  - market-only ML model: market features -> alpha_hat
  - vine copula model: real_alpha, alpha_hat, implied_upside
'''
