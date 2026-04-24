import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from dataset_processor import has_fundamentals, time_slicing
from valuation_marker import target_row, label_maker


#Notes
'''
Dataset Construction Methodology

The empirical analysis identifies a regime-dependent relationship between DCF-implied upside
and realized three-year excess return (real_alpha = stock return - S&P 500 return over a
three-year horizon). The conditioning variable is the US 10-Year Treasury Bond rate (T-Bond),
sourced from Damodaran's annual risk-free rate estimates.

Two regimes are defined by a threshold of T_Bond = 3.0%, motivated by the observed structural
break in Kendall's tau between implied_upside and real_alpha:
  * Low-rate regime  (T_Bond < 3.0%,  n=1708): tau = 0.057
  * High-rate regime (T_Bond >= 3.0%, n=569):  tau = 0.172

The high-rate regime corresponds predominantly to 2022, a period of rapid monetary tightening
where fundamental valuation re-emerged as a dominant pricing factor. The low-rate regime spans
2014-2021, characterized by momentum-driven markets where DCF signals had limited predictive
power.

A bivariate Clayton copula is fitted separately for each regime via Canonical Maximum Likelihood
Estimation (CMLE) on empirical pseudo-observations. The Clayton family was selected based on
superior log-likelihood and information criteria relative to Gaussian, Student-t, Gumbel, and
Frank alternatives (Clayton: loglik=18.88, AIC=-35.77, BIC=-30.92). Clayton captures lower tail
dependence — the empirically observed asymmetry whereby the DCF signal is most reliable when
identifying severely overvalued securities.

Fitted parameters:
  * Low-rate Clayton:  theta=0.2586, tau_implied=0.1145
  * High-rate Clayton: theta=0.3549, tau_implied=0.1507

The copula yields two soft labels per observation:
  P(alpha > 0 | DCF, regime) — conditional probability of positive excess return
  E(alpha | DCF, regime)     — conditional expected excess return

Feature selection is based on Kendall's tau with P(alpha > 0) across 1,145,933 daily
observations aggregated to ticker-year level:
  implied_upside:   tau = +0.834  (primary valuation signal)
  t_bond_rate:      tau = -0.293  (regime conditioning variable)
  volatility_21d:   tau = -0.177  (market uncertainty proxy)
  beta_ewm:         tau = -0.027  (systematic risk, weak signal)

Final feature set per ticker-year observation:
  Valuation:  implied_upside, P(alpha > 0), E(alpha)
  Macro:      t_bond_rate
  Risk:       beta_ewm (median), volatility_21d (mean, max)
  Volume:     log_volume (mean)
  Regime:     market_deviation (mean, std), market_momentum (mean, std)
  Label:      real_alpha (three-year realized excess return vs S&P 500)
'''

#cfg
TICKERS_DIR = Path("Datasets/tickers")
TRAIN_DIR   = Path("Datasets/train")
OUTPUT_PATH = Path("Datasets/train_dataset.parquet")
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
        if ticker_is_valid(t)]


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
    Output: single-row pd.Series with all features and label, or None if invalid
    """
    valuation_year = target.get("valuation_year")
    implied_upside = target.get("implied_upside")
    t_bond_rate    = target.get("t_bond_rate")
    real_alpha     = target.get("real_alpha")

    if any(v is None for v in [valuation_year, implied_upside, t_bond_rate, real_alpha]):
        return None

    try:
        labels = label_maker(float(implied_upside), float(t_bond_rate))
    except Exception:
        return None

    if labels is None or labels.empty:
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
        "p_alpha_gt_0":   float(labels["P(alpha > 0)"]),
        "e_alpha":        float(labels["E(alpha)"]),
        "real_alpha":     float(real_alpha),
    }
    row.update(market_features)

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
    total = len(tickers)

    print(f"[build_json] {total} tickers found")

    skipped_no_fund = 0
    skipped_no_rows = 0
    total_saved = 0

    for i, ticker in enumerate(tickers, 1):
        if not has_fundamentals(ticker):
            skipped_no_fund += 1
            continue

        saved = 0
        for year in range(2014, 2023):
            try:
                row = target_row(ticker, year, alpha_horizon=3)
            except Exception as e:
                print(f"  [{ticker}] {year}: target_row failed — {e}")
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
                "alpha_horizon":  3,
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

    print(f"\n[build_json] done — {total_saved} json files saved across "
          f"{total - skipped_no_fund - skipped_no_rows} tickers")


def build_dataset() -> None:
    tickers = list_tickers()
    total   = len(tickers)

    print(f"[build_dataset] {total} tickers found")

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
                print(f"  [{ticker}] {target.get('valuation_year')}: FAILED — {e}")
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
        "p_alpha_gt_0", "e_alpha",
        "beta_ewm_median",
        "volatility_21d_mean", "volatility_21d_max",
        "log_volume_mean",
        "market_deviation_mean", "market_deviation_std",
        "market_momentum_mean", "market_momentum_std",
        "real_alpha",
    ]
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n[build_dataset] done — {len(df)} rows saved to {OUTPUT_PATH}")
    print(df.describe().round(4))


if __name__ == "__main__":
    try:
        import shutil
        shutil.rmtree("Datasets/train/")
    except FileNotFoundError:
        pass

    build_json()
    build_dataset()