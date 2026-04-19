import os
import json
from pathlib import Path
import pandas as pd
from dataset_processor import has_fundamentals, time_slicing
from valuation_marker import target_row, label_maker


# cfg
TICKERS_DIR = Path("Datasets/tickers")
TRAIN_DIR = Path("Datasets/train") # 898 tickers


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


def save_target_json(ticker: str, payload: dict):
    ticker_dir = TRAIN_DIR / ticker
    ensure_dir(ticker_dir)

    path = ticker_dir / "target.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_for_ticker(
    ticker: str,
    start_year: int = 2014,
    end_year: int = 2025,
    prefer_latest: bool = True,
):
    # skip if no fundamentals
    if not has_fundamentals(ticker):
        return

    years = list(range(start_year, end_year + 1))
    if prefer_latest:
        years = years[::-1]

    for year in years:
        try:
            row = target_row(ticker, year)
        except Exception:
            break

        if not row_is_valid(row) or abs(float(row["implied_upside"])) > 1.22:
            continue

        payload = {
            "ticker": ticker,
            "valuation_year": int(row["valuation_year"]),
            "real_alpha": float(row["real_alpha"]),
            "implied_upside": float(row["implied_upside"]),
        }
        save_target_json(ticker, payload)
        return


def build_json():
    ensure_dir(TRAIN_DIR)
    tickers = list_tickers()

    for ticker in tickers:
        try:
            build_for_ticker(ticker)
        except Exception:
            continue


def load_target_json(ticker: str) -> dict | None:
    path = TRAIN_DIR / ticker / "target.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def build_ml_for_ticker(ticker: str):
    target = load_target_json(ticker)
    if target is None:
        return

    valuation_year = target.get("valuation_year")
    implied_upside = target.get("implied_upside")

    if valuation_year is None or implied_upside is None:
        return

    try:
        labels = label_maker(float(implied_upside))
    except Exception:
        return

    if labels is None or labels.empty:
        return

    try:
        df = time_slicing(
            ticker,
            valuation_year - 1,
            valuation_year
        )
    except Exception:
        return

    if df is None or df.empty:
        return

    df = df[["market_deviation", "market_momentum", "return_market", "ret", "beta_ewm",
    "volatility_21d", "log_volume"]].copy()

    df["implied_upside"] = float(implied_upside)

    df["P(alpha > 0)"] = float(labels["P(alpha > 0)"])
    df["E(alpha)"] = float(labels["E(alpha)"])

    save_path = TRAIN_DIR / ticker / "dataset.parquet"

    try:
        df.to_parquet(save_path, index=True)
    except Exception:
        pass


def build_dataset():
    tickers = list_tickers()

    for ticker in tickers:
        try:
            build_ml_for_ticker(ticker)
        except Exception:
            continue


if __name__ == "__main__":
    build_json()
    build_dataset()