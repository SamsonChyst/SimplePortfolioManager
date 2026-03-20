import requests
import pandas as pd
import numpy as np
import os
import yfinance as yf
from dotenv import load_dotenv
load_dotenv()

DataFrame = pd.read_csv("Datasets/stock_prices_daily.csv")[["Ticker", "Company_Name", "Sector", "Industry"]]


#SEC reports parse agent
HEADERS = {"User-Agent": os.getenv("Mail")}

def get_cik(ticker: str) -> str | None:
    '''
    Input: TICKER for US Company
    Return: str Central Index Key of a company for operating in EDGAR SEC db
    '''
    url = "https://www.sec.gov/files/company_tickers.json"
    data = requests.get(url, headers=HEADERS).json()
    for item in data.values():
        if item["ticker"] == ticker.upper():
            return str(item["cik_str"]).zfill(10)
    return None

def get_company_facts(cik: str) -> dict:
    '''
    Input: Central Index Key
    Return: eXtensible Business Reporting Language filing
    '''
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    data = requests.get(url, headers=HEADERS).json()
    return data["facts"]["us-gaap"]

def get_metric(facts: dict, metric: str, annual_only=True) -> pd.Series:
    '''
    Input XBRL filing for a Company and a Financial metric to be yielded
    Return: Metric Values per periods
    '''
    try:
        units = facts[metric]["units"]
        if "USD" in units:
            values = units["USD"]
        else:
            values = list(units.values())[0]
        df = pd.DataFrame(values)
        df["end"] = pd.to_datetime(df["end"])
        if annual_only:
            df = df[df["form"] == "10-K"]
        df = df.sort_values("end").drop_duplicates("end", keep="last")
        return df.set_index("end")["val"].sort_index()
    except:
        return pd.Series(dtype=float)

def extract_fundamentals(facts: dict) -> pd.DataFrame | None:
    '''
    Input: XBRL filing of {N} companny
    Output: Metrics for all available periods
    '''
    metrics = {
        "revenue": "Revenues",
        "ebit": "OperatingIncomeLoss",
        "net_income": "NetIncomeLoss",
        "assets": "Assets",
        "equity": "StockholdersEquity",
        "debt": "LongTermDebt",
        "cash": "CashAndCashEquivalentsAtCarryingValue",
        "capex": "PaymentsToAcquirePropertyPlantAndEquipment",
        "ocf": "NetCashProvidedByUsedInOperatingActivities",
        "shares": "CommonStockSharesOutstanding",
        "interest": "InterestExpense"
    }
    data = {}
    for name, tag in metrics.items():
        series = get_metric(facts, tag)
        if not series.empty:
            data[name] = series
    if not data:
        return None
    df = pd.DataFrame(data)
    threshold = int(len(df.columns) * 0.7)
    df = df.dropna(thresh=threshold)
    df = df.sort_index()
    return df

#Market parse agent

def get_price(ticker: str, start="2010-01-01", end="2026-01-01") -> pd.DataFrame:
    '''
    Input: Company's Ticker and time horizon
    Output: Daily Closing price and traded Volume values
    '''
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df = data[["Close", "Volume"]].copy()
    df.columns = ["price", "volume"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Input: Closing price Dataframe
    Output: log-returns
    '''
    df = df.copy()
    df["price"] = df["price"].interpolate(method="time")
    df["ret"] = np.log(df["price"] / df["price"].shift(1))
    return df

def prepare_data(stock_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Input: pd.Dataframe of stock's and market's log-returns
    Output: Joins Stock's and Market's returns
    '''
    s = compute_returns(stock_df)
    m = compute_returns(market_df)
    df = pd.merge(
        s[["ret"]],
        m[["ret"]],
        left_index=True,
        right_index=True,
        how="inner",
        suffixes=("_stock", "_market")
    )
    df.columns = ["return_stock", "return_market"]
    return df

def compute_beta(df: pd.DataFrame, window=252) -> pd.DataFrame:
    '''
    Input: prepare_data dataframe and a window for Beta
    Output: Beta as a pd.Dataframe
    '''
    df = df.copy()
    x = df["return_stock"]
    y = df["return_market"]
    mean_x = x.rolling(window).mean()
    mean_y = y.rolling(window).mean()
    mean_xy = (x * y).rolling(window).mean()
    cov = mean_xy - mean_x * mean_y
    var = y.rolling(window).var()
    df["beta"] = cov / var
    return df

def build_beta(ticker: str, market_ticker="^GSPC", start="2010-01-01",
               end="2026-01-01", window=252) -> pd.DataFrame:
    '''
    Input: Ticker and a time-horizon
    Output: a Beta and log-return Dataframe for Stock and market
    '''
    extended_start = pd.to_datetime(start) - pd.DateOffset(days=window * 2)
    stock_df = get_price(ticker, extended_start, end)
    market_df = get_price(market_ticker, extended_start, end)
    df = prepare_data(stock_df, market_df)
    df = compute_beta(df, window)
    df = df.loc[start:end]
    df["volume"] = get_price(ticker)["volume"]
    return df

#DataFrame construction
if __name__ == "__main__":
    all_results = []

    # Итерируемся по тикерам из вашего первого DataFrame
    for ticker in DataFrame['Ticker'].unique():
        try:
            print(f"Обработка {ticker}...")

            # 1. Получаем CIK и фундаментал
            cik = get_cik(ticker)
            if cik:
                facts = get_company_facts(cik)
                fundamentals = extract_fundamentals(facts)
            else:
                fundamentals = None

            # 2. Получаем рыночные данные (Beta и доходности)
            beta_data = build_beta(ticker)

            # 3. Объединяем их, если оба набора данных получены
            if fundamentals is not None and not beta_data.empty:
                # Склеиваем по дате (индексу)
                # Так как фундаментал обычно годовой (10-K), используем ffill (forward fill),
                # чтобы протянуть значения фундаментала на ежедневные данные беты
                combined = beta_data.join(fundamentals).ffill()

                # Добавляем колонку с тикером, чтобы различать компании в общем датасете
                combined['Ticker'] = ticker

                all_results.append(combined)

        except Exception as e:
            print(f"Ошибка при обработке {ticker}: {e}")

    # Финальное объединение всех тикеров в одну таблицу
    if all_results:
        final_dataset = pd.concat(all_results)
        # Сбрасываем индекс, чтобы дата стала обычной колонкой
        final_dataset = final_dataset.reset_index().rename(columns={'index': 'Date'})
    else:
        print("Данные не были собраны.")