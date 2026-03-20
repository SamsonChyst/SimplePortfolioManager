import requests
import pandas as pd
import os
import yfinance as yf
from dotenv import load_dotenv
load_dotenv()


db = pd.read_csv("Datasets/stock_prices_daily.csv")[["Ticker", "Company_Name", "Sector", "Industry"]]


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

#Market Multiples parse agent
def get_price(ticker: str, start="2010-01-01", end="2026-01-01") -> pd.DataFrame:
    '''
    Input: Company's Ticker and time horizon
    Output: Daily Closing price and traded Volume values
    '''
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df = data[["Close", "Volume"]].copy()
    df.columns = ["price", "volume"]
    return df


#Data Validation & Filtering
print(get_price("NVDA"))
