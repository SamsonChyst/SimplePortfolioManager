if __name__ == "__main__":
    from report_export import create_report

    Ticker: str = input("Enter a ticker of a US company (e.g NVDA)")
    ImpliedUpside: float = input("Enter your DCF-based implied upside (e.g 0.3 for 30%)")
    TBondRate: float = input("Enter current 10Y US T-Bond Yield Rate (e.g 0.042 for 2026)")

    create_report(Ticker, ImpliedUpside, TBondRate)