from pathlib import Path
import re
import sys
import traceback
from datetime import datetime
from report_export import create_report

#cfg
DEFAULT_REPORTS_DIR = Path("Reports")
DEFAULT_TEMPLATE_PATH = Path("Reports/report_template.pptm")
DEFAULT_MODEL_PATH = Path("Models/market_model.cbm")


class InputCancelled(Exception):
    pass


def normalize_ticker(value: str) -> str:
    if value is None:
        raise ValueError("Ticker is empty")

    ticker = str(value).strip().upper()

    if not ticker:
        raise ValueError("Ticker is empty")

    if not re.fullmatch(r"[A-Z0-9.\-]{1,12}", ticker):
        raise ValueError("Ticker must contain only letters, digits, dots or hyphens")

    return ticker


def parse_float(value, field_name: str) -> float:
    if value is None:
        raise ValueError(f"{field_name} is empty")

    if isinstance(value, (int, float)):
        result = float(value)
    else:
        text = str(value).strip().replace(",", ".")

        if text.endswith("%"):
            text = text[:-1].strip()
            result = float(text) / 100.0
        else:
            result = float(text)

    if not result == result or result in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} is not finite")

    return result


def validate_inputs(ticker: str, implied_upside: float, t_bond_rate: float) -> tuple[str, float, float]:
    ticker = normalize_ticker(ticker)
    implied_upside = parse_float(implied_upside, "Implied upside")
    t_bond_rate = parse_float(t_bond_rate, "T-Bond rate")

    if implied_upside <= -1.0:
        raise ValueError("Implied upside must be greater than -1.0")

    if abs(implied_upside) > 10:
        raise ValueError("Implied upside looks too extreme. Use 0.3 or 30%, not 30 for 30%")

    if t_bond_rate < 0:
        raise ValueError("T-Bond rate cannot be negative")

    if t_bond_rate > 1:
        raise ValueError("T-Bond rate looks too high. Use 0.042 or 4.2%, not 4.2")

    return ticker, implied_upside, t_bond_rate


def check_runtime_files(template_path: Path = DEFAULT_TEMPLATE_PATH, model_path: Path = DEFAULT_MODEL_PATH) -> None:
    if not template_path.exists():
        raise FileNotFoundError(f"PowerPoint template was not found: {template_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"ML model was not found: {model_path}")

    DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def terminal_inputs() -> tuple[str, float, float]:
    ticker = input("Enter ticker, e.g. NVDA: ")
    implied_upside = input("Enter DCF-based implied upside, e.g. 0.3 or 30%: ")
    t_bond_rate = input("Enter current 10Y US T-Bond rate, e.g. 0.042 or 4.2%: ")

    return validate_inputs(ticker, implied_upside, t_bond_rate)


def gui_inputs() -> tuple[str, float, float]:
    try:
        import tkinter as tk
        from tkinter import simpledialog, messagebox
    except Exception:
        return terminal_inputs()

    root = tk.Tk()
    root.withdraw()
    root.update()

    try:
        while True:
            try:
                ticker = simpledialog.askstring(
                    "Portfolio Report",
                    "Enter ticker of a US company, e.g. NVDA:",
                    parent=root,
                )

                if ticker is None:
                    raise InputCancelled("Input cancelled")

                implied_upside = simpledialog.askstring(
                    "Portfolio Report",
                    "Enter DCF-based implied upside, e.g. 0.3 or 30%:",
                    parent=root,
                )

                if implied_upside is None:
                    raise InputCancelled("Input cancelled")

                t_bond_rate = simpledialog.askstring(
                    "Portfolio Report",
                    "Enter current 10Y US T-Bond rate, e.g. 0.042 or 4.2%:",
                    parent=root,
                )

                if t_bond_rate is None:
                    raise InputCancelled("Input cancelled")

                return validate_inputs(ticker, implied_upside, t_bond_rate)

            except InputCancelled:
                raise

            except Exception as e:
                messagebox.showerror("Invalid input", str(e), parent=root)

    finally:
        root.destroy()


def show_success(path: Path) -> None:
    message = f"Report created:\n{path.resolve()}"

    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Done", message, parent=root)
        root.destroy()
    except Exception:
        print(message)


def show_error(error: Exception) -> None:
    message = str(error)

    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Report failed", message, parent=root)
        root.destroy()
    except Exception:
        print(f"Report failed: {message}", file=sys.stderr)


def save_error_log(error: Exception) -> Path:
    DEFAULT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    log_path = DEFAULT_REPORTS_DIR / "last_error.log"
    text = [
        f"time={datetime.now().isoformat(timespec='seconds')}",
        f"error={type(error).__name__}: {error}",
        "",
        traceback.format_exc(),
    ]

    log_path.write_text("\n".join(text), encoding="utf-8")
    return log_path

