from pathlib import Path
import subprocess
import sys
import time
import zipfile
from datetime import datetime
import pandas as pd

#cfg

REPORTS_DIR = Path("Reports")
TEMPLATE_PATH = Path("Reports/report_template.pptm")


def fmt_pct(x, digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{float(x) * 100:.{digits}f}%"


def fmt_num(x, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{float(x):.{digits}f}"


def safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_str(x, default: str = "") -> str:
    try:
        if x is None or pd.isna(x):
            return default
        return str(x)
    except Exception:
        return default


#Qualitive word assignment

def qualitative_probability(prob_positive: float) -> str:
    if prob_positive >= 0.65:
        return "high"
    if prob_positive >= 0.55:
        return "moderate"
    if prob_positive >= 0.45:
        return "uncertain"
    return "low"


def skew_description(expected_alpha: float, median_alpha: float) -> str:
    if expected_alpha > median_alpha + 0.03:
        return "right-skewed"
    if expected_alpha < median_alpha - 0.03:
        return "left-skewed"
    return "fairly balanced"


def risk_level(prob_minus_10: float, prob_minus_20: float) -> str:
    if prob_minus_20 >= 0.20:
        return "high"
    if prob_minus_10 >= 0.25:
        return "material"
    return "limited"


def normalize_report_date(x) -> str:
    if x is None:
        return datetime.today().strftime("%Y-%m-%d")

    try:
        if pd.isna(x):
            return datetime.today().strftime("%Y-%m-%d")
    except Exception:
        pass

    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d")

    if isinstance(x, datetime):
        return x.strftime("%Y-%m-%d")

    text = str(x).strip()

    if not text:
        return datetime.today().strftime("%Y-%m-%d")

    try:
        return pd.to_datetime(text).strftime("%Y-%m-%d")
    except Exception:
        return text


def write_params(data: pd.Series, pdf_path: Path, params_path: Path) -> None:
    prob_positive = safe_float(data.get("prob_positive"))
    prob_minus_10 = safe_float(data.get("prob_alpha_lt_minus_10"))
    prob_minus_20 = safe_float(data.get("prob_alpha_lt_minus_20"))
    expected_alpha = safe_float(data.get("expected_alpha"))
    median_alpha = safe_float(data.get("median_alpha"))

    valuation_date = normalize_report_date(
        data.get("valuation_date", data.get("report_date", datetime.today()))
    )

    alpha_horizon = safe_str(
        data.get("alpha_horizon", data.get("horizon", "1Y")),
        "1Y",
    )

    lines = [
        f"ticker={safe_str(data.get('ticker', '')).upper()}",
        f"valuation_date={valuation_date}",
        f"alpha_horizon={alpha_horizon}",
        f"pdf_path={pdf_path.resolve()}",
        f"prob_positive={prob_positive}",
        f"prob_negative={safe_float(data.get('prob_negative'))}",
        f"prob_alpha_lt_minus_10={prob_minus_10}",
        f"prob_alpha_lt_minus_20={prob_minus_20}",
        f"median_alpha={median_alpha}",
        f"expected_alpha={expected_alpha}",
        f"alpha_efficiency={safe_float(data.get('alpha_efficiency'))}",
        f"implied_upside={safe_float(data.get('implied_upside'))}",
        f"t_bond_rate={safe_float(data.get('t_bond_rate'))}",
        f"alpha_hat={safe_float(data.get('alpha_hat'))}",
        f"qualitative_word={qualitative_probability(prob_positive)}",
        f"skew_description={skew_description(expected_alpha, median_alpha)}",
        f"risk_level={risk_level(prob_minus_10, prob_minus_20)}",
        f"q05={safe_float(data.get('alpha_q05'))}",
        f"q25={safe_float(data.get('alpha_q25'))}",
        f"q50={safe_float(data.get('alpha_q50'))}",
        f"q75={safe_float(data.get('alpha_q75'))}",
        f"q95={safe_float(data.get('alpha_q95'))}",
    ]

    params_path.write_text("\n".join(lines), encoding="utf-8")


def _copy_pptm_template(template_pptm: Path, pptm_out: Path) -> None:
    with zipfile.ZipFile(template_pptm, "r") as zin:
        with zipfile.ZipFile(pptm_out, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.namelist():
                zout.writestr(item, zin.read(item))


#VBA Macros runner

def run_macro_macos(pptm_path: Path) -> None:
    pptm_abs = str(pptm_path.resolve()).replace('"', '\\"')
    macro_name = f"{pptm_path.name}!GenerateReport"

    script = f'''
tell application "Microsoft PowerPoint"
    activate
    set presObj to open POSIX file "{pptm_abs}"
    delay 4
    run VB macro macro name "{macro_name}"
    delay 10
end tell
'''

    result = subprocess.run(
        ["osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"AppleScript macro failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )


def run_macro_windows(pptm_path: Path) -> None:
    pptm_abs = str(pptm_path.resolve()).replace("'", "''")

    script = f"""
$app = New-Object -ComObject PowerPoint.Application
$app.Visible = [Microsoft.Office.Core.MsoTriState]::msoTrue
$pres = $app.Presentations.Open('{pptm_abs}')
Start-Sleep -Seconds 3
$app.Run('{pptm_path.name}!GenerateReport')
Start-Sleep -Seconds 10
$pres.Close()
$app.Quit()
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres) | Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
[System.GC]::Collect()
[System.GC]::WaitForPendingFinalizers()
"""

    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"PowerShell macro failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )


def run_macro(pptm_path: Path) -> None:
    if sys.platform == "darwin":
        run_macro_macos(pptm_path)
        return

    if sys.platform == "win32":
        run_macro_windows(pptm_path)
        return

    raise RuntimeError(f"Unsupported platform: {sys.platform}")


def wait_for_pdf(pdf_path: Path, attempts: int = 60, delay: float = 0.5) -> None:
    for _ in range(attempts):
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            return
        time.sleep(delay)

    raise RuntimeError(f"PDF was not created: {pdf_path}")


def cleanup_temp_files(*paths: Path) -> None:
    for path in paths:
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def create_report_from_data(
        data: pd.Series,
        template_path: Path = TEMPLATE_PATH,
        reports_dir: Path = REPORTS_DIR,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)

    ticker = safe_str(data.get("ticker", "report"), "report").upper().strip()
    pptm_path = reports_dir / f"{ticker}.pptm"
    pdf_path = reports_dir / f"{ticker}.pdf"
    params_path = reports_dir / "params.txt"

    try:
        write_params(data, pdf_path, params_path)
        _copy_pptm_template(template_path, pptm_path)
        run_macro(pptm_path)
        wait_for_pdf(pdf_path)

    finally:
        cleanup_temp_files(pptm_path, params_path)

    return pdf_path


#Report Finalisation

def create_report(
        ticker: str,
        implied_upside: float,
        t_bond_rate: float,
        valuation_date=None,
        alpha_horizon: str = "1Y",
        template_path: Path = TEMPLATE_PATH,
        reports_dir: Path = REPORTS_DIR,
) -> Path:
    from modules_processor import get_copula_data

    data = get_copula_data(
        ticker=ticker,
        implied_upside=implied_upside,
        t_bond_rate=t_bond_rate,
    )

    if data.empty:
        raise ValueError(f"No report data was generated for {ticker}")

    data = data.copy()
    data["valuation_date"] = normalize_report_date(valuation_date)
    data["alpha_horizon"] = alpha_horizon

    return create_report_from_data(
        data=data,
        template_path=template_path,
        reports_dir=reports_dir,
    )


def create_reports_batch(
        requests: list[tuple],
        template_path: Path = TEMPLATE_PATH,
        reports_dir: Path = REPORTS_DIR,
) -> list[Path]:
    paths = []

    for request in requests:
        ticker = request[0]
        implied_upside = request[1]
        t_bond_rate = request[2]

        valuation_date = None
        alpha_horizon = "1Y"

        if len(request) >= 4:
            valuation_date = request[3]

        if len(request) >= 5:
            alpha_horizon = request[4]

        path = create_report(
            ticker=ticker,
            implied_upside=implied_upside,
            t_bond_rate=t_bond_rate,
            valuation_date=valuation_date,
            alpha_horizon=alpha_horizon,
            template_path=template_path,
            reports_dir=reports_dir,
        )
        paths.append(path)

    return paths

#Notes
'''
Generates PDF reports from a .pptm template with an embedded VBA macro.
Python writes params.txt, copies the template, and triggers GenerateReport via
osascript (macOS) or PowerShell COM (Windows). VBA handles all slide edits and
exports the PDF. Temp files are cleaned up regardless of outcome.
Requires Microsoft PowerPoint - no Office, no PDF.
'''