from pathlib import Path
from main_helper import *
from report_export import create_report


if __name__ == "__main__":
    try:
        check_runtime_files()

        ticker, implied_upside, t_bond_rate = gui_inputs()

        pdf_path = create_report(
            ticker=ticker,
            implied_upside=implied_upside,
            t_bond_rate=t_bond_rate,
        )

        show_success(Path(pdf_path))

    except InputCancelled:
        pass

    except Exception as e:
        log_path = save_error_log(e)
        show_error(RuntimeError(f"{e}\n\nDetails were saved to: {log_path}"))