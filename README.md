SimplePortfolioManager is a quant-style research pet project focused on the US equity market. It is designed as an assistive layer for discretionary fundamental analysis rather than as a production trading system. The main idea is simple: instead of asking the user to trust one DCF fair value or one machine-learning alpha forecast, the project combines market-based and valuation-based signals into a conditional probability distribution of future excess return.
The project explores signal fusion under weak signal-to-noise conditions. In equity research, both fundamental valuation and market-derived signals are noisy. A single point estimate often hides that uncertainty. SimplePortfolioManager tries to make the output more decision-useful by estimating probabilities, conditional quantiles, median alpha, downside and upside tails, and a Sharpe-like efficiency metric.
The final user-facing output is a one-page PDF report for a selected ticker.
Core idea
The project separates two different views of a company:
1.	Market-side view: A machine-learning model estimates alpha_hat using market and regime features such as beta, volatility, volume, market momentum, deviation from moving averages, market return, and the 10Y Treasury rate.
2.	Fundamental-side view: A DCF-based implied upside is calculated from parsed company fundamentals, FCFF, capital structure, shares outstanding, WACC assumptions, and market prices.
3.	Probabilistic fusion layer: The two signals are combined through a Clayton-based vine copula probabilistic fusion layer. The goal is not only to improve a point forecast, but to estimate the conditional distribution of one-year market-relative alpha.
This makes the project different from a simple DCF model or a pure ML regression model. The final output is not just:
expected alpha = x
but a richer decision object:
P(alpha > 0)
median alpha
expected alpha
expected upside
expected downside
tail probabilities
Sharpe-like ratio
conditional football-field range
That is much easier to interpret in a research workflow than one deterministic value.
Regime-specific hypothesis
SimplePortfolioManager is intentionally built around the US equity market and a higher-rate macro regime.
The main empirical assumption is that DCF-derived implied upside is weak in low-rate environments and should not be used as a standalone forecasting factor. However, when the 10Y Treasury yield is above 3%, valuation information becomes more useful as an orthogonal correction to the market-side model.
For that reason, the copula training regime is filtered to:
10Y Treasury rate >= 3%
This does not mean that DCF “predicts alpha” by itself. The project treats DCF as an additional fundamental signal that is fused with the market’s view of the company.
Current dataset scale
The current research pipeline was built on:
2,792 tickers parsed
4,226 total valuation rows saved
2,469 valuation observations used for the copula regime with 10Y Treasury rate >= 3%
2,384 rows in the full runtime dataset
The one-year realised alpha target is filtered at:
-200% <= real_alpha <= +200%
This filter is used to reduce the effect of extreme values and unstable tails in the realised one-year alpha distribution.
Validation snapshot
The market-side model is validated with repeated out-of-fold validation.
Latest stable OOF result:
RMSE:     0.3970
Spearman: 0.3937
Kendall:  0.2762
The pure market model can slightly outperform the fused output on simple rank-correlation metrics. However, the copula layer is still central to the project because it changes the object being predicted.
A pure ML model gives one value. The copula fusion layer gives a conditional distribution.
That distribution is the main research value of the project.
Pipeline overview
The project has two main workflows:
1.	Research / training pipeline
2.	Runtime / report-generation pipeline
1. Research pipeline
The full dataset and model pipeline follows this order:
parser.py
-> train_builder.py
-> ml.py
-> copula_logic.py
parser.py
Parses and builds the raw per-ticker dataset.
Main responsibilities:
•	Load the US ticker universe.
•	Download market data through yfinance.
•	Pull annual fundamentals from SEC EDGAR company facts.
•	Compute price returns, market returns, beta, volatility, log-volume, and other market features.
•	Parse or infer fundamental fields such as revenue, EBIT, net income, assets, equity, debt, cash, capex, operating cash flow, shares outstanding, market cap, enterprise value, net debt, and FCFF.
•	Save ticker-level time series into the local dataset structure.
train_builder.py
Builds the valuation-year research dataset.
Main responsibilities:
•	Iterate through parsed tickers and valuation years.
•	Build one valuation observation per ticker-year.
•	Compute DCF-implied upside.
•	Compute realised one-year market-relative alpha.
•	Attach the relevant 10Y Treasury rate.
•	Filter invalid rows and extreme values.
•	Save ticker-year JSON files.
•	Aggregate market features into a flat ML-ready dataset.
The target variable is:
real_alpha = one-year stock return - one-year S&P 500 return
ml.py
Trains the market-side alpha model.
Main responsibilities:
•	Load the flattened training dataset.
•	Train a CatBoost regression model.
•	Tune model parameters with Optuna.
•	Produce repeated out-of-fold predictions.
•	Save the final alpha_hat dataset.
•	Save the trained market model.
The model intentionally uses market and regime features. DCF-implied upside is kept separate and is not simply injected as another ML feature.
copula_logic.py
Fits the probabilistic fusion layer.
Main responsibilities:
•	Build empirical marginal distributions for realised alpha, market-side alpha_hat, and DCF-implied upside.
•	Estimate Clayton-based dependence parameters.
•	Save the copula configuration.
•	Produce conditional distribution metrics for runtime inference.
This layer is the methodological core of the project. It allows the system to answer questions like:
Given the market model's view and the DCF signal,
what is the conditional probability that alpha will be positive?
instead of only:
What is the predicted alpha?
Runtime workflow
The runtime workflow is used to generate a single-ticker report.
main.py
-> modules_processor.py
-> report_export.py
-> PowerPoint macro template
-> PDF report
main.py
Collects user inputs:
ticker
DCF-implied upside
current 10Y Treasury rate
Then it sends those inputs into the report-generation pipeline.
modules_processor.py
Builds the live feature row for the selected ticker.
Main responsibilities:
•	Parse recent market data.
•	Build the same market features used by the ML model.
•	Load the trained CatBoost model.
•	Estimate live alpha_hat.
•	Load the copula configuration.
•	Produce the final conditional alpha distribution.
report_export.py
Generates the final report.
Main responsibilities:
•	Prepare report parameters.
•	Copy the PowerPoint report template.
•	Run the VBA macro.
•	Export the report as a PDF.
•	Save the final output into the Reports directory.
Report output
The final report is designed to be read like a compact research summary.
It includes:
•	ticker,
•	valuation date,
•	alpha horizon,
•	probability of positive alpha,
•	expected alpha,
•	median alpha,
•	downside and upside conditional ranges,
•	tail probabilities,
•	Sharpe-like ratio,
•	football-field style conditional distribution visualisation.
The football field is not a deterministic valuation range. It is a visual representation of the conditional alpha distribution estimated by the fusion layer.
Data sources
The project uses public and research-friendly data sources:
•	SEC EDGAR company facts for annual fundamentals.
•	yfinance for market prices, index data, volume, shares, and selected reference fields.
•	10Y Treasury rate as the main macro regime variable.
•	WACC-related inputs based on public US-market assumptions, including risk-free rate, implied equity risk premium, US marginal tax rate, and credit spread proxy.
The project is research-only and does not use proprietary market data.
Repository structure
SimplePortfolioManager/
|
|-- Datasets/
|   |-- companies.csv
|   |-- US_WaccComponents_Timeseries.csv
|   |-- train_dataset.csv
|   `-- train_dataset_full.csv
|
|-- Models/
|   |-- market_model.cbm
|   `-- copula_cfg.json
|
|-- Reports/
|   `-- report_template.pptm
|
|-- parser.py
|-- train_builder.py
|-- ml.py
|-- copula_logic.py
|-- modules_processor.py
|-- valuation_dcf.py
|-- report_export.py
|-- main.py
`-- requirements.txt
Large intermediate parsing folders and ticker-year JSON folders are not meant to be treated as the main public artefacts. They can be regenerated by running the research pipeline.
Setup
1. Clone the repository
git clone https://github.com/SamsonChyst/SimplePortfolioManager.git
cd SimplePortfolioManager
2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate
On Windows:
.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Create .env
Create a .env file in the project root:
Mail="yourmail@gmail.com"
This is used for SEC-friendly parsing. Automated SEC requests should identify the user agent properly and should not be blocked by the system or network environment.
5. PowerPoint requirements
PDF report generation uses a PowerPoint macro-enabled template.
You need:
•	Microsoft PowerPoint installed,
•	macro permissions enabled,
•	access to Reports/report_template.pptm.
During the first run, PowerPoint may ask for macro permissions. They need to be allowed for the report export workflow to work.
Running the research pipeline
To rebuild the full dataset and models from scratch, the intended order is:
python parser.py
python train_builder.py
python ml.py
python copula_logic.py
This will:
1.	parse ticker-level data,
2.	build valuation-year rows,
3.	train the market-side alpha model,
4.	build the copula fusion configuration.
Depending on the size of the ticker universe and API availability, the parsing step can take a long time.
Running a single-ticker report
After the required datasets and model artefacts are available, run:
python main.py
The script asks for:
Ticker
DCF-implied upside
Current 10Y Treasury rate
Then it generates a PDF report in the Reports directory.
Important limitations
This is a research project, not investment advice.
It is also not production trading infrastructure. The project does not currently include:
•	portfolio optimisation,
•	position sizing,
•	transaction-cost modelling,
•	broker execution,
•	live risk monitoring,
•	production-grade data validation,
•	institutional data licensing.
The model should be treated as an analytical assistant. Its purpose is to make fundamental analysis more structured and probabilistic, not to replace human judgement.
Why this project exists
The project was built to gain practical experience with a realistic quant-style research workflow:
•	data parsing,
•	financial statement processing,
•	DCF modelling,
•	WACC assumptions,
•	market feature engineering,
•	machine learning,
•	out-of-fold validation,
•	copula-based dependence modelling,
•	probabilistic forecasting,
•	automated PDF report generation.
The main learning goal was to connect fundamental valuation with market-regime modelling and to turn both into a usable probability-based research output.
In short:
DCF gives a fundamental signal.
ML gives a market-regime signal.
The copula fusion layer turns both into a conditional alpha distribution.
That is the core idea behind SimplePortfolioManager.
