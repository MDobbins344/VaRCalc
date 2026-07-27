# VaRCalc

A Python tool for calculating **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)** for individual stocks and multi-asset portfolios, with an interactive Streamlit dashboard for visual analysis.

VaRCalc pulls historical price data from Yahoo Finance and estimates the potential loss a position could experience over a given time horizon, using three complementary VaR methodologies plus tail-risk analysis.

## Features

- **Multiple VaR methods**
  - **Historical VaR** — empirical percentile of actual past returns
  - **Parametric VaR** — closed-form estimate assuming normally distributed returns
  - **Monte Carlo VaR** — simulation-based estimate (10,000 draws) from a fitted normal distribution
  - **Conditional VaR (CVaR / Expected Shortfall)** — average loss in the tail beyond the VaR threshold
- **Single-stock or portfolio analysis** — analyze one ticker, or a weighted portfolio of up to 10 tickers
- **Diversification insights** — compares portfolio VaR against the weighted average of individual holdings to highlight diversification benefit (or hidden correlation risk)
- **Visualizations** — return distribution histograms with VaR/CVaR overlays, method-comparison bar charts, portfolio-vs-individual risk charts, VaR-vs-CVaR tail-risk charts, rolling VaR/volatility over time, and a combined dashboard view
- **Interactive Streamlit app** — configure tickers, weights, portfolio value, date range, and confidence level, then get a full risk report with plain-English interpretation and recommendations

## Project Structure

```
VaRCalc/
├── gui/
│   ├── app.py              # Streamlit application (main entry point)
│   └── components.py       # (reserved for reusable UI components)
├── src/
│   ├── data_processing.py  # DataProcessor: fetches prices via yfinance, computes returns
│   ├── calculations.py     # Historical, Parametric, Monte Carlo VaR and CVaR
│   ├── visualizations.py   # Matplotlib/Seaborn plotting functions
│   └── utils.py            # (reserved for shared helpers)
├── tests/                  # Unit tests (in progress)
├── docs/                   # Additional documentation (in progress)
├── data/                   # Sample/local data
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.9+
- Internet access (for fetching live price data via Yahoo Finance)

### Installation

```bash
git clone https://github.com/<your-username>/VaRCalc.git
cd VaRCalc

python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Running the App

Launch the Streamlit dashboard from the project root:

```bash
streamlit run gui/app.py
```

Then in the browser UI:

1. Choose **Single Stock** or **Portfolio** analysis
2. Enter ticker symbol(s) (and weights, for a portfolio)
3. Set the total portfolio value, date range, and confidence level
4. Click **Calculate VaR** to view metrics, comparison tables, and visualizations

### Using the Library Directly

The core calculations can also be used without the GUI:

```python
from src.data_processing import DataProcessor
from src.calculations import historical_var, parametric_var, monte_carlo_var, conditional_var

handler = DataProcessor()
prices = handler.fetch_data("AAPL", "2023-01-01", "2024-01-01")
returns = handler.calculate_returns(prices)

hist_var = historical_var(returns, confidence_level=0.95)
param_var = parametric_var(returns, confidence_level=0.95)
mc_var = monte_carlo_var(returns, confidence_level=0.95, num_simulations=10000)
cvar = conditional_var(returns, hist_var)

print(f"Historical VaR (95%): {hist_var:.2%}")
print(f"Conditional VaR (95%): {cvar:.2%}")
```

## Dependencies

- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) — data manipulation
- [scipy](https://scipy.org/) — statistical functions (normal distribution, z-scores)
- [yfinance](https://github.com/ranaroussi/yfinance) — historical market data retrieval
- [matplotlib](https://matplotlib.org/) / [seaborn](https://seaborn.pydata.org/) — visualizations
- [streamlit](https://streamlit.io/) — web app UI

See `requirements.txt` for pinned versions.

## Project Status

This project is under active development. Core VaR/CVaR calculations and visualizations are functional; the GUI, test suite, and documentation are still being built out. Currently investing API alternatives Polygon.io and Alpha Vantage to replace the yfinance library.

## Disclaimer

This tool is intended for **educational purposes** to demonstrate financial risk analysis concepts. It is not investment advice, and past performance/statistical estimates do not guarantee future results.
