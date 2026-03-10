# 📈 Derivatives Pricing & Volatility Toolkit

An interactive platform for options pricing, volatility analysis, strategy building, and structured product comparison. Built with Python and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Quick Start

```bash
git clone https://github.com/your-username/derivatives-toolkit.git
cd derivatives-toolkit
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Modules

### Module A — Vanilla Option Pricer

Price European options with two independent methods and full risk analysis.

- **Black-Scholes** closed-form pricing with analytical Greeks
- **Monte Carlo** simulation (GBM, configurable up to 1M paths) with finite-difference Greeks
- **Interactive Greek profiles** — Delta, Gamma, Vega, Theta, Rho across spot prices
- **2D sensitivity heatmaps** — any parameter pair (e.g. Spot × Vol → Price)
- **Time decay visualization** — theta acceleration near expiry
- **P&L decomposition** — waterfall chart breaking down P&L into Greek contributions
- **BS vs MC convergence** — validate simulation against analytical solution

### Module B — Implied Volatility & Vol Surface

Extract implied volatilities from live market data and visualize the volatility surface.

- **Real-time option chain data** from Yahoo Finance (via yfinance)
- **IV solver** using Brent's root-finding method on the BS inverse
- **Volatility smile** — IV across strikes for a single expiry
- **3D volatility surface** — interactive Strike × Maturity × IV visualization
- **ATM term structure** — how ATM vol changes across maturities
- **Multi-expiry skew comparison** — overlay smiles on a moneyness axis

Supported tickers: SPY, AAPL, TSLA, QQQ, MSFT, GLD, and any US equity with listed options.

### Module C — Strategy Builder & Payoff Visualizer

Build and analyze multi-leg options strategies.

- **12 preset strategies**: Long Call, Long Put, Covered Call, Protective Put, Bull Call Spread, Bear Put Spread, Long Straddle, Short Straddle, Long Strangle, Iron Condor, Butterfly Spread, Collar
- **Custom builder** — up to 6 legs, any combination of calls/puts, long/short
- **Payoff diagram** — P&L at expiry + pre-expiry curves at multiple time slices
- **Strategy Greeks** — aggregate Delta, Gamma, Vega, Theta across spot prices
- **Leg decomposition** — see how individual legs contribute to the total payoff
- **Key metrics** — max profit, max loss, breakevens, net premium, net delta

### Module D — Turbo / Knock-Out vs Vanilla Comparison

Compare leveraged knock-out products against vanilla options.

- **Turbo Put pricing** using the Rubinstein-Reiner barrier option formula
- **P&L comparison** — Turbo vs Vanilla put with knock-out zone visualization
- **Leverage profile** — how leverage changes as spot approaches the barrier
- **Scenario analysis** — return comparison across ±20% spot moves
- **Time decay comparison** — different theta dynamics for barrier vs vanilla
- **Interview talking points** — key insights for finance interview discussions

---

## Technical Details

### Pricing Methods

| Method | Description | Used in |
|--------|-------------|---------|
| Black-Scholes | Closed-form analytical solution | Module A, B, C, D |
| Monte Carlo | GBM simulation with antithetic variates | Module A |
| Brent's method | Root-finding for IV extraction | Module B |
| Rubinstein-Reiner | Closed-form barrier option pricing | Module D |

### Greeks Convention

| Greek | Definition | Unit |
|-------|-----------|------|
| Delta (Δ) | ∂V/∂S | Per $1 spot move |
| Gamma (Γ) | ∂²V/∂S² | Per $1 spot move |
| Vega (ν) | ∂V/∂σ | Per 1% vol move |
| Theta (Θ) | ∂V/∂t | Per calendar day |
| Rho (ρ) | ∂V/∂r | Per 1% rate move |

### Validation

- **Put-call parity**: Call - Put = S - K×e^(-rT), verified to 4 decimal places
- **MC convergence**: <$0.05 error at 100k simulations vs BS analytical
- **Boundary conditions**: Delta → 1 (deep ITM call), Delta → 0 (deep OTM call), Gamma peak at ATM
- **IV round-trip**: Solved IV reproduces market prices within bid-ask spread

---

## Project Structure

```
derivatives-toolkit/
├── app.py              # Single-file Streamlit application (all 4 modules)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Tech Stack

- **Python 3.10+** — Core language
- **NumPy / SciPy** — Numerical computation, optimization, statistics
- **Plotly** — Interactive charts (3D surfaces, heatmaps, waterfall charts)
- **Streamlit** — Web interface with reactive widgets
- **yfinance** — Real-time market data from Yahoo Finance

## Requirements

```
numpy>=1.24
scipy>=1.10
plotly>=5.15
streamlit>=1.28
yfinance>=0.2.28
```

---

## Author

**Quentin Nadouce**  
M1 PGE — Grenoble École de Management  
Exchange semester at Trinity College Dublin

Portfolio project demonstrating applied financial engineering:
derivatives pricing theory, Python development, data visualization,
and practical market knowledge from managing a personal equity portfolio (PEA).

---

*This tool is for educational and demonstration purposes. It does not constitute financial advice.*
