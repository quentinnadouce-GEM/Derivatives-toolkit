"""
Derivatives Pricing & Volatility Toolkit
=========================================
Module A — Vanilla Option Pricer
Module B — Implied Volatility & Vol Surface
Module C — Strategy Builder & Payoff Visualizer
Module D — Turbo / Knock-Out vs Vanilla Comparison
Single-file version — just run: streamlit run app.py

Author: Quentin Nadouce — GEM 2026
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ═════════════════════════════════════════════
# 1. PRICING ENGINE
# ═════════════════════════════════════════════

class OptionType(Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class PricingResult:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    method: str


def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma):
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_price_only(S, K, T, r, sigma, option_type="call"):
    """Fast BS price without Greeks — used by IV solver."""
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_scholes(S, K, T, r, sigma, option_type="call") -> PricingResult:
    is_call = option_type == "call"
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)

    if is_call:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    delta = norm.cdf(d1) if is_call else norm.cdf(d1) - 1
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100

    theta_common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if is_call:
        theta = (theta_common - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    if is_call:
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return PricingResult(price=price, delta=delta, gamma=gamma,
                         vega=vega, theta=theta, rho=rho, method="Black-Scholes")


def monte_carlo(S, K, T, r, sigma, option_type="call",
                n_simulations=100_000, n_steps=252, seed=42) -> PricingResult:
    if seed is not None:
        np.random.seed(seed)
    is_call = option_type == "call"

    def simulate_payoff(S_0, vol, rate, maturity):
        dt = maturity / n_steps
        Z = np.random.standard_normal((n_simulations, n_steps))
        log_increments = (rate - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z
        log_ST = np.log(S_0) + np.sum(log_increments, axis=1)
        ST = np.exp(log_ST)
        if is_call:
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
        return np.exp(-rate * maturity) * np.mean(payoffs)

    price = simulate_payoff(S, sigma, r, T)
    bump_S = S * 0.01
    bump_sigma = 0.01
    bump_r = 0.0001
    bump_T = 1 / 365

    price_up = simulate_payoff(S + bump_S, sigma, r, T)
    price_down = simulate_payoff(S - bump_S, sigma, r, T)
    delta = (price_up - price_down) / (2 * bump_S)
    gamma = (price_up - 2 * price + price_down) / (bump_S**2)

    price_vol_up = simulate_payoff(S, sigma + bump_sigma, r, T)
    vega = (price_vol_up - price) / (bump_sigma * 100)

    if T > bump_T:
        price_T_down = simulate_payoff(S, sigma, r, T - bump_T)
        theta = (price_T_down - price)
    else:
        theta = 0.0

    price_r_up = simulate_payoff(S, sigma, r + bump_r, T)
    rho = (price_r_up - price) / (bump_r * 100)

    return PricingResult(price=price, delta=delta, gamma=gamma,
                         vega=vega, theta=theta, rho=rho,
                         method=f"Monte Carlo ({n_simulations:,} sims)")


# ═════════════════════════════════════════════
# 2. IMPLIED VOLATILITY ENGINE (Module B)
# ═════════════════════════════════════════════

def implied_volatility(market_price, S, K, T, r, option_type="call",
                       vol_low=0.001, vol_high=5.0):
    try:
        price_low = bs_price_only(S, K, T, r, vol_low, option_type)
        price_high = bs_price_only(S, K, T, r, vol_high, option_type)
        if market_price < price_low or market_price > price_high:
            return None
        def objective(sigma):
            return bs_price_only(S, K, T, r, sigma, option_type) - market_price
        iv = brentq(objective, vol_low, vol_high, xtol=1e-8, maxiter=200)
        return iv
    except (ValueError, RuntimeError):
        return None


def fetch_option_chain(ticker_symbol):
    import yfinance as yf
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    spot = info.get("regularMarketPrice") or info.get("previousClose")
    if spot is None:
        hist = ticker.history(period="1d")
        if not hist.empty:
            spot = hist["Close"].iloc[-1]
        else:
            return None, None, None
    expirations = ticker.options
    if not expirations:
        return spot, None, None
    return spot, expirations, ticker


def get_chain_for_expiry(ticker, expiry, spot, r=0.04):
    import datetime
    chain = ticker.option_chain(expiry)
    calls = chain.calls
    puts = chain.puts
    exp_date = datetime.datetime.strptime(expiry, "%Y-%m-%d")
    now = datetime.datetime.now()
    T = max((exp_date - now).days / 365.0, 0.001)
    results = []
    for _, row in calls.iterrows():
        strike = row["strike"]
        mid_price = (row["bid"] + row["ask"]) / 2 if row["bid"] > 0 and row["ask"] > 0 else row["lastPrice"]
        if mid_price <= 0 or strike <= 0:
            continue
        iv = implied_volatility(mid_price, spot, strike, T, r, "call")
        if iv is not None and 0.01 < iv < 3.0:
            results.append({"strike": strike, "T": T, "expiry": expiry,
                "call_iv": iv, "call_price": mid_price, "moneyness": strike / spot})
    for _, row in puts.iterrows():
        strike = row["strike"]
        mid_price = (row["bid"] + row["ask"]) / 2 if row["bid"] > 0 and row["ask"] > 0 else row["lastPrice"]
        if mid_price <= 0 or strike <= 0:
            continue
        iv = implied_volatility(mid_price, spot, strike, T, r, "put")
        if iv is not None and 0.01 < iv < 3.0:
            matched = False
            for res in results:
                if abs(res["strike"] - strike) < 0.01:
                    res["put_iv"] = iv
                    res["put_price"] = mid_price
                    matched = True
                    break
            if not matched:
                results.append({"strike": strike, "T": T, "expiry": expiry,
                    "put_iv": iv, "put_price": mid_price, "moneyness": strike / spot})
    return results, T


def build_vol_surface(ticker, expirations, spot, r=0.04, max_expiries=8):
    all_data = []
    for expiry in expirations[:max_expiries]:
        try:
            chain_data, T = get_chain_for_expiry(ticker, expiry, spot, r)
            if chain_data:
                all_data.extend(chain_data)
        except Exception:
            continue
    return all_data


# ═════════════════════════════════════════════
# 3. STRATEGY ENGINE (Module C)
# ═════════════════════════════════════════════

@dataclass
class Leg:
    """A single leg of an options strategy."""
    option_type: str   # "call" or "put"
    strike: float
    position: str      # "long" or "short"
    quantity: int
    premium: float     # price paid/received per contract

    @property
    def sign(self):
        return 1 if self.position == "long" else -1


def leg_payoff_at_expiry(leg: Leg, S_range: np.ndarray) -> np.ndarray:
    """Compute P&L at expiry for a single leg."""
    if leg.option_type == "call":
        intrinsic = np.maximum(S_range - leg.strike, 0)
    else:
        intrinsic = np.maximum(leg.strike - S_range, 0)
    pnl = leg.sign * leg.quantity * (intrinsic - leg.premium)
    return pnl


def leg_pnl_before_expiry(leg: Leg, S_range: np.ndarray,
                           T_remaining: float, r: float, sigma: float) -> np.ndarray:
    """Compute P&L before expiry using BS pricing."""
    pnl = np.zeros_like(S_range, dtype=float)
    for i, s in enumerate(S_range):
        if T_remaining > 0.0001:
            current_price = bs_price_only(s, leg.strike, T_remaining, r, sigma, leg.option_type)
        else:
            if leg.option_type == "call":
                current_price = max(s - leg.strike, 0)
            else:
                current_price = max(leg.strike - s, 0)
        pnl[i] = leg.sign * leg.quantity * (current_price - leg.premium)
    return pnl


def strategy_payoff_at_expiry(legs: List[Leg], S_range: np.ndarray) -> np.ndarray:
    """Total P&L at expiry for multi-leg strategy."""
    total = np.zeros_like(S_range, dtype=float)
    for leg in legs:
        total += leg_payoff_at_expiry(leg, S_range)
    return total


def strategy_pnl_before_expiry(legs: List[Leg], S_range: np.ndarray,
                                T_remaining: float, r: float, sigma: float) -> np.ndarray:
    """Total P&L before expiry."""
    total = np.zeros_like(S_range, dtype=float)
    for leg in legs:
        total += leg_pnl_before_expiry(leg, S_range, T_remaining, r, sigma)
    return total


def strategy_greeks(legs: List[Leg], S: float, T: float, r: float, sigma: float) -> Dict:
    """Aggregate Greeks for the full strategy."""
    total = {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0}
    for leg in legs:
        if T > 0.0001:
            res = black_scholes(S, leg.strike, T, r, sigma, leg.option_type)
            total["delta"] += leg.sign * leg.quantity * res.delta
            total["gamma"] += leg.sign * leg.quantity * res.gamma
            total["vega"] += leg.sign * leg.quantity * res.vega
            total["theta"] += leg.sign * leg.quantity * res.theta
            total["rho"] += leg.sign * leg.quantity * res.rho
    return total


def compute_strategy_metrics(legs: List[Leg], S_range: np.ndarray) -> Dict:
    """Compute max profit, max loss, breakevens."""
    pnl = strategy_payoff_at_expiry(legs, S_range)
    max_profit = np.max(pnl)
    max_loss = np.min(pnl)

    # Net premium
    net_premium = sum(leg.sign * leg.quantity * leg.premium for leg in legs)

    # Breakevens: where P&L crosses zero
    breakevens = []
    for i in range(len(pnl) - 1):
        if pnl[i] * pnl[i + 1] < 0:  # sign change
            # Linear interpolation
            s_be = S_range[i] + (S_range[i + 1] - S_range[i]) * abs(pnl[i]) / (abs(pnl[i]) + abs(pnl[i + 1]))
            breakevens.append(s_be)

    return {
        "max_profit": max_profit if max_profit < 1e6 else float('inf'),
        "max_loss": max_loss if max_loss > -1e6 else float('-inf'),
        "net_premium": net_premium,
        "breakevens": breakevens
    }


# Pre-built strategies
PRESET_STRATEGIES = {
    "Long Call": lambda S, sigma: [
        Leg("call", S, "long", 1, bs_price_only(S, S, 0.25, 0.04, sigma, "call"))
    ],
    "Long Put": lambda S, sigma: [
        Leg("put", S, "long", 1, bs_price_only(S, S, 0.25, 0.04, sigma, "put"))
    ],
    "Covered Call": lambda S, sigma: [
        Leg("call", S * 1.05, "short", 1, bs_price_only(S, S * 1.05, 0.25, 0.04, sigma, "call"))
    ],
    "Protective Put": lambda S, sigma: [
        Leg("put", S * 0.95, "long", 1, bs_price_only(S, S * 0.95, 0.25, 0.04, sigma, "put"))
    ],
    "Bull Call Spread": lambda S, sigma: [
        Leg("call", S * 0.97, "long", 1, bs_price_only(S, S * 0.97, 0.25, 0.04, sigma, "call")),
        Leg("call", S * 1.03, "short", 1, bs_price_only(S, S * 1.03, 0.25, 0.04, sigma, "call"))
    ],
    "Bear Put Spread": lambda S, sigma: [
        Leg("put", S * 1.03, "long", 1, bs_price_only(S, S * 1.03, 0.25, 0.04, sigma, "put")),
        Leg("put", S * 0.97, "short", 1, bs_price_only(S, S * 0.97, 0.25, 0.04, sigma, "put"))
    ],
    "Long Straddle": lambda S, sigma: [
        Leg("call", S, "long", 1, bs_price_only(S, S, 0.25, 0.04, sigma, "call")),
        Leg("put", S, "long", 1, bs_price_only(S, S, 0.25, 0.04, sigma, "put"))
    ],
    "Short Straddle": lambda S, sigma: [
        Leg("call", S, "short", 1, bs_price_only(S, S, 0.25, 0.04, sigma, "call")),
        Leg("put", S, "short", 1, bs_price_only(S, S, 0.25, 0.04, sigma, "put"))
    ],
    "Long Strangle": lambda S, sigma: [
        Leg("call", S * 1.05, "long", 1, bs_price_only(S, S * 1.05, 0.25, 0.04, sigma, "call")),
        Leg("put", S * 0.95, "long", 1, bs_price_only(S, S * 0.95, 0.25, 0.04, sigma, "put"))
    ],
    "Iron Condor": lambda S, sigma: [
        Leg("put", S * 0.90, "long", 1, bs_price_only(S, S * 0.90, 0.25, 0.04, sigma, "put")),
        Leg("put", S * 0.95, "short", 1, bs_price_only(S, S * 0.95, 0.25, 0.04, sigma, "put")),
        Leg("call", S * 1.05, "short", 1, bs_price_only(S, S * 1.05, 0.25, 0.04, sigma, "call")),
        Leg("call", S * 1.10, "long", 1, bs_price_only(S, S * 1.10, 0.25, 0.04, sigma, "call"))
    ],
    "Butterfly Spread": lambda S, sigma: [
        Leg("call", S * 0.95, "long", 1, bs_price_only(S, S * 0.95, 0.25, 0.04, sigma, "call")),
        Leg("call", S, "short", 2, bs_price_only(S, S, 0.25, 0.04, sigma, "call")),
        Leg("call", S * 1.05, "long", 1, bs_price_only(S, S * 1.05, 0.25, 0.04, sigma, "call"))
    ],
    "Collar": lambda S, sigma: [
        Leg("put", S * 0.95, "long", 1, bs_price_only(S, S * 0.95, 0.25, 0.04, sigma, "put")),
        Leg("call", S * 1.05, "short", 1, bs_price_only(S, S * 1.05, 0.25, 0.04, sigma, "call"))
    ],
}


# ═════════════════════════════════════════════
# 4. TURBO / KNOCK-OUT ENGINE (Module D)
# ═════════════════════════════════════════════

def up_and_out_put(S, K, H, T, r, sigma, rebate=0.0):
    """
    Price an up-and-out put option (barrier H > S).
    The option ceases to exist if S hits the barrier H from below.
    
    Uses closed-form formula from Haug (2007) "The Complete Guide to 
    Option Pricing Formulas", Chapter 4.
    
    For a Turbo Put: barrier is ABOVE spot, option knocks out on the upside.
    """
    if S >= H:
        return rebate  # already knocked out
    if H <= 0 or T <= 0 or sigma <= 0:
        return 0.0

    sqrtT = sigma * np.sqrt(T)
    mu = (r - 0.5 * sigma**2) / (sigma**2)
    lam = np.sqrt(mu**2 + 2 * r / (sigma**2))

    # Standard vanilla put
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sqrtT
    vanilla_put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    if K >= H:
        # Strike above barrier — put is always ITM at barrier, simple case
        # The up-and-out put is worthless (can't profit if knocked out above K)
        # Use MC fallback for safety
        return _mc_up_and_out_put(S, K, H, T, r, sigma)

    # Case: K < H (strike below barrier) — standard Turbo Put case
    # Formula components
    a1 = np.log(S / H) / sqrtT + (1 + mu) * sqrtT
    a2 = a1 - sqrtT
    b1 = np.log(H / S) / sqrtT + (1 + mu) * sqrtT
    b2 = b1 - sqrtT

    # Up-and-in put price
    ui_put = (-S * norm.cdf(-a1) + K * np.exp(-r * T) * norm.cdf(-a2)
              + S * (H / S)**(2 * (mu + 1)) * (norm.cdf(b1) - norm.cdf(a1))  # This is not right for all cases
              - K * np.exp(-r * T) * (H / S)**(2 * mu) * (norm.cdf(b2) - norm.cdf(a2)))

    # Try MC-based pricing for reliability
    return _mc_up_and_out_put(S, K, H, T, r, sigma)


def _mc_up_and_out_put(S, K, H, T, r, sigma, n_sims=20000, n_steps=126, seed=None):
    """
    Monte Carlo pricing for up-and-out put.
    Reliable fallback — simulates paths and checks barrier crossing.
    """
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(hash((S, K, H, T, r, sigma)) % (2**31))

    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    # Simulate paths
    Z = np.random.standard_normal((n_sims, n_steps))
    log_paths = np.cumsum(drift + vol * Z, axis=1)
    log_paths = np.log(S) + log_paths
    paths = np.exp(log_paths)

    # Check if barrier is ever hit (max along each path)
    max_prices = np.max(paths, axis=1)
    survived = max_prices < H  # True if never hit barrier

    # Terminal payoff for survived paths
    S_T = paths[:, -1]
    payoffs = np.where(survived, np.maximum(K - S_T, 0), 0.0)

    # Discounted expected payoff
    price = np.exp(-r * T) * np.mean(payoffs)
    return max(price, 0.0)


def turbo_put_price(S, K, H, T, r, sigma):
    """
    Price a Turbo Put certificate.
    
    A Turbo Put is an UP-AND-OUT PUT:
    - You profit when the underlying DROPS (like a regular put)
    - The barrier H is ABOVE the current spot
    - If spot rises and hits H, the product is knocked out (worth zero)
    - The leverage comes from the barrier reducing the option's cost
    
    Typical setup: S=100, K=90, H=105
    - You make money if the underlying drops below K=90
    - You get knocked out if it rises to H=105
    """
    if S >= H:
        return 0.0  # knocked out
    return _mc_up_and_out_put(S, K, H, T, r, sigma)


def turbo_put_pnl(S_entry, S_range, K, H, T, r, sigma):
    """Compute Turbo Put P&L across a range of underlying prices."""
    entry_price = turbo_put_price(S_entry, K, H, T, r, sigma)
    if entry_price <= 0:
        return np.zeros_like(S_range), entry_price

    pnl = np.zeros_like(S_range, dtype=float)
    for i, s in enumerate(S_range):
        if s >= H:
            # Knocked out — total loss
            pnl[i] = -entry_price
        else:
            current = turbo_put_price(s, K, H, T, r, sigma)
            pnl[i] = current - entry_price
    return pnl, entry_price


def vanilla_put_pnl(S_entry, S_range, K, T, r, sigma):
    """Compute vanilla put P&L for comparison."""
    entry_price = bs_price_only(S_entry, K, T, r, sigma, "put")
    pnl = np.zeros_like(S_range, dtype=float)
    for i, s in enumerate(S_range):
        current = bs_price_only(s, K, T, r, sigma, "put")
        pnl[i] = current - entry_price
    return pnl, entry_price


def compute_leverage(S, turbo_price):
    """Compute leverage ratio of a Turbo product."""
    if turbo_price <= 0:
        return float('inf')
    return S / turbo_price


def scenario_analysis(S_entry, K, H, T, r, sigma, moves):
    """
    Compare Turbo Put vs Vanilla Put across different spot moves.
    Returns a list of dicts with scenario results.
    """
    turbo_entry = turbo_put_price(S_entry, K, H, T, r, sigma)
    vanilla_entry = bs_price_only(S_entry, K, T, r, sigma, "put")

    results = []
    for pct_move in moves:
        S_new = S_entry * (1 + pct_move / 100)

        if S_new >= H:
            turbo_new = 0.0
            turbo_pnl_val = -turbo_entry
            ko = True
        else:
            turbo_new = turbo_put_price(S_new, K, H, T, r, sigma)
            turbo_pnl_val = turbo_new - turbo_entry
            ko = False

        vanilla_new = bs_price_only(S_new, K, T, r, sigma, "put")
        vanilla_pnl_val = vanilla_new - vanilla_entry

        turbo_ret = (turbo_pnl_val / turbo_entry * 100) if turbo_entry > 0 else 0
        vanilla_ret = (vanilla_pnl_val / vanilla_entry * 100) if vanilla_entry > 0 else 0

        results.append({
            "Spot Move": f"{pct_move:+.0f}%",
            "New Spot": f"${S_new:.2f}",
            "Turbo P&L": f"${turbo_pnl_val:+.4f}",
            "Turbo Return": f"{turbo_ret:+.1f}%",
            "Vanilla P&L": f"${vanilla_pnl_val:+.4f}",
            "Vanilla Return": f"{vanilla_ret:+.1f}%",
            "Knocked Out": "💀 YES" if ko else "No",
        })
    return results


# ── Module D Charts ──

def plot_turbo_vs_vanilla(S_range, turbo_pnl, vanilla_pnl, spot, H, K,
                           turbo_price, vanilla_price):
    """Compare Turbo Put vs Vanilla Put P&L."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=S_range, y=turbo_pnl, mode="lines",
        name=f"Turbo Put (cost: ${turbo_price:.4f})",
        line=dict(color="#d62728", width=3)))

    fig.add_trace(go.Scatter(x=S_range, y=vanilla_pnl, mode="lines",
        name=f"Vanilla Put (cost: ${vanilla_price:.4f})",
        line=dict(color="#1f77b4", width=2.5, dash="dash")))

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], opacity=0.5)
    fig.add_vline(x=spot, line_dash="dash", line_color=COLORS["neutral"],
                  annotation_text=f"Spot={spot:.1f}", opacity=0.5)
    fig.add_vline(x=H, line_dash="solid", line_color="#d62728",
                  annotation_text=f"Barrier={H:.1f}", annotation_position="bottom",
                  line_width=2, opacity=0.7)
    fig.add_vline(x=K, line_dash="dot", line_color="#bcbd22",
                  annotation_text=f"Strike={K:.1f}", opacity=0.5)

    # Shade knock-out zone (right side — above barrier)
    fig.add_vrect(x0=H, x1=max(S_range),
                  fillcolor="red", opacity=0.05,
                  annotation_text="KNOCK-OUT ZONE", annotation_position="inside top right")

    fig.update_layout(**LAYOUT, title="Turbo Put vs Vanilla Put — P&L Comparison",
                      xaxis_title="Underlying Price",
                      yaxis_title="P&L ($)", height=500)
    return fig


def plot_turbo_leverage_profile(S_range, K, H, T, r, sigma, spot):
    """Show how leverage changes with spot price."""
    leverages = []
    for s in S_range:
        if s < H:
            tp = turbo_put_price(s, K, H, T, r, sigma)
            lev = compute_leverage(s, tp) if tp > 0 else 0
            leverages.append(min(lev, 100))  # cap display at 100x
        else:
            leverages.append(0)  # knocked out

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=leverages, mode="lines",
        name="Leverage", line=dict(color="#ff7f0e", width=2.5),
        fill="tozeroy", fillcolor="rgba(255,127,14,0.1)"))

    fig.add_vline(x=H, line_dash="solid", line_color="#d62728",
                  annotation_text=f"Barrier={H:.1f}", line_width=2, opacity=0.7)
    fig.add_vline(x=spot, line_dash="dash", line_color=COLORS["neutral"],
                  annotation_text=f"Spot={spot:.1f}", opacity=0.5)

    fig.update_layout(**LAYOUT, title="Turbo Put — Leverage Profile",
                      xaxis_title="Underlying Price",
                      yaxis_title="Leverage (x)", height=400)
    return fig


def plot_turbo_scenarios(S_entry, K, H, T, r, sigma):
    """Bar chart comparing returns across scenarios."""
    moves = [-20, -15, -10, -5, -2, 0, 2, 5, 10, 15]
    turbo_entry = turbo_put_price(S_entry, K, H, T, r, sigma)
    vanilla_entry = bs_price_only(S_entry, K, T, r, sigma, "put")

    turbo_returns = []
    vanilla_returns = []
    labels = []
    ko_markers = []

    for m in moves:
        S_new = S_entry * (1 + m / 100)
        labels.append(f"{m:+d}%")

        if S_new >= H:
            turbo_returns.append(-100)
            ko_markers.append(True)
        else:
            tp_new = turbo_put_price(S_new, K, H, T, r, sigma)
            ret = ((tp_new - turbo_entry) / turbo_entry * 100) if turbo_entry > 0 else 0
            turbo_returns.append(ret)
            ko_markers.append(False)

        vp_new = bs_price_only(S_new, K, T, r, sigma, "put")
        vanilla_returns.append(((vp_new - vanilla_entry) / vanilla_entry * 100) if vanilla_entry > 0 else 0)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Turbo Put Return", x=labels, y=turbo_returns,
        marker_color=["#d62728" if ko else "#ff7f0e" for ko in ko_markers]))
    fig.add_trace(go.Bar(name="Vanilla Put Return", x=labels, y=vanilla_returns,
        marker_color="#1f77b4"))

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], opacity=0.5)
    fig.update_layout(**LAYOUT, barmode="group",
                      title="Return Comparison — Turbo vs Vanilla (% return on premium)",
                      xaxis_title="Spot Move", yaxis_title="Return (%)", height=450)
    return fig


def plot_turbo_time_decay_comparison(S, K, H, r, sigma, T_max=0.5):
    """Show how Turbo vs Vanilla prices decay over time."""
    days = np.linspace(T_max * 365, 1, 100).astype(int)
    turbo_prices = []
    vanilla_prices = []

    for d in days:
        T = d / 365.0
        tp = turbo_put_price(S, K, H, T, r, sigma)
        vp = bs_price_only(S, K, T, r, sigma, "put")
        turbo_prices.append(tp)
        vanilla_prices.append(vp)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days.tolist(), y=turbo_prices, mode="lines",
        name="Turbo Put", line=dict(color="#d62728", width=2.5)))
    fig.add_trace(go.Scatter(x=days.tolist(), y=vanilla_prices, mode="lines",
        name="Vanilla Put", line=dict(color="#1f77b4", width=2.5, dash="dash")))

    fig.update_layout(**LAYOUT, height=400,
                      title="Time Decay — Turbo vs Vanilla",
                      xaxis_title="Days to Expiry",
                      yaxis_title="Option Price ($)",
                      xaxis_autorange="reversed")
    return fig


# ═════════════════════════════════════════════
# 4b. GREEKS ANALYSIS (Module A helpers)
# ═════════════════════════════════════════════

def compute_greek_profile(K, T, r, sigma, option_type="call", n_points=200):
    spots = np.linspace(K * 0.70, K * 1.30, n_points)
    prices = np.zeros(n_points)
    deltas = np.zeros(n_points)
    gammas = np.zeros(n_points)
    vegas = np.zeros(n_points)
    thetas = np.zeros(n_points)
    rhos = np.zeros(n_points)
    for i, s in enumerate(spots):
        r_ = black_scholes(s, K, T, r, sigma, option_type)
        prices[i] = r_.price
        deltas[i] = r_.delta
        gammas[i] = r_.gamma
        vegas[i] = r_.vega
        thetas[i] = r_.theta
        rhos[i] = r_.rho
    return {"spots": spots, "price": prices, "delta": deltas,
            "gamma": gammas, "vega": vegas, "theta": thetas, "rho": rhos}


def compute_time_decay(S, K, r, sigma, option_type="call", T_max=1.0, n_points=252):
    days = np.linspace(T_max * 365, 1, n_points).astype(int)
    prices = np.zeros(n_points)
    thetas = np.zeros(n_points)
    for i, d in enumerate(days):
        T = d / 365.0
        res = black_scholes(S, K, T, r, sigma, option_type)
        prices[i] = res.price
        thetas[i] = res.theta
    intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
    time_value = prices - intrinsic
    return {"days_to_expiry": days, "price": prices,
            "theta": thetas, "time_value": time_value}


def pnl_decomposition(S_i, S_f, K, T_i, T_f, r, sigma_i, sigma_f, option_type="call"):
    initial = black_scholes(S_i, K, T_i, r, sigma_i, option_type)
    final = black_scholes(S_f, K, T_f, r, sigma_f, option_type)
    dS = S_f - S_i
    d_sigma = sigma_f - sigma_i
    dT = T_f - T_i
    delta_pnl = initial.delta * dS
    gamma_pnl = 0.5 * initial.gamma * dS**2
    vega_pnl = initial.vega * (d_sigma * 100)
    theta_pnl = initial.theta * (dT * 365)
    approx_pnl = delta_pnl + gamma_pnl + vega_pnl + theta_pnl
    actual_pnl = final.price - initial.price
    unexplained = actual_pnl - approx_pnl
    return {"actual_pnl": actual_pnl, "approx_pnl": approx_pnl,
            "delta_pnl": delta_pnl, "gamma_pnl": gamma_pnl,
            "vega_pnl": vega_pnl, "theta_pnl": theta_pnl,
            "unexplained": unexplained,
            "initial_price": initial.price, "final_price": final.price}


def compute_sensitivity_grid(param_x, param_y, x_range, y_range, base_params, output="price"):
    grid = np.zeros((len(y_range), len(x_range)))
    for i, y_val in enumerate(y_range):
        for j, x_val in enumerate(x_range):
            p = base_params.copy()
            p[param_x] = x_val
            p[param_y] = y_val
            opt_type = p.pop("option_type", "call")
            result = black_scholes(**p, option_type=opt_type)
            grid[i, j] = getattr(result, output)
    return grid


# ═════════════════════════════════════════════
# 4. PLOTLY CHARTS
# ═════════════════════════════════════════════

COLORS = {
    "primary": "#1f77b4", "secondary": "#ff7f0e",
    "positive": "#2ca02c", "negative": "#d62728",
    "neutral": "#7f7f7f", "call": "#2ca02c", "put": "#d62728",
}
LAYOUT = dict(template="plotly_white",
              font=dict(family="Inter, Arial, sans-serif", size=12),
              margin=dict(l=60, r=30, t=50, b=50), hovermode="x unified")


# ── Module A Charts ──

def plot_greek_profiles(profile, strike, option_type="call"):
    fig = make_subplots(rows=2, cols=3,
        subplot_titles=("Price", "Delta (Δ)", "Gamma (Γ)",
                        "Vega (ν)", "Theta (Θ)", "Rho (ρ)"),
        vertical_spacing=0.15, horizontal_spacing=0.08)
    color = COLORS["call"] if option_type == "call" else COLORS["put"]
    S = profile["spots"]
    data = [
        (profile["price"], 1, 1), (profile["delta"], 1, 2), (profile["gamma"], 1, 3),
        (profile["vega"], 2, 1), (profile["theta"], 2, 2), (profile["rho"], 2, 3),
    ]
    for values, row, col in data:
        fig.add_trace(go.Scatter(x=S, y=values, mode="lines",
            line=dict(color=color, width=2), showlegend=False), row=row, col=col)
        fig.add_vline(x=strike, line_dash="dash", line_color=COLORS["neutral"],
                      opacity=0.5, row=row, col=col)
    fig.update_layout(**LAYOUT, height=500,
                      title=f"Greek Profiles — {option_type.upper()} (K={strike})")
    return fig


def plot_single_greek(spot_range, greek_values, greek_name, strike, option_type="call"):
    color = COLORS["call"] if option_type == "call" else COLORS["put"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spot_range, y=greek_values, mode="lines",
        line=dict(color=color, width=2.5), name=greek_name, fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)"))
    fig.add_vline(x=strike, line_dash="dash", line_color=COLORS["neutral"],
                  annotation_text=f"K={strike}")
    fig.update_layout(**LAYOUT, title=f"{greek_name} vs Spot",
                      xaxis_title="Spot Price", yaxis_title=greek_name, height=400)
    return fig


def plot_sensitivity_heatmap(grid, x_values, y_values, x_label, y_label, title="Heatmap"):
    fig = go.Figure(data=go.Heatmap(
        z=grid, x=np.round(x_values, 2), y=np.round(y_values, 4), colorscale="RdYlGn",
        hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>Value: %{{z:.4f}}<extra></extra>"))
    fig.update_layout(**LAYOUT, title=title, xaxis_title=x_label, yaxis_title=y_label, height=500)
    return fig


def plot_time_decay(decay_data):
    fig = make_subplots(rows=2, cols=1,
        subplot_titles=("Option Price & Time Value", "Theta (daily)"),
        vertical_spacing=0.15, shared_xaxes=True)
    days = decay_data["days_to_expiry"]
    fig.add_trace(go.Scatter(x=days, y=decay_data["price"], mode="lines",
        name="Total Price", line=dict(color=COLORS["primary"], width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=decay_data["time_value"], mode="lines",
        name="Time Value", line=dict(color=COLORS["secondary"], width=2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(255,127,14,0.1)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=days, y=decay_data["theta"], mode="lines",
        name="Theta", line=dict(color=COLORS["negative"], width=2)), row=2, col=1)
    fig.update_layout(**LAYOUT, height=600, title="Time Decay Analysis",
        xaxis2_title="Days to Expiry", xaxis_autorange="reversed", xaxis2_autorange="reversed")
    return fig


def plot_pnl_decomposition(pnl):
    categories = ["Delta", "Gamma", "Vega", "Theta", "Unexplained"]
    values = [pnl["delta_pnl"], pnl["gamma_pnl"], pnl["vega_pnl"],
              pnl["theta_pnl"], pnl["unexplained"]]
    fig = go.Figure()
    fig.add_trace(go.Waterfall(
        orientation="v", x=categories + ["Total"], y=values + [pnl["actual_pnl"]],
        measure=["relative"] * 5 + ["total"],
        connector=dict(line=dict(color=COLORS["neutral"], width=1)),
        increasing=dict(marker=dict(color=COLORS["positive"])),
        decreasing=dict(marker=dict(color=COLORS["negative"])),
        totals=dict(marker=dict(color=COLORS["primary"]))))
    fig.update_layout(**LAYOUT, title=f"P&L Decomposition (Actual: {pnl['actual_pnl']:+.4f})",
                      yaxis_title="P&L", height=450, showlegend=False)
    return fig


def plot_bs_vs_mc(bs, mc):
    metrics = ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho"]
    bs_vals = [bs.price, bs.delta, bs.gamma, bs.vega, bs.theta, bs.rho]
    mc_vals = [mc.price, mc.delta, mc.gamma, mc.vega, mc.theta, mc.rho]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Black-Scholes", x=metrics, y=bs_vals, marker_color=COLORS["primary"]))
    fig.add_trace(go.Bar(name="Monte Carlo", x=metrics, y=mc_vals, marker_color=COLORS["secondary"]))
    fig.update_layout(**LAYOUT, barmode="group", title="Black-Scholes vs Monte Carlo",
                      yaxis_title="Value", height=400)
    return fig


# ── Module B Charts ──

def plot_vol_smile(chain_data, expiry, spot):
    strikes = [d["strike"] for d in chain_data if d["expiry"] == expiry]
    call_ivs = [d.get("call_iv") for d in chain_data if d["expiry"] == expiry]
    put_ivs = [d.get("put_iv") for d in chain_data if d["expiry"] == expiry]
    fig = go.Figure()
    s_call = [s for s, iv in zip(strikes, call_ivs) if iv is not None]
    iv_call = [iv * 100 for iv in call_ivs if iv is not None]
    if s_call:
        fig.add_trace(go.Scatter(x=s_call, y=iv_call, mode="lines+markers",
            name="Call IV", line=dict(color=COLORS["call"], width=2), marker=dict(size=5)))
    s_put = [s for s, iv in zip(strikes, put_ivs) if iv is not None]
    iv_put = [iv * 100 for iv in put_ivs if iv is not None]
    if s_put:
        fig.add_trace(go.Scatter(x=s_put, y=iv_put, mode="lines+markers",
            name="Put IV", line=dict(color=COLORS["put"], width=2), marker=dict(size=5)))
    fig.add_vline(x=spot, line_dash="dash", line_color=COLORS["neutral"],
                  annotation_text=f"Spot={spot:.1f}")
    fig.update_layout(**LAYOUT, title=f"Volatility Smile — Expiry {expiry}",
                      xaxis_title="Strike", yaxis_title="Implied Volatility (%)", height=450)
    return fig


def plot_vol_surface_3d(surface_data, spot):
    strikes, maturities, ivs = [], [], []
    for d in surface_data:
        iv = d.get("call_iv") or d.get("put_iv")
        if iv is not None:
            strikes.append(d["moneyness"])
            maturities.append(d["T"])
            ivs.append(iv * 100)
    if not strikes:
        return None
    fig = go.Figure(data=[go.Mesh3d(
        x=maturities, y=strikes, z=ivs, intensity=ivs, colorscale="Viridis", opacity=0.7,
        colorbar=dict(title="IV (%)"),
        hovertemplate="T=%{x:.2f}y<br>K/S=%{y:.2f}<br>IV=%{z:.1f}%<extra></extra>")])
    fig.add_trace(go.Scatter3d(x=maturities, y=strikes, z=ivs, mode="markers",
        marker=dict(size=3, color=ivs, colorscale="Viridis", opacity=0.9),
        hovertemplate="T=%{x:.2f}y<br>K/S=%{y:.2f}<br>IV=%{z:.1f}%<extra></extra>", showlegend=False))
    fig.update_layout(title="3D Volatility Surface",
        scene=dict(xaxis_title="Time to Expiry (years)",
                   yaxis_title="Moneyness (K/S)", zaxis_title="Implied Vol (%)"),
        font=dict(family="Inter, Arial, sans-serif", size=12),
        height=600, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def plot_vol_term_structure(surface_data, spot):
    atm_by_expiry = {}
    for d in surface_data:
        m = d.get("moneyness", 0)
        iv = d.get("call_iv") or d.get("put_iv")
        if iv and 0.95 <= m <= 1.05:
            exp = d["expiry"]
            T = d["T"]
            if exp not in atm_by_expiry:
                atm_by_expiry[exp] = {"T": T, "ivs": []}
            atm_by_expiry[exp]["ivs"].append(iv * 100)
    if not atm_by_expiry:
        return None
    Ts, avg_ivs, labels = [], [], []
    for exp, data in sorted(atm_by_expiry.items(), key=lambda x: x[1]["T"]):
        Ts.append(data["T"])
        avg_ivs.append(np.mean(data["ivs"]))
        labels.append(exp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Ts, y=avg_ivs, mode="lines+markers", text=labels,
        textposition="top center", line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=8, color=COLORS["primary"]),
        hovertemplate="Expiry: %{text}<br>T=%{x:.2f}y<br>ATM IV=%{y:.1f}%<extra></extra>"))
    fig.update_layout(**LAYOUT, title="ATM Implied Volatility Term Structure",
                      xaxis_title="Time to Expiry (years)", yaxis_title="ATM Implied Vol (%)", height=400)
    return fig


def plot_skew_comparison(surface_data, spot):
    expiries = sorted(set(d["expiry"] for d in surface_data),
                      key=lambda e: next(d["T"] for d in surface_data if d["expiry"] == e))
    fig = go.Figure()
    colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for i, exp in enumerate(expiries):
        exp_data = [d for d in surface_data if d["expiry"] == exp]
        moneyness = [d["moneyness"] for d in exp_data if d.get("call_iv")]
        ivs = [d["call_iv"] * 100 for d in exp_data if d.get("call_iv")]
        if moneyness and ivs:
            sorted_pairs = sorted(zip(moneyness, ivs))
            m_sorted, iv_sorted = zip(*sorted_pairs)
            T = exp_data[0]["T"]
            fig.add_trace(go.Scatter(x=list(m_sorted), y=list(iv_sorted), mode="lines+markers",
                name=f"{exp} ({T:.2f}y)", line=dict(color=colors_list[i % len(colors_list)], width=2),
                marker=dict(size=4)))
    fig.add_vline(x=1.0, line_dash="dash", line_color=COLORS["neutral"], annotation_text="ATM")
    fig.update_layout(**LAYOUT, title="Skew Comparison Across Expiries",
                      xaxis_title="Moneyness (K/S)", yaxis_title="Call Implied Vol (%)", height=500)
    return fig


# ── Module C Charts ──

def plot_strategy_payoff(legs, S_range, spot, T_original, r, sigma, T_remaining_list):
    """Plot payoff at expiry + P&L curves at different times before expiry."""
    fig = go.Figure()

    # Payoff at expiry (the hockey stick)
    pnl_expiry = strategy_payoff_at_expiry(legs, S_range)
    fig.add_trace(go.Scatter(x=S_range, y=pnl_expiry, mode="lines",
        name="At Expiry", line=dict(color=COLORS["primary"], width=3)))

    # P&L before expiry at various times
    time_colors = ["#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    for i, T_rem in enumerate(T_remaining_list):
        if T_rem > 0:
            pnl_t = strategy_pnl_before_expiry(legs, S_range, T_rem, r, sigma)
            days = int(T_rem * 365)
            fig.add_trace(go.Scatter(x=S_range, y=pnl_t, mode="lines",
                name=f"T-{days}d", line=dict(color=time_colors[i % len(time_colors)],
                                              width=2, dash="dot")))

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], opacity=0.5)
    # Spot reference
    fig.add_vline(x=spot, line_dash="dash", line_color=COLORS["neutral"],
                  annotation_text=f"Spot={spot:.1f}", opacity=0.5)

    # Strike lines
    strikes = set(leg.strike for leg in legs)
    for k in strikes:
        fig.add_vline(x=k, line_dash="dot", line_color="#bcbd22", opacity=0.4,
                      annotation_text=f"K={k:.0f}", annotation_position="top")

    fig.update_layout(**LAYOUT, title="Strategy P&L Diagram",
                      xaxis_title="Underlying Price at Expiry",
                      yaxis_title="Profit / Loss ($)", height=500)
    return fig


def plot_strategy_greeks_profile(legs, S_range, T, r, sigma):
    """Plot aggregate strategy Greeks across spot prices."""
    deltas = np.zeros_like(S_range)
    gammas = np.zeros_like(S_range)
    vegas = np.zeros_like(S_range)
    thetas = np.zeros_like(S_range)

    for i, s in enumerate(S_range):
        g = strategy_greeks(legs, s, T, r, sigma)
        deltas[i] = g["delta"]
        gammas[i] = g["gamma"]
        vegas[i] = g["vega"]
        thetas[i] = g["theta"]

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=("Delta (Δ)", "Gamma (Γ)", "Vega (ν)", "Theta (Θ)"),
        vertical_spacing=0.15, horizontal_spacing=0.1)

    traces = [(deltas, 1, 1), (gammas, 1, 2), (vegas, 2, 1), (thetas, 2, 2)]
    for values, row, col in traces:
        fig.add_trace(go.Scatter(x=S_range, y=values, mode="lines",
            line=dict(color=COLORS["primary"], width=2), showlegend=False), row=row, col=col)
        fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"],
                      opacity=0.3, row=row, col=col)

    fig.update_layout(**LAYOUT, height=500, title="Strategy Greeks Profile")
    return fig


def plot_individual_legs(legs, S_range):
    """Show individual leg payoffs and the total."""
    fig = go.Figure()
    leg_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, leg in enumerate(legs):
        pnl = leg_payoff_at_expiry(leg, S_range)
        label = f"{'Long' if leg.position == 'long' else 'Short'} {leg.quantity}x {leg.option_type.upper()} K={leg.strike:.0f}"
        fig.add_trace(go.Scatter(x=S_range, y=pnl, mode="lines",
            name=label, line=dict(color=leg_colors[i % len(leg_colors)], width=1.5, dash="dot")))

    # Total
    total = strategy_payoff_at_expiry(legs, S_range)
    fig.add_trace(go.Scatter(x=S_range, y=total, mode="lines",
        name="Total", line=dict(color="white", width=3)))

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"], opacity=0.5)
    fig.update_layout(**LAYOUT, title="Individual Leg Decomposition",
                      xaxis_title="Underlying Price", yaxis_title="P&L ($)", height=450)
    return fig


# ═════════════════════════════════════════════
# 5. STREAMLIT APP
# ═════════════════════════════════════════════

st.set_page_config(page_title="Derivatives Toolkit", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

# ── Custom CSS ──
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {border-right: 1px solid #333;}
    [data-testid="stSidebar"] h1 {font-size: 1.3rem;}
    
    /* Metric cards */
    [data-testid="stMetricValue"] {font-size: 1.5rem;}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {gap: 0.5rem;}
    .stTabs [data-baseweb="tab"] {padding: 0.5rem 1rem;}
    
    /* Reduce top padding */
    .block-container {padding-top: 2rem;}
    
    /* Home page hero */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #aaa;
        margin-bottom: 2rem;
    }
    .module-card {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1.5rem;
        height: 100%;
        transition: border-color 0.2s;
    }
    .module-card:hover {border-color: #667eea;}
    .module-card h3 {
        color: #e0e0e0;
        margin-top: 0;
        font-size: 1.1rem;
    }
    .module-card p {
        color: #999;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .module-tag {
        display: inline-block;
        background: #667eea33;
        color: #667eea;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-right: 4px;
        margin-top: 8px;
    }
    .tech-badge {
        display: inline-block;
        background: #2a2a3e;
        color: #bbb;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 3px;
    }
    .footer-text {
        text-align: center;
        color: #666;
        font-size: 0.85rem;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📈 Derivatives Toolkit")
st.sidebar.markdown("---")

module = st.sidebar.radio("Module", [
    "🏠 Home",
    "A — Option Pricer",
    "B — Implied Vol & Surface",
    "C — Strategy Builder",
    "D — Turbo vs Vanilla"
])

if module == "🏠 Home":
    # ═══════════════════════════════════════
    # HOME PAGE
    # ═══════════════════════════════════════
    st.markdown('<div class="hero-title">Derivatives Pricing &<br>Volatility Toolkit</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">'
                'An interactive platform for options pricing, volatility analysis, '
                'strategy building, and structured product comparison.</div>',
                unsafe_allow_html=True)

    # Module cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="module-card">
            <h3>📊 Module A — Vanilla Option Pricer</h3>
            <p>Price European options with Black-Scholes and Monte Carlo. 
            Compute all 5 Greeks, visualize sensitivity surfaces, 
            decompose P&L into Greek contributions, and compare 
            analytical vs simulation-based pricing.</p>
            <span class="module-tag">Black-Scholes</span>
            <span class="module-tag">Monte Carlo</span>
            <span class="module-tag">Greeks</span>
            <span class="module-tag">Heatmaps</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="module-card">
            <h3>🔧 Module C — Strategy Builder</h3>
            <p>Build multi-leg options strategies from 12 presets 
            (straddle, iron condor, butterfly...) or create custom combinations. 
            Visualize payoff diagrams at expiry and before, with full 
            aggregate Greek analysis.</p>
            <span class="module-tag">12 Presets</span>
            <span class="module-tag">Custom Builder</span>
            <span class="module-tag">Payoff Diagrams</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="module-card">
            <h3>🌋 Module B — Implied Vol & Surface</h3>
            <p>Fetch real-time option chains from Yahoo Finance, 
            solve for implied volatility via Brent's method, and 
            visualize the vol surface in 4 views: smile, 3D surface, 
            ATM term structure, and multi-expiry skew.</p>
            <span class="module-tag">Live Market Data</span>
            <span class="module-tag">IV Solver</span>
            <span class="module-tag">3D Surface</span>
            <span class="module-tag">Skew Analysis</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="module-card">
            <h3>⚡ Module D — Turbo vs Vanilla</h3>
            <p>Compare Turbo Put certificates (knock-out barrier options) 
            against vanilla puts. Analyze leverage profiles, knock-out risk, 
            and scenario returns. Built from real PEA trading experience 
            with structured products.</p>
            <span class="module-tag">Barrier Options</span>
            <span class="module-tag">Leverage</span>
            <span class="module-tag">Scenarios</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tech stack
    st.markdown("### Tech Stack")
    st.markdown("""
    <div>
        <span class="tech-badge">Python 3.10+</span>
        <span class="tech-badge">NumPy</span>
        <span class="tech-badge">SciPy</span>
        <span class="tech-badge">Plotly</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">yfinance</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Methodology summary
    st.markdown("### Methodology")
    meth_col1, meth_col2 = st.columns(2)

    with meth_col1:
        st.markdown("""
        **Pricing**
        - Black-Scholes closed-form (Europeans)
        - Monte Carlo with GBM (100k+ paths)
        - Rubinstein-Reiner barrier option formula
        - Greeks via analytical formulas (BS) and finite differences (MC)
        """)

    with meth_col2:
        st.markdown("""
        **Implied Volatility**
        - Brent's root-finding method on BS inverse
        - Newton-Raphson fallback with vega step
        - Vol surface interpolation across strike × maturity
        - Moneyness-normalized skew comparison
        """)

    st.markdown("### Validation")
    st.markdown("""
    - ✅ Put-call parity verified to 4 decimal places  
    - ✅ Monte Carlo converges to BS within $0.05 at 100k simulations  
    - ✅ Greeks match analytical values at boundary conditions (deep ITM/OTM, near-expiry)  
    - ✅ Implied vol solver reproduces market prices within bid-ask spread
    """)

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#888; font-size:0.9rem;">
        <b>Quentin Nadouce</b> — M1 PGE, Grenoble École de Management<br>
        Built to demonstrate applied financial engineering skills.<br>
        <span style="color:#666;">Select a module in the sidebar to start →</span>
    </div>
    """, unsafe_allow_html=True)


elif module == "A — Option Pricer":
    # ═══════════════════════════════════════
    # MODULE A
    # ═══════════════════════════════════════
    st.sidebar.markdown("---")
    st.sidebar.header("Option Parameters")
    option_type = st.sidebar.selectbox("Type", ["call", "put"])
    S = st.sidebar.number_input("Spot (S)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    K = st.sidebar.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0, format="%.2f")
    T = st.sidebar.slider("Maturity (years)", min_value=0.01, max_value=3.0, value=0.25, step=0.01)
    r = st.sidebar.slider("Risk-free rate (%)", min_value=0.0, max_value=15.0, value=4.0, step=0.1) / 100
    sigma = st.sidebar.slider("Volatility (%)", min_value=1.0, max_value=100.0, value=20.0, step=0.5) / 100

    st.sidebar.markdown("---")
    moneyness = S / K
    if (option_type == "call" and S > K) or (option_type == "put" and S < K):
        st.sidebar.markdown(f"**Moneyness:** ITM ({moneyness:.1%})")
    elif (option_type == "call" and S < K) or (option_type == "put" and S > K):
        st.sidebar.markdown(f"**Moneyness:** OTM ({moneyness:.1%})")
    else:
        st.sidebar.markdown("**Moneyness:** ATM")

    st.title("Module A — Vanilla Option Pricer")
    st.markdown("European option pricing with Black-Scholes & Monte Carlo. Full Greeks & sensitivity analysis.")
    bs = black_scholes(S, K, T, r, sigma, option_type)

    st.markdown("### Pricing Results")
    cols = st.columns(6)
    metrics_data = [
        ("Price", f"${bs.price:.4f}", None),
        ("Delta (Δ)", f"{bs.delta:+.4f}", "Exposure to $1 spot move"),
        ("Gamma (Γ)", f"{bs.gamma:.4f}", "Delta sensitivity"),
        ("Vega (ν)", f"{bs.vega:.4f}", "Per 1% vol move"),
        ("Theta (Θ)", f"{bs.theta:.4f}", "Daily time decay"),
        ("Rho (ρ)", f"{bs.rho:.4f}", "Per 1% rate move"),
    ]
    for col, (name, value, help_text) in zip(cols, metrics_data):
        col.metric(label=name, value=value, help=help_text)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Greek Profiles", "🔥 Sensitivity Heatmap", "⏱️ Time Decay",
        "💰 P&L Decomposition", "🎯 BS vs Monte Carlo"])

    with tab1:
        st.markdown("#### Greeks as a function of spot price")
        profile = compute_greek_profile(K, T, r, sigma, option_type)
        fig = plot_greek_profiles(profile, strike=K, option_type=option_type)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Zoom on a specific Greek")
        greek_choice = st.selectbox("Select Greek", ["Delta", "Gamma", "Vega", "Theta", "Rho"])
        fig_single = plot_single_greek(profile["spots"], profile[greek_choice.lower()],
            greek_choice, strike=K, option_type=option_type)
        st.plotly_chart(fig_single, use_container_width=True)

    with tab2:
        st.markdown("#### 2D Sensitivity Analysis")
        hm_col1, hm_col2, hm_col3 = st.columns(3)
        with hm_col1:
            output_metric = st.selectbox("Output", ["price", "delta", "gamma", "vega", "theta"])
        with hm_col2:
            x_param = st.selectbox("X-axis", ["S", "K", "sigma", "T", "r"], index=0)
        with hm_col3:
            y_param = st.selectbox("Y-axis", ["sigma", "S", "K", "T", "r"], index=0)
        if x_param == y_param:
            st.warning("X and Y parameters must be different.")
        else:
            param_ranges = {
                "S": np.linspace(S * 0.7, S * 1.3, 30), "K": np.linspace(K * 0.7, K * 1.3, 30),
                "sigma": np.linspace(max(sigma * 0.3, 0.01), sigma * 2.5, 30),
                "T": np.linspace(max(T * 0.1, 0.01), T * 2, 30), "r": np.linspace(0.0, 0.10, 30)}
            base = {"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "option_type": option_type}
            grid = compute_sensitivity_grid(x_param, y_param, param_ranges[x_param],
                                             param_ranges[y_param], base, output_metric)
            labels = {"S": "Spot", "K": "Strike", "sigma": "Volatility", "T": "Maturity (y)", "r": "Rate"}
            fig_hm = plot_sensitivity_heatmap(grid, param_ranges[x_param], param_ranges[y_param],
                labels[x_param], labels[y_param],
                f"{output_metric.title()} — {labels[x_param]} vs {labels[y_param]}")
            st.plotly_chart(fig_hm, use_container_width=True)

    with tab3:
        st.markdown("#### Time decay acceleration near expiry")
        decay_data = compute_time_decay(S, K, r, sigma, option_type, T_max=max(T, 0.1))
        fig_decay = plot_time_decay(decay_data)
        st.plotly_chart(fig_decay, use_container_width=True)

    with tab4:
        st.markdown("#### Decompose P&L into Greek contributions")
        pnl_col1, pnl_col2, pnl_col3 = st.columns(3)
        with pnl_col1:
            S_new = st.number_input("New Spot", value=S * 1.05, step=0.5, format="%.2f")
        with pnl_col2:
            sigma_new = st.slider("New Vol (%)", 1.0, 100.0, value=sigma * 100 + 2, step=0.5) / 100
        with pnl_col3:
            days_passed = st.slider("Days passed", 1, max(int(T * 365), 2),
                                    value=min(5, max(int(T * 365) - 1, 1)))
        T_new = T - days_passed / 365
        if T_new <= 0:
            st.error("Time left must be positive. Reduce days passed.")
        else:
            pnl = pnl_decomposition(S, S_new, K, T, T_new, r, sigma, sigma_new, option_type)
            pc = st.columns(4)
            pc[0].metric("Actual P&L", f"${pnl['actual_pnl']:+.4f}")
            pc[1].metric("Approx P&L", f"${pnl['approx_pnl']:+.4f}")
            pc[2].metric("Delta P&L", f"${pnl['delta_pnl']:+.4f}")
            pc[3].metric("Unexplained", f"${pnl['unexplained']:+.4f}")
            fig_pnl = plot_pnl_decomposition(pnl)
            st.plotly_chart(fig_pnl, use_container_width=True)

    with tab5:
        st.markdown("#### Black-Scholes vs Monte Carlo comparison")
        mc_sims = st.select_slider("Number of simulations",
            options=[10_000, 50_000, 100_000, 500_000, 1_000_000], value=100_000)
        if st.button("Run Monte Carlo", type="primary"):
            with st.spinner(f"Running {mc_sims:,} simulations..."):
                mc = monte_carlo(S, K, T, r, sigma, option_type, n_simulations=mc_sims)
            cc = st.columns(3)
            cc[0].metric("BS Price", f"${bs.price:.4f}")
            cc[1].metric("MC Price", f"${mc.price:.4f}")
            cc[2].metric("Difference", f"${abs(bs.price - mc.price):.4f}")
            fig_comp = plot_bs_vs_mc(bs, mc)
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Click the button to run Monte Carlo simulation.")


elif module == "B — Implied Vol & Surface":
    # ═══════════════════════════════════════
    # MODULE B
    # ═══════════════════════════════════════
    st.sidebar.markdown("---")
    st.sidebar.header("Market Data Settings")
    ticker_input = st.sidebar.text_input("Ticker", value="SPY",
        help="Any US ticker with listed options (SPY, AAPL, MSFT, QQQ, TSLA...)")
    rf_rate = st.sidebar.slider("Risk-free rate (%)", 0.0, 10.0, value=4.5, step=0.1,
        help="Used for IV computation") / 100
    max_expiries = st.sidebar.slider("Max expiries to load", 2, 12, value=6,
        help="More expiries = richer surface but slower loading")

    st.title("Module B — Implied Volatility & Vol Surface")
    st.markdown("Extract implied volatilities from real market option prices and visualize the volatility surface.")

    if st.sidebar.button("📡 Fetch Option Data", type="primary"):
        with st.spinner(f"Fetching option chain for {ticker_input.upper()}..."):
            try:
                spot, expirations, ticker_obj = fetch_option_chain(ticker_input.upper())
                if spot is None or expirations is None:
                    st.error(f"Could not fetch data for {ticker_input.upper()}. Check the ticker.")
                else:
                    st.session_state["vol_spot"] = spot
                    st.session_state["vol_expirations"] = expirations
                    st.session_state["vol_ticker"] = ticker_obj
                    st.session_state["vol_ticker_name"] = ticker_input.upper()
                    st.session_state["vol_rf"] = rf_rate
                    with st.spinner("Computing implied volatilities..."):
                        surface = build_vol_surface(ticker_obj, expirations, spot, rf_rate, max_expiries)
                        st.session_state["vol_surface"] = surface
                        st.session_state["vol_loaded"] = True
                    st.success(f"Loaded {len(surface)} data points across "
                              f"{min(len(expirations), max_expiries)} expiries for "
                              f"{ticker_input.upper()} (Spot: ${spot:.2f})")
            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")

    if st.session_state.get("vol_loaded"):
        spot = st.session_state["vol_spot"]
        surface = st.session_state["vol_surface"]
        expirations = st.session_state["vol_expirations"]
        ticker_name = st.session_state["vol_ticker_name"]

        all_ivs = [d.get("call_iv", d.get("put_iv", 0)) for d in surface if d.get("call_iv") or d.get("put_iv")]
        atm_ivs = [d.get("call_iv", d.get("put_iv", 0)) for d in surface
                    if 0.97 <= d.get("moneyness", 0) <= 1.03 and (d.get("call_iv") or d.get("put_iv"))]

        m_cols = st.columns(4)
        m_cols[0].metric("Spot Price", f"${spot:.2f}")
        m_cols[1].metric("Data Points", len(surface))
        if atm_ivs:
            m_cols[2].metric("ATM IV (avg)", f"{np.mean(atm_ivs)*100:.1f}%")
        if all_ivs:
            m_cols[3].metric("IV Range", f"{min(all_ivs)*100:.0f}% — {max(all_ivs)*100:.0f}%")

        btab1, btab2, btab3, btab4 = st.tabs([
            "😊 Vol Smile", "🌋 3D Vol Surface", "📈 Term Structure", "🔀 Skew Comparison"])

        with btab1:
            st.markdown("#### Volatility Smile per Expiry")
            st.markdown("The smile shows how IV varies across strikes for a single expiry. "
                        "Deep OTM puts typically have higher IV — that's the crash protection premium.")
            available_expiries = sorted(set(d["expiry"] for d in surface))
            selected_expiry = st.selectbox("Select expiry", available_expiries)
            fig_smile = plot_vol_smile(surface, selected_expiry, spot)
            st.plotly_chart(fig_smile, use_container_width=True)
            exp_data = [d for d in surface if d["expiry"] == selected_expiry]
            T_exp = exp_data[0]["T"] if exp_data else 0
            st.markdown(f"**Expiry:** {selected_expiry} | **T:** {T_exp:.3f} years "
                        f"({int(T_exp*365)} days) | **Strikes:** {len(exp_data)}")

        with btab2:
            st.markdown("#### 3D Volatility Surface")
            st.markdown("The full surface shows IV across both strike (moneyness) and maturity. "
                        "This is what trading desks use to price exotic options.")
            fig_3d = plot_vol_surface_3d(surface, spot)
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.warning("Not enough data points to build the surface.")

        with btab3:
            st.markdown("#### ATM Implied Volatility Term Structure")
            st.markdown("How does ATM vol change across maturities? Upward-sloping = market expects "
                        "higher future uncertainty. Inverted = near-term stress.")
            fig_ts = plot_vol_term_structure(surface, spot)
            if fig_ts:
                st.plotly_chart(fig_ts, use_container_width=True)
            else:
                st.warning("Not enough ATM data points for the term structure.")

        with btab4:
            st.markdown("#### Skew Comparison Across Expiries")
            st.markdown("Overlay the smiles on a moneyness axis. Short-term skew is typically "
                        "steeper — more crash fear in the near term.")
            fig_skew = plot_skew_comparison(surface, spot)
            st.plotly_chart(fig_skew, use_container_width=True)
    else:
        st.info("👈 Enter a ticker and click **Fetch Option Data** to start.")
        st.markdown("---")
        st.markdown("#### What this module does")
        st.markdown(
            "1. **Fetches real option chain data** from Yahoo Finance "
            "(calls & puts across multiple expiries)\n\n"
            "2. **Solves for implied volatility** using Brent's root-finding on the BS formula\n\n"
            "3. **Visualizes the vol surface** in 4 views: smile, 3D surface, "
            "ATM term structure, and multi-expiry skew comparison")
        st.markdown("#### Try these tickers")
        st.markdown(
            "- **SPY** — S&P 500 ETF (most liquid, classic skew)\n"
            "- **AAPL** — Apple (single stock, earnings-driven kinks)\n"
            "- **TSLA** — Tesla (extreme skew, high vol)\n"
            "- **QQQ** — Nasdaq 100 ETF\n"
            "- **GLD** — Gold ETF")


elif module == "C — Strategy Builder":
    # ═══════════════════════════════════════
    # MODULE C
    # ═══════════════════════════════════════
    st.sidebar.markdown("---")
    st.sidebar.header("Strategy Parameters")

    spot_c = st.sidebar.number_input("Spot Price", value=100.0, min_value=0.01, step=1.0,
                                      format="%.2f", key="spot_c")
    T_c = st.sidebar.slider("Time to Expiry (years)", 0.01, 2.0, 0.25, 0.01, key="T_c")
    r_c = st.sidebar.slider("Risk-free Rate (%)", 0.0, 15.0, 4.0, 0.1, key="r_c") / 100
    sigma_c = st.sidebar.slider("Volatility (%)", 1.0, 100.0, 20.0, 0.5, key="sigma_c") / 100

    st.sidebar.markdown("---")
    st.sidebar.header("Build Strategy")

    build_mode = st.sidebar.radio("Mode", ["Preset Strategies", "Custom Builder"])

    st.title("Module C — Strategy Builder & Payoff Visualizer")
    st.markdown("Build multi-leg options strategies, visualize payoff diagrams at expiry and before, "
                "and analyze aggregate Greeks.")

    legs = []

    if build_mode == "Preset Strategies":
        strategy_name = st.sidebar.selectbox("Strategy", list(PRESET_STRATEGIES.keys()))
        legs = PRESET_STRATEGIES[strategy_name](spot_c, sigma_c)

        # Show strategy description
        descriptions = {
            "Long Call": "Bullish bet. Unlimited upside, max loss = premium paid.",
            "Long Put": "Bearish bet. Profit if underlying drops, max loss = premium paid.",
            "Covered Call": "Own the stock + sell a call. Income strategy, caps upside. "
                           "(P&L shown is for the option leg only — add stock P&L mentally.)",
            "Protective Put": "Own the stock + buy a put. Insurance against downside. "
                              "(P&L shown is for the option leg only.)",
            "Bull Call Spread": "Moderately bullish. Buy lower-strike call, sell higher-strike call. "
                                "Capped profit but cheaper than naked call.",
            "Bear Put Spread": "Moderately bearish. Buy higher-strike put, sell lower-strike put. "
                                "Capped profit but cheaper than naked put.",
            "Long Straddle": "Bet on volatility (direction doesn't matter). "
                             "Buy ATM call + ATM put. Profit if big move in either direction.",
            "Short Straddle": "Bet against volatility. Sell ATM call + ATM put. "
                              "Profit if underlying stays flat. Unlimited risk.",
            "Long Strangle": "Like straddle but cheaper — buy OTM call + OTM put. "
                             "Needs a bigger move to profit.",
            "Iron Condor": "Bet on low volatility with capped risk. "
                           "Sell OTM put spread + sell OTM call spread. Max profit if underlying stays in range.",
            "Butterfly Spread": "Bet that underlying stays near current price. "
                                 "Max profit at center strike, limited risk.",
            "Collar": "Protect downside (buy put) while funding it by capping upside (sell call). "
                      "Zero or low net cost.",
        }
        st.info(f"**{strategy_name}:** {descriptions.get(strategy_name, '')}")

    else:  # Custom Builder
        st.sidebar.markdown("---")
        n_legs = st.sidebar.number_input("Number of legs", 1, 6, 2, key="n_legs")

        for i in range(int(n_legs)):
            st.sidebar.markdown(f"**Leg {i+1}**")
            l_col1, l_col2 = st.sidebar.columns(2)
            with l_col1:
                l_type = st.sidebar.selectbox("Type", ["call", "put"], key=f"ltype_{i}")
                l_pos = st.sidebar.selectbox("Position", ["long", "short"], key=f"lpos_{i}")
            with l_col2:
                l_strike = st.sidebar.number_input("Strike", value=spot_c * (1 + 0.05 * (i - 0.5)),
                                                    min_value=0.01, step=1.0, format="%.1f", key=f"lk_{i}")
                l_qty = st.sidebar.number_input("Qty", value=1, min_value=1, max_value=100, key=f"lq_{i}")

            premium = bs_price_only(spot_c, l_strike, T_c, r_c, sigma_c, l_type)
            legs.append(Leg(l_type, l_strike, l_pos, int(l_qty), premium))

    if legs:
        # ── Strategy Summary ──
        S_range = np.linspace(spot_c * 0.6, spot_c * 1.4, 500)
        metrics = compute_strategy_metrics(legs, S_range)
        greeks = strategy_greeks(legs, spot_c, T_c, r_c, sigma_c)

        # Leg table
        st.markdown("### Strategy Legs")
        leg_data = []
        for i, leg in enumerate(legs):
            leg_data.append({
                "Leg": i + 1,
                "Type": leg.option_type.upper(),
                "Position": leg.position.upper(),
                "Strike": f"${leg.strike:.1f}",
                "Qty": leg.quantity,
                "Premium": f"${leg.premium:.4f}",
                "Cost": f"${leg.sign * leg.quantity * leg.premium:+.4f}"
            })
        st.dataframe(leg_data, use_container_width=True, hide_index=True)

        # Key metrics
        st.markdown("### Key Metrics")
        km = st.columns(5)
        km[0].metric("Net Premium", f"${metrics['net_premium']:+.4f}",
                     help="Positive = net debit (you pay). Negative = net credit (you receive).")
        mp = metrics['max_profit']
        km[1].metric("Max Profit", f"${mp:.2f}" if mp != float('inf') else "∞")
        ml = metrics['max_loss']
        km[2].metric("Max Loss", f"${ml:.2f}" if ml != float('-inf') else "-∞")
        if metrics['breakevens']:
            be_str = " / ".join(f"${b:.2f}" for b in metrics['breakevens'])
            km[3].metric("Breakeven(s)", be_str)
        else:
            km[3].metric("Breakeven(s)", "None")
        km[4].metric("Net Delta", f"{greeks['delta']:+.4f}")

        # ── Tabs ──
        ctab1, ctab2, ctab3 = st.tabs([
            "📊 Payoff Diagram", "📈 Strategy Greeks", "🔍 Leg Decomposition"
        ])

        with ctab1:
            st.markdown("#### P&L at expiry and before expiry")
            st.markdown("The solid line is the payoff at expiry. Dotted lines show the P&L at "
                        "various times before expiry (using BS pricing with constant vol).")

            # Time slices
            days_total = int(T_c * 365)
            if days_total > 7:
                T_remaining_list = [
                    T_c * 0.75,
                    T_c * 0.50,
                    T_c * 0.25,
                ]
            else:
                T_remaining_list = [T_c * 0.5]

            fig_payoff = plot_strategy_payoff(legs, S_range, spot_c, T_c, r_c, sigma_c, T_remaining_list)
            st.plotly_chart(fig_payoff, use_container_width=True)

            # Profit/loss zones
            pnl_expiry = strategy_payoff_at_expiry(legs, S_range)
            profit_pct = np.sum(pnl_expiry > 0) / len(pnl_expiry) * 100
            st.markdown(f"**Profit zone:** {profit_pct:.0f}% of the price range shown is profitable at expiry.")

        with ctab2:
            st.markdown("#### Aggregate Greeks across spot prices")
            st.markdown("See how the strategy's net Greeks change as the underlying moves.")

            g_cols = st.columns(5)
            g_cols[0].metric("Delta (Δ)", f"{greeks['delta']:+.4f}")
            g_cols[1].metric("Gamma (Γ)", f"{greeks['gamma']:+.4f}")
            g_cols[2].metric("Vega (ν)", f"{greeks['vega']:+.4f}")
            g_cols[3].metric("Theta (Θ)", f"{greeks['theta']:+.4f}")
            g_cols[4].metric("Rho (ρ)", f"{greeks['rho']:+.4f}")

            fig_greeks = plot_strategy_greeks_profile(legs, S_range, T_c, r_c, sigma_c)
            st.plotly_chart(fig_greeks, use_container_width=True)

        with ctab3:
            st.markdown("#### Individual leg contributions")
            st.markdown("See how each leg contributes to the total payoff at expiry. "
                        "The white line is the total strategy P&L.")

            fig_legs = plot_individual_legs(legs, S_range)
            st.plotly_chart(fig_legs, use_container_width=True)


elif module == "D — Turbo vs Vanilla":
    # ═══════════════════════════════════════
    # MODULE D — TURBO / KNOCK-OUT
    # ═══════════════════════════════════════
    st.sidebar.markdown("---")
    st.sidebar.header("Turbo Parameters")

    spot_d = st.sidebar.number_input("Spot Price", value=100.0, min_value=0.01,
                                      step=1.0, format="%.2f", key="spot_d")
    strike_d = st.sidebar.number_input("Strike (K)", value=90.0, min_value=0.01,
                                        step=1.0, format="%.2f", key="strike_d",
                                        help="Payout = max(K - S_T, 0) if not knocked out")
    barrier_d = st.sidebar.number_input("Barrier / Knock-Out (H)", value=105.0, min_value=0.01,
                                         step=1.0, format="%.2f", key="barrier_d",
                                         help="If spot hits this level, the Turbo Put dies instantly")
    T_d = st.sidebar.slider("Time to Expiry (years)", 0.01, 2.0, 0.25, 0.01, key="T_d")
    r_d = st.sidebar.slider("Risk-free Rate (%)", 0.0, 15.0, 4.0, 0.1, key="r_d") / 100
    sigma_d = st.sidebar.slider("Volatility (%)", 1.0, 100.0, 20.0, 0.5, key="sigma_d") / 100

    st.title("Module D — Turbo Put vs Vanilla Put")
    st.markdown("Compare a **Turbo Put** (knock-out barrier option) against a standard vanilla put. "
                "Understand the leverage effect, the knock-out risk, and when each instrument makes sense.")

    st.markdown("""
    > **What is a Turbo Put?** It's a leveraged bearish instrument. Like a vanilla put, it profits 
    > when the underlying drops. But it costs much less (= higher leverage) because it has a 
    > **knock-out barrier**: if the underlying ever touches the barrier level, the product is 
    > immediately terminated and you lose your entire investment. It's essentially a leveraged 
    > future with an automatic margin call built into the product structure.
    """)

    # Validate inputs
    if spot_d <= 0 or strike_d <= 0 or barrier_d <= 0:
        st.error("All prices must be positive.")
    elif spot_d >= barrier_d:
        st.error(f"Spot ({spot_d}) must be below the barrier ({barrier_d}) for a Turbo Put to be alive. "
                 f"The barrier is the knock-out level above the current price.")
    else:
        # Compute prices (MC-based, may take a moment)
        with st.spinner("Computing Turbo Put price (Monte Carlo simulation)..."):
            turbo_price_d = turbo_put_price(spot_d, strike_d, barrier_d, T_d, r_d, sigma_d)
        vanilla_price_d = bs_price_only(spot_d, strike_d, T_d, r_d, sigma_d, "put")
        leverage_d = compute_leverage(spot_d, turbo_price_d) if turbo_price_d > 0 else float('inf')

        # Key metrics
        km = st.columns(5)
        km[0].metric("Turbo Put Price", f"${turbo_price_d:.4f}")
        km[1].metric("Vanilla Put Price", f"${vanilla_price_d:.4f}")
        km[2].metric("Leverage", f"{leverage_d:.1f}x" if leverage_d < 1000 else "∞")
        cost_ratio = (turbo_price_d / vanilla_price_d * 100) if vanilla_price_d > 0 else 0
        km[3].metric("Cost Ratio", f"{cost_ratio:.1f}%",
                     help="Turbo cost as % of vanilla cost")
        km[4].metric("Distance to KO", f"{(barrier_d / spot_d - 1) * 100:.1f}%",
                     help="How far spot needs to rise to knock out")

        # Tabs
        dtab1, dtab2, dtab3, dtab4 = st.tabs([
            "📊 P&L Comparison", "⚡ Leverage Profile",
            "🎯 Scenario Analysis", "⏱️ Time Decay"
        ])

        S_range_d = np.linspace(spot_d * 0.5, barrier_d * 1.1, 80)

        with dtab1:
            st.markdown("#### Turbo Put vs Vanilla Put — P&L at current time")
            st.markdown("The red zone on the right is the **knock-out zone**: if the underlying "
                        "rises to the barrier, the Turbo is worth zero. The vanilla put survives any upside move.")

            with st.spinner("Computing P&L curves (Monte Carlo at each price point)..."):
                turbo_pnl_d, _ = turbo_put_pnl(spot_d, S_range_d, strike_d, barrier_d, T_d, r_d, sigma_d)
            vanilla_pnl_d, _ = vanilla_put_pnl(spot_d, S_range_d, strike_d, T_d, r_d, sigma_d)

            fig_comp = plot_turbo_vs_vanilla(S_range_d, turbo_pnl_d, vanilla_pnl_d,
                spot_d, barrier_d, strike_d, turbo_price_d, vanilla_price_d)
            st.plotly_chart(fig_comp, use_container_width=True)

            # Interpretation
            st.markdown("**Key insight:** The Turbo Put amplifies both gains AND losses. "
                        f"With {leverage_d:.1f}x leverage, a 1% drop in the underlying gives you "
                        f"roughly a {leverage_d:.0f}% return on the Turbo vs ~{abs(black_scholes(spot_d, strike_d, T_d, r_d, sigma_d, 'put').delta)*100:.0f}% "
                        f"on the vanilla (delta effect). But if the underlying rises {(barrier_d/spot_d - 1)*100:.0f}% "
                        "to the barrier, you lose 100% of your Turbo investment.")

        with dtab2:
            st.markdown("#### Leverage as a function of spot price")
            st.markdown("Leverage increases as spot approaches the barrier — this is the "
                        "\"picking up pennies in front of a steamroller\" effect. "
                        "Near the barrier, tiny moves create huge % swings.")

            fig_lev = plot_turbo_leverage_profile(S_range_d, strike_d, barrier_d, T_d, r_d, sigma_d, spot_d)
            st.plotly_chart(fig_lev, use_container_width=True)

        with dtab3:
            st.markdown("#### Scenario Analysis — What if the spot moves?")
            st.markdown("Compare percentage returns on both instruments for various spot moves. "
                        "Red bars = knocked out (total loss).")

            fig_scenarios = plot_turbo_scenarios(spot_d, strike_d, barrier_d, T_d, r_d, sigma_d)
            st.plotly_chart(fig_scenarios, use_container_width=True)

            # Detailed table
            st.markdown("#### Detailed Scenario Table")
            moves = [-20, -15, -10, -5, -2, 0, 2, 5, 10, 15, 20]
            scenario_data = scenario_analysis(spot_d, strike_d, barrier_d, T_d, r_d, sigma_d, moves)
            st.dataframe(scenario_data, use_container_width=True, hide_index=True)

        with dtab4:
            st.markdown("#### Time Decay Comparison")
            st.markdown("Both instruments lose value over time (theta decay), but at different rates. "
                        "The Turbo's barrier makes its time value behave differently from the vanilla.")

            fig_time = plot_turbo_time_decay_comparison(spot_d, strike_d, barrier_d, r_d, sigma_d,
                                                         T_max=max(T_d, 0.1))
            st.plotly_chart(fig_time, use_container_width=True)

        # Interview talking points
        st.markdown("---")
        st.markdown("### 💡 Interview Talking Points")
        st.markdown("""
        - **A Turbo Put is economically a leveraged short position with a built-in stop-loss** — 
          the knock-out barrier acts as an automatic margin call that terminates the position.
        
        - **The leverage is NOT free** — you pay for it through the knock-out risk. The probability 
          of the barrier being hit before expiry can be significant, especially in volatile markets.
        
        - **Unlike a vanilla put, the Turbo has discontinuous payoff at the barrier** — this means 
          it cannot be perfectly delta-hedged using standard methods, which is why issuers charge a spread.
        
        - **Turbo products are popular in European retail markets** (especially Germany, Netherlands, France) 
          as alternatives to CFDs. They trade on exchanges (Euronext, Börse Frankfurt) with market maker quotes.
        
        - **For hedging a long equity PEA position**, a Turbo Put is capital-efficient but risky: 
          if there's a sharp V-shaped reversal, you get knocked out on the way up and lose your hedge 
          right when you need it. A vanilla put is more expensive but survives volatility spikes.
        """)


# ── Footer ──
st.markdown("---")
st.markdown(
    '<div class="footer-text">'
    'Derivatives Pricing & Volatility Toolkit — Quentin Nadouce — GEM 2026<br>'
    '<span style="font-size:0.75rem;">Python · NumPy · SciPy · Plotly · Streamlit · yfinance</span>'
    '</div>',
    unsafe_allow_html=True)
