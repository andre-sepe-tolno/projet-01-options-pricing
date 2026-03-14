"""
black_scholes.py
================
Pricing analytique d'options européennes (call & put)
via la formule de Black-Scholes (1973).

Paramètres :
    S     : Prix actuel du sous-jacent
    K     : Strike (prix d'exercice)
    T     : Temps à maturité en années
    r     : Taux sans risque (ex: 0.05 = 5%)
    sigma : Volatilité (ex: 0.20 = 20%)
"""

import numpy as np
from scipy.stats import norm


def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def _d2(S, K, T, r, sigma):
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S, K, T, r, sigma):
    """Prix d'un call européen — formule Black-Scholes."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, T, r, sigma):
    """Prix d'un put européen — formule Black-Scholes."""
    if T <= 0:
        return max(K - S, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def verify_put_call_parity(S, K, T, r, sigma):
    """Vérifie C - P = S - K*exp(-rT)  (relation d'arbitrage fondamentale)."""
    C = call_price(S, K, T, r, sigma)
    P = put_price(S, K, T, r, sigma)
    return abs((C - P) - (S - K * np.exp(-r * T))) < 1e-8


# ── Test rapide ──────────────────────────────────────────
if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    C = call_price(S, K, T, r, sigma)
    P = put_price(S, K, T, r, sigma)

    print("=" * 45)
    print("  Black-Scholes — ATM (S=K=100, T=1an, σ=20%)")
    print("=" * 45)
    print(f"  Prix du CALL   = {C:.4f} €")
    print(f"  Prix du PUT    = {P:.4f} €")
    print(f"  Parité put-call: {'✓ OK' if verify_put_call_parity(S, K, T, r, sigma) else '✗ ERREUR'}")
    print("=" * 45)