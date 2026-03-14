"""
greeks.py
=========
Calcul des sensibilités (Greeks) d'options européennes.

    Delta (Δ) : sensibilité au prix du sous-jacent
    Gamma (Γ) : sensibilité du delta (courbure)
    Vega  (ν) : sensibilité à la volatilité
    Theta (Θ) : perte de valeur par jour (time decay)
    Rho   (ρ) : sensibilité au taux d'intérêt
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def delta(S, K, T, r, sigma, option_type="call"):
    """
    Call : entre 0 et 1   → combien d'actions acheter pour couvrir
    Put  : entre -1 et 0
    """
    if T <= 0:
        return (1.0 if S > K else 0.0) if option_type == "call" else (-1.0 if S < K else 0.0)
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0


def gamma(S, K, T, r, sigma):
    """
    Vitesse de changement du delta.
    Même valeur pour call et put.
    """
    if T <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """
    Gain/perte pour +1% de volatilité.
    Même valeur pour call et put.
    """
    if T <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100


def theta(S, K, T, r, sigma, option_type="call"):
    """
    Perte de valeur par jour calendaire.
    Généralement négatif (l'option perd de la valeur avec le temps).
    """
    if T <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        return (common - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        return (common + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365


def rho(S, K, T, r, sigma, option_type="call"):
    """Gain/perte pour +1% de taux d'intérêt."""
    if T <= 0:
        return 0.0
    _, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


def all_greeks(S, K, T, r, sigma, option_type="call"):
    """Retourne tous les Greeks dans un dictionnaire."""
    return {
        "delta": delta(S, K, T, r, sigma, option_type),
        "gamma": gamma(S, K, T, r, sigma),
        "vega":  vega(S, K, T, r, sigma),
        "theta": theta(S, K, T, r, sigma, option_type),
        "rho":   rho(S, K, T, r, sigma, option_type),
    }


# ── Test rapide ──────────────────────────────────────────
if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    print("=" * 50)
    print("  Greeks — ATM (S=K=100, T=1an, σ=20%)")
    print("=" * 50)

    for opt in ["call", "put"]:
        g = all_greeks(S, K, T, r, sigma, opt)
        print(f"\n  {opt.upper()} :")
        print(f"    Delta  (Δ) = {g['delta']:+.4f}   ← titres à détenir pour couvrir")
        print(f"    Gamma  (Γ) = {g['gamma']:+.4f}   ← vitesse de variation du delta")
        print(f"    Vega   (ν) = {g['vega']:+.4f}   ← gain si vol monte de 1%")
        print(f"    Theta  (Θ) = {g['theta']:+.4f}   ← perte par jour")
        print(f"    Rho    (ρ) = {g['rho']:+.4f}   ← gain si taux monte de 1%")

    print("\n" + "=" * 50)
    print("  Intuition clé :")
    print("  Delta call ≈ 0.64 → pour couvrir 1 call vendu,")
    print("  tu achètes 0.64 action en ce moment.")
    print("=" * 50)