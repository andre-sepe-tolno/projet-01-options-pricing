"""
monte_carlo.py
==============
Pricing d'options par simulation Monte Carlo.

Principe : simuler N chemins de prix possibles,
calculer le payoff à maturité sur chaque chemin,
puis prendre la moyenne actualisée.

C'est une alternative à la formule analytique BS
utile pour des options complexes ou exotiques.
"""

import numpy as np
from black_scholes import call_price, put_price


def monte_carlo_price(S, K, T, r, sigma,
                      option_type="call",
                      n_simulations=100_000,
                      seed=42):
    """
    Prix d'une option européenne par Monte Carlo.

    On simule N valeurs finales du sous-jacent S_T :
        S_T = S * exp((r - σ²/2)*T + σ*√T*Z)
    avec Z ~ N(0,1)

    Puis on calcule le payoff moyen actualisé.

    Returns:
        price   : prix estimé
        std_err : erreur standard de l'estimation
    """
    np.random.seed(seed)

    # Simulation vectorisée de N valeurs terminales
    Z = np.random.randn(n_simulations)
    S_T = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Payoff à maturité
    if option_type == "call":
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    # Actualisation
    prix_actualises = np.exp(-r * T) * payoffs

    price   = prix_actualises.mean()
    std_err = prix_actualises.std() / np.sqrt(n_simulations)

    return price, std_err


def comparer_bs_mc(S, K, T, r, sigma, n_simulations=100_000):
    """
    Compare le prix Black-Scholes analytique vs Monte Carlo.
    Retourne un dictionnaire avec les deux résultats.
    """
    # Prix analytiques BS
    bs_call = call_price(S, K, T, r, sigma)
    bs_put  = put_price(S, K, T, r, sigma)

    # Prix Monte Carlo
    mc_call, se_call = monte_carlo_price(S, K, T, r, sigma, "call", n_simulations)
    mc_put,  se_put  = monte_carlo_price(S, K, T, r, sigma, "put",  n_simulations)

    return {
        "call": {
            "BS":       bs_call,
            "MC":       mc_call,
            "erreur":   abs(mc_call - bs_call),
            "std_err":  se_call,
        },
        "put": {
            "BS":       bs_put,
            "MC":       mc_put,
            "erreur":   abs(mc_put - bs_put),
            "std_err":  se_put,
        }
    }


def convergence_mc(S, K, T, r, sigma, option_type="call"):
    """
    Montre comment le prix MC converge vers BS
    en augmentant le nombre de simulations.

    Utile pour illustrer la loi des grands nombres.
    """
    bs = call_price(S, K, T, r, sigma) if option_type == "call" else put_price(S, K, T, r, sigma)

    resultats = []
    for n in [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]:
        prix, se = monte_carlo_price(S, K, T, r, sigma, option_type, n)
        resultats.append({
            "n_simulations": n,
            "prix_MC":       round(prix, 6),
            "prix_BS":       round(bs, 6),
            "erreur_abs":    round(abs(prix - bs), 6),
            "std_err":       round(se, 6),
        })

    return resultats


# ── Test rapide ──────────────────────────────────────────
if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    print("=" * 58)
    print("  Monte Carlo vs Black-Scholes (100 000 simulations)")
    print("=" * 58)

    res = comparer_bs_mc(S, K, T, r, sigma)
    for opt in ["call", "put"]:
        r_ = res[opt]
        print(f"\n  {opt.upper()} :")
        print(f"    Prix BS          = {r_['BS']:.6f} €")
        print(f"    Prix MC          = {r_['MC']:.6f} €")
        print(f"    Erreur absolue   = {r_['erreur']:.6f} €")
        print(f"    Erreur standard  = {r_['std_err']:.6f} €")

    print("\n" + "=" * 58)
    print("  Convergence MC (call) :")
    print("=" * 58)
    print(f"  {'N simul':>10}  {'Prix MC':>10}  {'Prix BS':>10}  {'Erreur':>10}")
    print("  " + "-" * 46)
    for row in convergence_mc(S, K, T, r, sigma, "call"):
        print(f"  {row['n_simulations']:>10,}  "
              f"{row['prix_MC']:>10.4f}  "
              f"{row['prix_BS']:>10.4f}  "
              f"{row['erreur_abs']:>10.6f}")