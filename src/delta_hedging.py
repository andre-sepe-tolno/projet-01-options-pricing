"""
delta_hedging.py
================
Simulation d'une stratégie de delta-hedging dynamique.

Idée : un trader vend un call et se couvre en achetant
delta actions à chaque période. On mesure le P&L final
et l'erreur de couverture.
"""

import numpy as np
import pandas as pd
from black_scholes import call_price
from greeks import delta


def simuler_prix(S0, r, sigma, T, n_steps, seed=42):
    """
    Simule un chemin de prix avec le mouvement brownien géométrique.
    C'est le modèle de Black-Scholes pour le sous-jacent.

    dS = S * (r*dt + sigma*dW)
    """
    np.random.seed(seed)
    dt = T / n_steps
    rendements = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.randn(n_steps)
    prix = S0 * np.exp(np.cumsum(rendements))
    return np.insert(prix, 0, S0)  # on rajoute S0 au début


def delta_hedging(S0, K, T, r, sigma, n_steps=252):
    """
    Simule le delta-hedging jour par jour.

    À chaque pas :
      1. Calcule le delta de l'option
      2. Ajuste la position en actions pour matcher le delta
      3. Finance l'ajustement par emprunt/prêt
      4. Enregistre tout dans un tableau

    Retourne un DataFrame avec l'historique complet.
    """
    dt = T / n_steps
    prix = simuler_prix(S0, r, sigma, T, n_steps)

    # ── Initialisation ───────────────────────────────────
    # À t=0 : on vend le call, on reçoit sa prime
    prime = call_price(S0, K, T, r, sigma)
    delta_0 = delta(S0, K, T, r, sigma, "call")

    # On achète delta_0 actions financées par emprunt
    position_actions = delta_0          # nb d'actions détenues
    cash = prime - delta_0 * S0         # cash = prime reçue - coût des actions

    historique = []

    for i in range(n_steps + 1):
        S = prix[i]
        t_restant = T - i * dt
        t_restant = max(t_restant, 1e-10)  # évite division par zéro

        # Valeur actuelle de l'option (mark-to-market)
        valeur_option = call_price(S, K, t_restant, r, sigma)

        # Valeur du portefeuille de couverture
        valeur_portefeuille = position_actions * S + cash * np.exp(r * i * dt)

        # P&L : portefeuille de couverture - valeur option
        pnl = valeur_portefeuille - valeur_option

        historique.append({
            "step":              i,
            "prix_action":       round(S, 4),
            "delta":             round(delta(S, K, t_restant, r, sigma, "call"), 4),
            "valeur_option":     round(valeur_option, 4),
            "position_actions":  round(position_actions, 4),
            "cash":              round(cash, 4),
            "valeur_portefeuille": round(valeur_portefeuille, 4),
            "pnl":               round(pnl, 4),
        })

        # ── Rebalancement (sauf au dernier pas) ──────────
        if i < n_steps:
            S_next = prix[i + 1]
            t_next = T - (i + 1) * dt
            t_next = max(t_next, 1e-10)

            nouveau_delta = delta(S_next, K, t_next, r, sigma, "call")
            ajustement = nouveau_delta - position_actions

            # Coût du rebalancement (positif = achat, négatif = vente)
            cash -= ajustement * S_next
            position_actions = nouveau_delta

    return pd.DataFrame(historique)


# ── Test rapide ──────────────────────────────────────────
if __name__ == "__main__":
    S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
    n_steps = 252  # rebalancement quotidien sur 1 an

    print("Simulation du delta-hedging en cours...")
    df = delta_hedging(S0, K, T, r, sigma, n_steps)

    print("\n" + "=" * 55)
    print("  Delta-Hedging — Résultats (S=K=100, σ=20%, 252 pas)")
    print("=" * 55)
    print(f"  Prix initial du call  : {call_price(S0, K, T, r, sigma):.4f} €")
    print(f"  Prix final du S       : {df['prix_action'].iloc[-1]:.4f} €")
    print(f"  P&L final             : {df['pnl'].iloc[-1]:.4f} €")
    print(f"  P&L moyen (abs)       : {df['pnl'].abs().mean():.4f} €")
    print(f"  P&L max               : {df['pnl'].max():.4f} €")
    print(f"  P&L min               : {df['pnl'].min():.4f} €")
    print("=" * 55)
    print("\n  Aperçu des 5 premières lignes :")
    print(df[["step", "prix_action", "delta",
              "valeur_option", "pnl"]].head().to_string(index=False))
    print("\n  Aperçu des 5 dernières lignes :")
    print(df[["step", "prix_action", "delta",
              "valeur_option", "pnl"]].tail().to_string(index=False))