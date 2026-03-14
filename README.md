#  Projet 01 — Pricing d'Options & Delta-Hedging

> **Tolno Andre Sepe** · L3 Génie Civil → Candidat M1 Maths & Applications (Sorbonne)  
> Spécialisation visée : **M2 Quant Trading**

##  Objectif
Implémentation complète du modèle **Black-Scholes** pour le pricing d'options européennes,
le calcul des Greeks, la simulation Monte Carlo et une stratégie de **delta-hedging dynamique**.

##  Structure
```
projet_01_options/
├── src/
│   ├── black_scholes.py   # Pricing analytique BS
│   ├── greeks.py          # Delta, Gamma, Vega, Theta, Rho
│   ├── monte_carlo.py     # Simulation Monte Carlo
│   └── delta_hedging.py   # Stratégie de couverture
├── notebooks/
│   └── 01_pricing.ipynb   # Notebook Jupyter complet
├── dashboard/
│   └── app.py             # Dashboard Streamlit interactif
└── report/
    └── rapport_final.pdf  # Write-up complet
```

##  Lancer le dashboard
```bash
pip install -r requirements.txt
cd dashboard
streamlit run app.py
```

##  Résultats clés
| Métrique | Valeur |
|----------|--------|
| Prix CALL (BS) | 10.4506 € |
| Prix PUT (BS) | 5.5735 € |
| Erreur MC (100k simul.) | < 0.03 € |
| Erreur de couverture | 5.4% de la prime |

##  Stack technique
`Python 3.14` · `NumPy` · `SciPy` · `Pandas` · `Matplotlib` · `Streamlit` · `Jupyter`