"""
app.py — Dashboard Streamlit
============================
Dashboard interactif pour le pricing d'options,
les Greeks et la simulation de delta-hedging.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from black_scholes import call_price, put_price
from greeks import all_greeks, delta
from monte_carlo import monte_carlo_price
from delta_hedging import delta_hedging, simuler_prix

# ── Configuration de la page ─────────────────────────────
st.set_page_config(
    page_title="Options Quant Dashboard",
    page_icon="📈",
    layout="wide"
)

# ── Style CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric { background-color: #f0f2f6; border-radius: 8px; padding: 0.5rem; }
    h1 { color: #1e3c72; }
    h2 { color: #2a5298; border-bottom: 2px solid #2a5298; padding-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# ── Titre ────────────────────────────────────────────────
st.title("📈 Options Quant Dashboard")
st.markdown("**Pricing · Greeks · Monte Carlo · Delta-Hedging**")
st.markdown("---")

# ════════════════════════════════════════════════════════
#  SIDEBAR — Paramètres
# ════════════════════════════════════════════════════════
st.sidebar.header("⚙️ Paramètres du modèle")

S0 = st.sidebar.slider("Prix du sous-jacent S (€)", 50, 200, 100, step=1)
K  = st.sidebar.slider("Strike K (€)", 50, 200, 100, step=1)
T  = st.sidebar.slider("Maturité T (années)", 0.1, 3.0, 1.0, step=0.1)
r  = st.sidebar.slider("Taux sans risque r (%)", 0.0, 10.0, 5.0, step=0.1) / 100
sigma = st.sidebar.slider("Volatilité σ (%)", 5.0, 80.0, 20.0, step=0.5) / 100

st.sidebar.markdown("---")
st.sidebar.header("🎲 Monte Carlo")
n_sims = st.sidebar.selectbox("Nombre de simulations", [1_000, 10_000, 50_000, 100_000], index=2)

st.sidebar.markdown("---")
st.sidebar.header("📊 Delta-Hedging")
n_steps = st.sidebar.slider("Nombre de rebalancements", 10, 252, 252, step=10)
seed = st.sidebar.number_input("Seed (scénario)", min_value=0, max_value=999, value=42)

# ════════════════════════════════════════════════════════
#  CALCULS
# ════════════════════════════════════════════════════════
C = call_price(S0, K, T, r, sigma)
P = put_price(S0, K, T, r, sigma)
greeks_call = all_greeks(S0, K, T, r, sigma, "call")
greeks_put  = all_greeks(S0, K, T, r, sigma, "put")
mc_call, se_call = monte_carlo_price(S0, K, T, r, sigma, "call", n_sims)
mc_put,  se_put  = monte_carlo_price(S0, K, T, r, sigma, "put",  n_sims)

# Moneyness
if S0 > K * 1.02:
    moneyness = "🟢 In the Money (ITM)"
elif S0 < K * 0.98:
    moneyness = "🔴 Out of the Money (OTM)"
else:
    moneyness = "🟡 At the Money (ATM)"

# ════════════════════════════════════════════════════════
#  SECTION 1 — Prix & Moneyness
# ════════════════════════════════════════════════════════
st.header("1. Prix des options")
st.markdown(f"**Moneyness :** {moneyness}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Prix CALL (BS)",   f"{C:.4f} €")
col2.metric("Prix PUT (BS)",    f"{P:.4f} €")
col3.metric("Prix CALL (MC)",   f"{mc_call:.4f} €", delta=f"Δ={mc_call-C:+.4f}€")
col4.metric("Prix PUT (MC)",    f"{mc_put:.4f} €",  delta=f"Δ={mc_put-P:+.4f}€")

# Graphique prix vs spot
spots = np.linspace(max(10, S0 - 60), S0 + 60, 300)
calls_curve = [call_price(S, K, T, r, sigma) for S in spots]
puts_curve  = [put_price(S, K, T, r, sigma)  for S in spots]
intrinsic_c = np.maximum(spots - K, 0)
intrinsic_p = np.maximum(K - spots, 0)

fig1, axes = plt.subplots(1, 2, figsize=(14, 4))
fig1.patch.set_facecolor('white')

for ax, curve, intrinsic, color, label in zip(
    axes,
    [calls_curve, puts_curve],
    [intrinsic_c, intrinsic_p],
    ['steelblue', 'tomato'],
    ['CALL', 'PUT']
):
    ax.plot(spots, curve, color=color, linewidth=2.5, label=f'Prix BS {label}')
    ax.plot(spots, intrinsic, '--', color='gray', linewidth=1.5, label='Valeur intrinsèque')
    ax.axvline(K,  color='red',   linestyle=':',  alpha=0.7, label=f'Strike={K}€')
    ax.axvline(S0, color='green', linestyle='--', alpha=0.7, label=f'S={S0}€')
    ax.fill_between(spots, intrinsic, curve, alpha=0.15, color=color, label='Valeur temps')
    ax.set_xlabel('Prix du sous-jacent (€)')
    ax.set_ylabel('Prix option (€)')
    ax.set_title(f'Prix du {label} européen', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
st.pyplot(fig1)
plt.close()

# ════════════════════════════════════════════════════════
#  SECTION 2 — Greeks
# ════════════════════════════════════════════════════════
st.markdown("---")
st.header("2. Greeks")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Delta Δ",  f"{greeks_call['delta']:+.4f}", help="Position en actions à détenir")
col2.metric("Gamma Γ",  f"{greeks_call['gamma']:+.4f}", help="Vitesse de variation du delta")
col3.metric("Vega ν",   f"{greeks_call['vega']:+.4f}",  help="Gain pour +1% de volatilité")
col4.metric("Theta Θ",  f"{greeks_call['theta']:+.4f}", help="Perte par jour (€)")
col5.metric("Rho ρ",    f"{greeks_call['rho']:+.4f}",   help="Gain pour +1% de taux")

# Graphique Greeks
spots2 = np.linspace(max(10, S0 - 60), S0 + 60, 200)
g_delta  = [delta(S, K, T, r, sigma, "call") for S in spots2]
g_gamma  = [all_greeks(S, K, T, r, sigma)["gamma"] for S in spots2]
g_vega   = [all_greeks(S, K, T, r, sigma)["vega"]  for S in spots2]
g_theta  = [all_greeks(S, K, T, r, sigma)["theta"] for S in spots2]

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 8))
fig2.patch.set_facecolor('white')

greek_data = [
    (g_delta, 'Delta (Δ)',  'steelblue', 'Position en actions'),
    (g_gamma, 'Gamma (Γ)',  'green',     'Coût du rebalancement'),
    (g_vega,  'Vega (ν)',   'purple',    'Sensibilité à la vol'),
    (g_theta, 'Theta (Θ)', 'tomato',    'Perte par jour (€)'),
]

for ax, (data, name, color, ylabel) in zip(axes2.flatten(), greek_data):
    ax.plot(spots2, data, color=color, linewidth=2.5)
    ax.axvline(K,  color='red',   linestyle=':', alpha=0.6, label=f'K={K}')
    ax.axvline(S0, color='green', linestyle='--', alpha=0.6, label=f'S={S0}')
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel('Prix sous-jacent (€)')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Greeks du CALL européen', fontsize=13, fontweight='bold')
plt.tight_layout()
st.pyplot(fig2)
plt.close()

# ════════════════════════════════════════════════════════
#  SECTION 3 — Delta-Hedging
# ════════════════════════════════════════════════════════
st.markdown("---")
st.header("3. Simulation Delta-Hedging")

with st.spinner("Simulation en cours..."):
    df = delta_hedging(S0, K, T, r, sigma, n_steps=n_steps)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Prime reçue",          f"{C:.4f} €")
col2.metric("Prix final action",    f"{df['prix_action'].iloc[-1]:.4f} €")
col3.metric("P&L final",            f"{df['pnl'].iloc[-1]:.4f} €")
col4.metric("Erreur moy. absolue",  f"{df['pnl'].abs().mean():.4f} €",
            delta=f"{df['pnl'].abs().mean()/C*100:.1f}% de la prime")

fig3, axes3 = plt.subplots(3, 1, figsize=(14, 11))
fig3.patch.set_facecolor('white')

# Prix
axes3[0].plot(df["step"], df["prix_action"], color='steelblue', linewidth=2, label='Prix action')
axes3[0].axhline(K, color='red', linestyle='--', alpha=0.6, label=f'Strike K={K}€')
axes3[0].set_title('Évolution du prix du sous-jacent', fontweight='bold')
axes3[0].set_ylabel('Prix (€)')
axes3[0].legend()
axes3[0].grid(alpha=0.3)

# Delta
axes3[1].plot(df["step"], df["delta"], color='green', linewidth=2, label='Delta')
axes3[1].fill_between(df["step"], 0, df["delta"], alpha=0.15, color='green')
axes3[1].set_title('Delta — position en actions détenues', fontweight='bold')
axes3[1].set_ylabel('Delta')
axes3[1].set_ylim(-0.05, 1.05)
axes3[1].legend()
axes3[1].grid(alpha=0.3)

# P&L
axes3[2].plot(df["step"], df["pnl"], color='purple', linewidth=2, label='P&L')
axes3[2].axhline(0, color='red', linestyle='--', alpha=0.6)
axes3[2].fill_between(df["step"], 0, df["pnl"],
                      where=[p >= 0 for p in df["pnl"]],
                      alpha=0.2, color='green', label='P&L positif')
axes3[2].fill_between(df["step"], 0, df["pnl"],
                      where=[p < 0  for p in df["pnl"]],
                      alpha=0.2, color='red', label='P&L négatif')
axes3[2].set_title('P&L de la stratégie de couverture', fontweight='bold')
axes3[2].set_ylabel('P&L (€)')
axes3[2].set_xlabel('Jours')
axes3[2].legend()
axes3[2].grid(alpha=0.3)

for ax in axes3:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle(f'Delta-Hedging — {n_steps} rebalancements', fontsize=13, fontweight='bold')
plt.tight_layout()
st.pyplot(fig3)
plt.close()

# ════════════════════════════════════════════════════════
#  FOOTER
# ════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray; font-size:0.85em'>
    Projet 01 — Pricing & Delta-Hedging · Black-Scholes · Monte Carlo<br>
    Master Mathématiques & Applications — Sorbonne Université
</div>
""", unsafe_allow_html=True)
