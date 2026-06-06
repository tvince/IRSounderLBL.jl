"""
Plot the per-layer breakdown of Julia's cross-section pipeline at
2364.105 cm⁻¹ for two atmospheres (Julia native vs LBLRTM-TAPE6).

Six panels (all vs layer mid-altitude z):
  1. T(z)                       — level temperature profile (truth)
  2. σ(2364.105 cm⁻¹)            — per-layer summed CO₂ cross-section
  3. N_CO₂(z)                   — per-layer column density
  4. τ_layer(z)                 — per-layer optical depth at line center
  5. τ_TOA_cum(z)               — cumulative τ from TOA, with the τ=1
                                  saturation level marked
  6. relative Δ(LBL atm / Julia atm) for σ, N, τ_layer

Usage:
  python scripts/plot_2364_line_instrumentation.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "data/lblrtm/instrument_2364_line.csv"
PROFILE = "data/afgl_us_standard_50lev.csv"

D = np.genfromtxt(CSV, delimiter=",", names=True)
z = 0.5 * (D["z_from_km"] + D["z_to_km"])

# also pull the level T profile (for the top panel)
prof = np.genfromtxt(PROFILE, delimiter=",", names=True)
z_lev = prof["z_km"]; T_lev = prof["T_K"]

fig, ax = plt.subplots(3, 2, figsize=(13, 11), sharex=True)

# ── (1) T(z) ─────────────────────────────────────────────────────
ax[0,0].plot(z_lev, T_lev, color="k", lw=1.2, marker="o", ms=2)
ax[0,0].axhline(187, color="grey", lw=0.5, ls="--",
                label="mesopause T = 187 K")
ax[0,0].set_ylabel("T [K]")
ax[0,0].set_title("Atmosphere T(z)")
ax[0,0].legend(fontsize=8)
ax[0,0].grid(True, alpha=0.3)

# ── (2) σ(2364.105) per layer ────────────────────────────────────
ax[0,1].plot(z, D["sigma_jul"], color="tab:red",  lw=1.0, marker="o", ms=3,
             label="Julia atm")
ax[0,1].plot(z, D["sigma_lbl"], color="tab:blue", lw=1.0, marker="o", ms=3,
             alpha=0.7, label="LBLRTM atm + Julia formulas")
ax[0,1].set_yscale("log")
ax[0,1].set_ylabel("σ(2364.105) [cm²/molec]")
ax[0,1].set_title("Per-layer summed CO$_2$ cross-section at the line core")
ax[0,1].legend(fontsize=8)
ax[0,1].grid(True, alpha=0.3, which="both")

# ── (3) N_CO2 per layer ──────────────────────────────────────────
ax[1,0].plot(z, D["N_jul"], color="tab:red",  lw=1.0, marker="o", ms=3,
             label="Julia atm")
ax[1,0].plot(z, D["N_lbl"], color="tab:blue", lw=1.0, marker="o", ms=3,
             alpha=0.7, label="LBLRTM atm")
ax[1,0].set_yscale("log")
ax[1,0].set_ylabel("N$_{CO_2}$ [molec cm$^{-2}$]")
ax[1,0].set_title("Per-layer CO$_2$ column density")
ax[1,0].legend(fontsize=8)
ax[1,0].grid(True, alpha=0.3, which="both")

# ── (4) τ_layer (= σ × N) ────────────────────────────────────────
ax[1,1].plot(z, D["tau_layer_jul"], color="tab:red",  lw=1.0, marker="o", ms=3,
             label="Julia atm")
ax[1,1].plot(z, D["tau_layer_lbl"], color="tab:blue", lw=1.0, marker="o", ms=3,
             alpha=0.7, label="LBLRTM atm")
ax[1,1].set_yscale("log")
ax[1,1].set_ylabel(r"$\tau_{layer}$")
ax[1,1].set_title(r"Per-layer optical depth at 2364.105 cm$^{-1}$")
ax[1,1].legend(fontsize=8)
ax[1,1].grid(True, alpha=0.3, which="both")

# ── (5) Cumulative τ_TOA(z) ──────────────────────────────────────
ax[2,0].plot(z, D["tau_above_jul"], color="tab:red",  lw=1.5, marker="o", ms=3,
             label="Julia atm")
ax[2,0].plot(z, D["tau_above_lbl"], color="tab:blue", lw=1.5, marker="o", ms=3,
             alpha=0.7, label="LBLRTM atm")
ax[2,0].axhline(1.0, color="grey", lw=0.7, ls="--", label="τ = 1 saturation")
ax[2,0].set_yscale("log")
ax[2,0].set_ylabel(r"$\tau_{above}$ (cum from TOA)")
ax[2,0].set_title(r"Cumulative $\tau$ above each layer top at 2364.105 cm$^{-1}$")
ax[2,0].set_xlabel("layer mid-altitude [km]")
ax[2,0].legend(fontsize=8, loc="lower left")
ax[2,0].grid(True, alpha=0.3, which="both")

# ── (6) Relative Δ (LBL atm − Julia atm) / Julia atm ─────────────
def reld(a, b):
    return np.where(a != 0, (b - a) / a * 100.0, np.nan)

ax[2,1].plot(z, reld(D["sigma_jul"],     D["sigma_lbl"]),
             color="tab:purple", lw=1.2, marker="o", ms=3, label="Δσ %")
ax[2,1].plot(z, reld(D["N_jul"],         D["N_lbl"]),
             color="tab:green",  lw=1.2, marker="o", ms=3, label="ΔN_CO₂ %")
ax[2,1].plot(z, reld(D["tau_layer_jul"], D["tau_layer_lbl"]),
             color="tab:orange", lw=1.2, marker="o", ms=3, label="Δτ_layer %")
ax[2,1].axhline(0, color="grey", lw=0.5)
ax[2,1].set_ylabel("Δ (LBL atm − Julia atm) [%]")
ax[2,1].set_title("Per-layer relative shift when LBLRTM atmosphere is substituted")
ax[2,1].set_xlabel("layer mid-altitude [km]")
ax[2,1].set_ylim(-15, 15)
ax[2,1].legend(fontsize=8)
ax[2,1].grid(True, alpha=0.3)

fig.suptitle(
    "Per-layer instrumentation of Julia's CO$_2$ cross-section pipeline at "
    "2364.105 cm$^{-1}$\n"
    "(Julia atm vs LBLRTM TAPE6 atm, identical Julia formulas)\n"
    "Predicted line-core BT: Julia atm 240.34 K (matches FWD), "
    "LBLRTM atm 242.14 K  →  atmosphere alone gives only +1.79 K "
    "(wrong direction for the 7 K LBLRTM gap)",
    fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = "data/lblrtm/instrument_2364_line.png"
plt.savefig(out, dpi=140)
print(f"wrote {out}")
